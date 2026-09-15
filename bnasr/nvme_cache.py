#!/usr/bin/env python3
"""this recipe training-audio cache on the NVMe root disk (finding 2026-09-06).

Why: /mnt/sdb is ONE rotational 16 TB HDD (~135 random reads/s). Training reads 252 clips per optimizer update, so the
expanded corpus (227 GB PCM16 for arm N, far beyond the 96 GB page cache that let the baseline run at 1,100 steps/h) is
IOPS-bound: the drills stalled at ~250 steps/h with the GPU at 20-40 %. The NVMe (/dev/nvme0n1p3, 279 GB free) has
~100k IOPS. Storing every training clip of the largest arm as FLAC (lossless; libsndfile decodes it in-process, exactly
the container IndicVoices already used in the baseline) needs ~125 GB and makes the input pipeline GPU-bound again.

What this tool does
  * reads the union of arm manifests (default: arm_N, a superset of every other arm), sorted by source path so the HDD
    is read as sequentially as possible, ≤8 workers, ionice/nice set by the launcher;
  * writes <cache>/<same relative path>.flac, verifies the decoded sample count equals the source, and records a
    mapping row {src, dst, n_samples, sha256_pcm_head} so a later audit can prove the cache is lossless;
  * `--rewrite-manifests` then writes arm_<name>.nvme.jsonl / panel manifests with audio_filepath pointing at the cache.
    The frozen arm manifests on /mnt/sdb remain the registered artefacts; the .nvme.jsonl files are their verbatim
    path-remapped copies (same rows, same order, same text, same durations) and their checksums are recorded next to them.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/p4cache")


def convert(args):
    src, dst = args
    try:
        if os.path.exists(dst):
            info = sf.info(dst)
            return src, dst, info.frames, None, "exists"
        # one sequential read syscall per file (200 KB typical) instead of libsndfile's chunked reads: on a rotational
        # disk each extra read is a seek, and IOPS, not bandwidth, is the limit
        with open(str(REPO / src), "rb") as fh:
            blob = fh.read()
        import io
        x, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if x.ndim > 1:
            x = x[:, 0]
        if sr != 16000:
            return src, dst, 0, f"sr={sr}", "error"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".tmp"
        sf.write(tmp, x, 16000, format="FLAC", subtype="PCM_16")
        y, _ = sf.read(tmp, dtype="int16")
        if y.shape[0] != x.shape[0] or not np.array_equal(x, y):
            os.remove(tmp)
            return src, dst, 0, "roundtrip_mismatch", "error"
        os.replace(tmp, dst)
        return src, dst, int(x.shape[0]), hashlib.sha256(x[:16000].tobytes()).hexdigest()[:16], "ok"
    except Exception as e:  # noqa: BLE001
        return src, dst, 0, f"{type(e).__name__}", "error"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", default=[str(REPO / "experiments/corpus/arm_N.jsonl")])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rewrite-manifests", nargs="*", default=None,
                    help="manifests to remap onto the cache (writes <name>.nvme.jsonl next to each)")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    if a.rewrite_manifests is not None:
        for m in a.rewrite_manifests:
            m = Path(m); out = m.with_suffix(".nvme.jsonl"); n = 0; missing = 0
            with open(out, "w", encoding="utf-8") as fh:
                for line in open(m, encoding="utf-8"):
                    r = json.loads(line)
                    dst = cache / (Path(r["audio_filepath"]).with_suffix(".flac"))
                    if not dst.exists():
                        missing += 1; continue
                    r["audio_filepath"] = str(dst); fh.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
            sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
            print(f"{out.name}: {n} rows, {missing} missing in cache, sha256 {sha[:16]}")
            assert missing == 0, f"{missing} rows missing from the cache for {m}"
        return
    srcs = set()
    for m in a.manifests:
        for line in open(m, encoding="utf-8"):
            srcs.add(json.loads(line)["audio_filepath"])
    jobs = sorted((s, str(cache / Path(s).with_suffix(".flac"))) for s in srcs)
    print(f"{len(jobs)} clips to cache under {cache}", flush=True)
    t0 = time.time(); stats = {"ok": 0, "exists": 0, "error": 0}; errs = []
    with open(cache / "cache_map.tsv", "a", encoding="utf-8") as fh, ProcessPoolExecutor(a.workers) as ex:
        for i, (src, dst, n, meta, st) in enumerate(ex.map(convert, jobs, chunksize=64), 1):
            stats[st] += 1
            if st == "ok":
                fh.write(f"{src}\t{dst}\t{n}\t{meta}\n")
            elif st == "error":
                errs.append((src, meta))
            if i % 50000 == 0:
                print(f"  {i}/{len(jobs)} {stats} {time.time()-t0:.0f}s", flush=True)
    (cache / "cache_report.json").write_text(json.dumps({"clips": len(jobs), "stats": stats, "errors": errs[:200],
                                                         "wall_s": round(time.time() - t0)}, indent=1))
    print(json.dumps({"clips": len(jobs), "stats": stats, "n_errors": len(errs)}))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
