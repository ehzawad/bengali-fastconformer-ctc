#!/usr/bin/env python
"""Fetch the base checkpoint the recipe starts from and register it.

Downloads `stt_en_fastconformer_ctc_large.nemo` from nvidia/stt_en_fastconformer_ctc_large
(NVIDIA, CC-BY-4.0) into experiments/inits/, records its SHA-256 in
experiments/inits/INITS.json, and prints the two config lines to set. The trainer
verifies the checkpoint's SHA-256 prefix at start-up (paths.init_N_sha256_prefix),
so a different snapshot is refused rather than silently trained from.

    python bnasr/fetch_init.py [--dest experiments/inits]
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPO_ID, FILE = "nvidia/stt_en_fastconformer_ctc_large", "stt_en_fastconformer_ctc_large.nemo"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(REPO / "experiments/inits"))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(REPO_ID, FILE)
    dst = dest / FILE
    if not dst.exists():
        shutil.copy(src, dst)
    sha = sha256_file(dst)
    reg = dest / "INITS.json"
    inits = json.loads(reg.read_text()) if reg.exists() else {}
    inits["N"] = {"repo": REPO_ID, "file": str(dst.relative_to(REPO)), "sha256": sha,
                  "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "licence": "CC-BY-4.0 (NVIDIA)"}
    reg.write_text(json.dumps(inits, indent=1))
    print(f"init N: {dst}  sha256 {sha}")
    print("set in configs/ctc_from_nvidia.yaml:")
    print(f"  paths.init_N: {dst.relative_to(REPO)}")
    print(f"  paths.init_N_sha256_prefix: \"{sha[:8]}\"")


if __name__ == "__main__":
    main()
