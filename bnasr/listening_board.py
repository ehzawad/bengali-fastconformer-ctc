#!/usr/bin/env python3
"""SPEC §3 G0.6 listening board: the owner's human label audit, as one HTML file.

What G0.6 actually asks for
---------------------------
SPEC §3 G0.6: "listening board of 100 uniformly random `macro_filtered` clips +
20 worst by the baseline disagreement (+ 50 random `madasr` clips). Material error =
>10 % WER of the reference vs what is heard; catastrophic = wrong pairing /
non-speech / gross truncation. **Pass: ≤5 material / 100 random and 0
catastrophic anywhere.**" It also rules that machine disagreement "is reported
as *ASR-disagreement*, not label noise" — the 20 worst clips are an enrichment
sample, deliberately not uniform, and they therefore contribute to the
catastrophic clause but NEVER to the ≤5/100 material rate. `--score` enforces
that separation; it is the whole reason the two strata are labelled in the
sample file rather than pooled.

Why a single self-contained HTML file
-------------------------------------
The audit is performed by the owner, not by this process, and the artefact has
to survive being opened from anywhere with no server, no repo checkout and no
network. So each clip's audio is transcoded to 32 kbps mono MP3 and embedded as
a base64 ``data:`` URI; the page is one file with no external reference of any
kind. 170 clips of ~6 s at 32 kbps is ~4 MB of audio and ~5.5 MB of base64,
under the 12 MB budget the orchestrator set, and ``--max-mb`` asserts it rather
than trusting the arithmetic.

The page carries NO absolute path. Clip ids and repo-relative paths only: the
board is a file that gets emailed, and the audit does not need the box's
directory layout to travel with it.

Reading the diff is the task, so the diff has to be honest
----------------------------------------------------------
The reference is shown on the FROZEN SURFACE (``scoring.frozen_norm``), the
same text the WER is computed on, because a reference displayed with
punctuation next to a WER computed without it invites the auditor to explain
away a real error as a comma. The the baseline hypothesis is word-diffed against it
with ``difflib`` and marked up as insert/delete/replace. And the WER shown is
the same ``wer_counts`` every other this recipe number uses.

The verdicts round-trip as JSON. The board writes ``{clip_id: verdict}`` into a
textarea for the owner to copy back; ``--score`` reads that JSON and applies the
G0.6 rule. Nothing about the pass/fail decision is computed in the browser: the
tally in the page is a progress aid, and the verdict of record comes from
``--score`` running against the sample file it was built from, which is how a
half-finished or hand-edited paste is caught instead of accepted.

Usage
-----
    # smoke (macro_validated is ready now; macro_filtered arrives later)
    CUDA_VISIBLE_DEVICES=GPU-3ce8... bnasr/listening_board.py build \
        --random-manifest data4/macro/manifest_macro_validated.jsonl \
        --n-random 20 --n-worst 5 --out experiments/audit/board_smoke.html

    # the real G0.6 board, once macro_filtered and madasr land
    CUDA_VISIBLE_DEVICES=GPU-3ce8... bnasr/listening_board.py build \
        --random-manifest data4/macro/manifest_macro_filtered.jsonl \
        --extra-manifest data4/respin/manifest_madasr.jsonl \
        --out experiments/audit/board_g06.html

    bnasr/listening_board.py score \
        --sample experiments/audit/board_g06.sample.json \
        --verdicts verdicts_from_owner.json
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import (  # noqa: E402
    anonymize_process, frozen_norm, load_manifest, pin_gpu, sha256_file,
    wer_counts,
)

anonymize_process()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import base64  # noqa: E402
import difflib  # noqa: E402
import html as htmllib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants (SPEC §3 G0.6), asserted against the text of the SPEC.
# ---------------------------------------------------------------------------

# "100 uniformly random macro_filtered clips + 20 worst by the baseline disagreement
#  (+ 50 random madasr clips)"
N_RANDOM = 100
N_WORST = 20
N_EXTRA = 50
assert N_RANDOM == 100, "SPEC §3 G0.6: 100 uniformly random clips"
assert N_WORST == 20, "SPEC §3 G0.6: 20 worst by the baseline disagreement"
assert N_EXTRA == 50, "SPEC §3 G0.6: 50 random madasr clips"

# "Pass: ≤5 material / 100 random and 0 catastrophic anywhere."
MAX_MATERIAL_PER_RANDOM = 5
MAX_CATASTROPHIC = 0
assert MAX_MATERIAL_PER_RANDOM == 5, "SPEC §3 G0.6 pass rule: ≤5 material / 100 random"
assert MAX_CATASTROPHIC == 0, "SPEC §3 G0.6 pass rule: 0 catastrophic anywhere"

# "Material error = >10 % WER of the reference vs what is heard". The threshold
# is the AUDITOR's, applied by ear — it is printed on the board rather than
# computed here, because what is being judged is the reference against the
# audio, not the reference against a model.
MATERIAL_WER_THRESHOLD = 0.10
assert MATERIAL_WER_THRESHOLD == 0.10, "SPEC §3 G0.6: material = >10 % of words"

# SPEC §2/§5/§6 use one seed for every registered draw; G0.6's sample is drawn
# with the same one so the board is reproducible from the manifests alone.
SEED = 20260905
assert SEED == 20260905, "SPEC §5/§6 registered seed"

VERDICTS = ("ok", "material", "catastrophic")

# 32 kbps mono MP3 — the orchestrator's registered transcode for this board.
# Low enough that 170 clips fit the size budget, high enough that a Bengali
# consonant cluster is still distinguishable; this is a listening aid, and the
# clip's own SHA is recorded so what was heard can be traced to what is on disk.
MP3_BITRATE = "32k"
MAX_MB_DEFAULT = 12.0

SEALED_TEST_DIR = Path(os.environ.get("BNASR_FLEURS_DIR", "/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn")).resolve()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _resolve(p):
    q = Path(p)
    q = q if q.is_absolute() else (REPO / q)
    rp = q.resolve()
    if not q.exists():
        # The macro/madasr manifests are still being ingested while this runs;
        # "no such file" from deep inside a JSON reader is a worse message than
        # naming the file the caller asked for and has not got yet.
        raise SystemExit(f"{q} does not exist yet")
    if rp == SEALED_TEST_DIR or SEALED_TEST_DIR in rp.parents:
        raise SystemExit(f"refusing to touch the sealed FLEURS test set: {p}")
    return q


def clip_id(row):
    """Stable id for a clip: the manifest's own id if it has one, else the
    audio file's stem. Verdicts are keyed by this, so it must not depend on the
    machine's directory layout — the exported JSON is pasted back on another
    day, possibly from another checkout."""
    for k in ("macro_id", "utt_id", "id"):
        if row.get(k):
            return str(row[k])
    return Path(row["audio_filepath"]).stem


def sample_clips(rows, n, seed, exclude=()):
    """Uniform sample WITHOUT replacement, deterministic given (rows, n, seed).

    Sampling on a sorted-by-id index rather than on manifest order: the
    ingestion jobs write these manifests concurrently, so manifest order is not
    a property of the data and a "seeded" sample that depends on it is not
    reproducible.
    """
    idx = sorted(range(len(rows)), key=lambda i: clip_id(rows[i]))
    idx = [i for i in idx if clip_id(rows[i]) not in set(exclude)]
    if n >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(idx), size=n, replace=False)
    return [idx[int(j)] for j in sorted(pick)]


# ---------------------------------------------------------------------------
# Decode + transcode
# ---------------------------------------------------------------------------

def decode_rows(rows, ckpt, batch_size=16, device="cuda"):
    """the baseline greedy hypotheses for the sampled clips (SPEC §3 G0.6: ≤0.5 GPU-h).

    Reuses ``panel_eval.load_ctc_model``/``decode_panel`` so the board's
    hypotheses come off exactly the decoder the panels are scored with; a board
    built with a different decoding config would show the owner errors the
    project never actually makes.
    """
    from bnasr.panel_eval import decode_panel, load_ctc_model
    model = load_ctc_model(ckpt, device=device)
    hyps, wall = decode_panel(model, rows, batch_size=batch_size)
    peak = None
    if device.startswith("cuda"):
        import torch
        peak = round(torch.cuda.max_memory_reserved() / (1 << 30), 3)
        del model
        torch.cuda.empty_cache()
    return hyps, {"wall_s": round(wall, 1), "peak_reserved_gib": peak,
                  "batch_size": batch_size}


def to_mp3_b64(path):
    """Transcode one clip to 32 kbps mono MP3 and return (base64, bytes).

    ffmpeg is invoked with an explicit output file rather than a pipe because a
    piped MP3 has no accurate duration header, and a browser that cannot seek in
    a 20 s clip makes the audit slower for no reason. A transcode failure raises:
    a board with a silent player is worse than no board, because the auditor
    would mark what they cannot hear.
    """
    with tempfile.TemporaryDirectory(prefix="p4board_") as td:
        out = Path(td) / "clip.mp3"
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
             "-ac", "1", "-b:a", MP3_BITRATE, "-map_metadata", "-1", str(out)],
            capture_output=True)
        if r.returncode != 0 or not out.exists():
            raise SystemExit(f"ffmpeg failed on {path}: "
                             f"{r.stderr.decode('utf-8', 'replace')[:300]}")
        raw = out.read_bytes()
    return base64.b64encode(raw).decode("ascii"), len(raw)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def word_diff_html(ref, hyp):
    """Word-level diff of hypothesis against reference, as escaped HTML.

    difflib on whitespace tokens of the FROZEN surface — the same tokenisation
    the WER denominator uses, so what is highlighted and what is counted are the
    same events.
    """
    a, b = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(htmllib.escape(" ".join(b[j1:j2])))
        elif tag == "replace":
            parts.append('<span class="sub" title="reference: '
                         + htmllib.escape(" ".join(a[i1:i2]), quote=True) + '">'
                         + htmllib.escape(" ".join(b[j1:j2])) + "</span>")
        elif tag == "insert":
            parts.append('<span class="ins">' + htmllib.escape(" ".join(b[j1:j2]))
                         + "</span>")
        elif tag == "delete":
            parts.append('<span class="del" title="missing from the hypothesis">'
                         + htmllib.escape(" ".join(a[i1:i2])) + "</span>")
    return " ".join(p for p in parts if p)


_CSS = """
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#666;--line:#e2e0dc;--card:#fff;
      --ins:#b45309;--del:#b91c1c;--sub:#1d4ed8;--ok:#15803d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 system-ui,-apple-system,"Noto Sans Bengali",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);
       padding:12px 20px;z-index:10}
h1{font-size:17px;margin:0 0 4px}
.meta{color:var(--mut);font-size:12.5px}
.tally{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:13px}
.tally b{font-variant-numeric:tabular-nums}
main{max-width:900px;margin:0 auto;padding:18px 20px 60px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:14px 16px;margin-bottom:14px}
.clip.done{border-color:#cfd8cf}
.cid{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut)}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;
       border:1px solid var(--line);color:var(--mut);margin-left:6px}
audio{width:100%;margin:10px 0 6px}
.lbl{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);
     margin-top:8px}
.txt{font-size:17px;line-height:1.7}
.ins{color:var(--ins);text-decoration:underline}
.del{color:var(--del);text-decoration:line-through}
.sub{color:var(--sub);border-bottom:2px solid var(--sub)}
.verdict{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;padding-top:10px;
         border-top:1px dashed var(--line);font-size:14px}
.verdict label{cursor:pointer}
textarea{width:100%;height:160px;font:12px ui-monospace,monospace;
         border:1px solid var(--line);border-radius:8px;padding:10px}
.note{color:var(--mut);font-size:13px}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:7px;
       background:var(--card);cursor:pointer}
@media (prefers-color-scheme:dark){
 :root{--bg:#16171a;--fg:#e8e6e3;--mut:#9a9a9a;--line:#2e3034;--card:#1e2024;
       --ins:#f0b429;--del:#f87171;--sub:#7aa2f7;--ok:#4ade80}}
"""

_JS = """
const N_RANDOM_STRATUM = %(n_random)d;
const v = {};
function tally(){
  let ok=0,mat=0,cat=0,matRandom=0;
  document.querySelectorAll('.clip').forEach(c=>{
    const val=v[c.dataset.cid];
    if(!val) return;
    c.classList.add('done');
    if(val==='ok')ok++;
    if(val==='material'){mat++; if(c.dataset.stratum==='random')matRandom++;}
    if(val==='catastrophic')cat++;
  });
  document.getElementById('t-done').textContent=ok+mat+cat;
  document.getElementById('t-ok').textContent=ok;
  document.getElementById('t-mat').textContent=mat;
  document.getElementById('t-matr').textContent=matRandom;
  document.getElementById('t-cat').textContent=cat;
  document.getElementById('export').value=JSON.stringify(v,null,1);
}
document.addEventListener('change',e=>{
  if(e.target.name && e.target.name.startsWith('v_')){
    v[e.target.dataset.cid]=e.target.value; tally();
  }
});
document.getElementById('copy').addEventListener('click',()=>{
  const t=document.getElementById('export'); t.select();
  document.execCommand('copy');
});
tally();
"""


def render_html(meta, clips, n_random_stratum):
    head = f"""<title>this recipe listening board — {htmllib.escape(meta['label'])}</title>
<style>{_CSS}</style>
<header>
<h1>this recipe human label audit (SPEC §3 G0.6) — {htmllib.escape(meta['label'])}</h1>
<div class="meta">{len(clips)} clips ·
 {n_random_stratum} uniformly random (the ≤{MAX_MATERIAL_PER_RANDOM}/100 rule applies
 to these only) · {len(clips) - n_random_stratum} enrichment / extra ·
 seed {meta['seed']} · the baseline hypotheses, greedy CTC.
 <b>Material</b> = more than {int(MATERIAL_WER_THRESHOLD * 100)} % of the words in the
 reference are not what you hear. <b>Catastrophic</b> = wrong pairing, non-speech, or
 gross truncation. Judge the <i>reference</i> against the <i>audio</i>; the model's
 hypothesis is shown only to point your ear at the disagreements.</div>
<div class="tally">
 <span>judged <b id="t-done">0</b>/{len(clips)}</span>
 <span>OK <b id="t-ok">0</b></span>
 <span>material <b id="t-mat">0</b> (random stratum <b id="t-matr">0</b>)</span>
 <span>catastrophic <b id="t-cat">0</b></span>
</div>
</header>
<main>
"""
    body = []
    for c in clips:
        cid = htmllib.escape(c["clip_id"])
        body.append(f"""<section class="clip" data-cid="{cid}" data-stratum="{c['stratum']}">
<span class="cid">{cid}</span>
<span class="badge">{htmllib.escape(c['stratum'])}</span>
<span class="badge">{htmllib.escape(c['source'])}</span>
<span class="badge">{c['duration']:.1f}s</span>
<span class="badge">the baseline WER {c['wer'] * 100:.0f}%</span>
<audio controls preload="none" src="data:audio/mpeg;base64,{c['mp3_b64']}"></audio>
<div class="lbl">reference (frozen surface — this is what is being audited)</div>
<div class="txt">{htmllib.escape(c['ref'])}</div>
<div class="lbl">the baseline hypothesis (diff vs reference)</div>
<div class="txt">{c['diff_html']}</div>
<div class="verdict">
 <label><input type="radio" name="v_{cid}" data-cid="{cid}" value="ok"> OK</label>
 <label><input type="radio" name="v_{cid}" data-cid="{cid}" value="material"> material error</label>
 <label><input type="radio" name="v_{cid}" data-cid="{cid}" value="catastrophic"> catastrophic</label>
</div>
</section>""")
    tail = f"""<section class="clip">
<h2 style="font-size:15px;margin:0 0 6px">Export verdicts</h2>
<div class="note">Copy this JSON and hand it back to the pipeline:
<code>listening_board.py score --sample {htmllib.escape(meta['sample_file'])}
--verdicts &lt;file&gt;</code>. The pass rule
(≤{MAX_MATERIAL_PER_RANDOM} material per 100 random clips, {MAX_CATASTROPHIC}
catastrophic anywhere) is evaluated there, not here.</div>
<textarea id="export" readonly></textarea>
<button id="copy">copy</button>
</section>
</main>
<script>{_JS % {'n_random': n_random_stratum}}</script>
"""
    return head + "\n".join(body) + tail


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(args):
    rand_rows = load_manifest(_resolve(args.random_manifest))
    extra_rows = (load_manifest(_resolve(args.extra_manifest))
                  if args.extra_manifest else [])

    # Stratum 1: uniformly random from the audited bucket.
    r_idx = sample_clips(rand_rows, args.n_random, SEED)
    random_ids = [clip_id(rand_rows[i]) for i in r_idx]

    # Stratum 2: the worst by the baseline disagreement. A "worst" needs a pool to be
    # worst OF, so a larger pool is drawn and ranked; the pool is disjoint from
    # the random stratum so a clip is never counted in both, which would let an
    # enrichment pick inflate the ≤5/100 denominator's numerator.
    pool_idx = sample_clips(rand_rows, args.worst_pool, SEED + 1, exclude=random_ids)
    extra_idx = sample_clips(extra_rows, args.n_extra, SEED + 2) if extra_rows else []

    rows_to_decode = ([rand_rows[i] for i in r_idx]
                      + [rand_rows[i] for i in pool_idx]
                      + [extra_rows[i] for i in extra_idx])
    for r in rows_to_decode:
        _resolve(r["audio_filepath"])
    print(f"decoding {len(rows_to_decode)} clips "
          f"({len(r_idx)} random + {len(pool_idx)} worst-pool + "
          f"{len(extra_idx)} extra)", flush=True)
    hyps, dmeta = decode_rows(rows_to_decode, args.ckpt, args.batch_size, args.device)

    n_r, n_p = len(r_idx), len(pool_idx)
    def rec(row, hyp, stratum):
        ref = frozen_norm(row["text"])
        e, n = wer_counts(ref, frozen_norm(hyp))
        return {"clip_id": clip_id(row), "stratum": stratum,
                "audio_relpath": str(Path(row["audio_filepath"])),
                "source": row.get("source", "?"),
                "duration": float(row.get("duration", 0.0)),
                "ref": ref, "hyp": frozen_norm(hyp),
                "wer": (e / n) if n else 1.0, "ref_words": n, "errors": e}

    randoms = [rec(r, h, "random") for r, h in zip(rows_to_decode[:n_r], hyps[:n_r])]
    pool = [rec(r, h, "worst") for r, h in
            zip(rows_to_decode[n_r:n_r + n_p], hyps[n_r:n_r + n_p])]
    extras = [rec(r, h, "extra") for r, h in
              zip(rows_to_decode[n_r + n_p:], hyps[n_r + n_p:])]
    # Ranked by disagreement, ties broken by clip id so the selection is a
    # function of the data and not of numpy's sort stability.
    pool.sort(key=lambda c: (-c["wer"], c["clip_id"]))
    worst = pool[:args.n_worst]

    clips = randoms + worst + extras
    total_mp3 = 0
    for c in clips:
        b64, nbytes = to_mp3_b64(_resolve(c["audio_relpath"]))
        c["mp3_b64"] = b64
        c["mp3_bytes"] = nbytes
        c["audio_sha256"] = sha256_file(_resolve(c["audio_relpath"]))
        c["diff_html"] = word_diff_html(c["ref"], c["hyp"])
        total_mp3 += nbytes

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_file = out.with_suffix(".sample.json")
    meta = {
        "tool": "bnasr/listening_board.py",
        "spec": "experiments/SPEC.md §3 G0.6",
        "label": args.label or Path(args.random_manifest).stem,
        "seed": SEED,
        "sample_file": sample_file.name,
        "registered_sizes": {"n_random": N_RANDOM, "n_worst": N_WORST,
                             "n_extra": N_EXTRA},
        "sizes_used": {"n_random": len(randoms), "n_worst": len(worst),
                       "n_extra": len(extras), "worst_pool": len(pool)},
        "is_registered_g06_board": (len(randoms) == N_RANDOM
                                    and len(worst) == N_WORST
                                    and len(extras) in (0, N_EXTRA)),
        "pass_rule": {"max_material_per_100_random": MAX_MATERIAL_PER_RANDOM,
                      "max_catastrophic_anywhere": MAX_CATASTROPHIC,
                      "material_definition_wer": MATERIAL_WER_THRESHOLD,
                      "note": "the ≤5/100 rate is computed over the RANDOM "
                              "stratum only; the worst-by-disagreement clips are "
                              "an enrichment sample (SPEC §3 G0.6: machine "
                              "disagreement is ASR-disagreement, not label noise)"},
        "manifests": {
            "random": {"path": str(args.random_manifest),
                       "sha256": sha256_file(_resolve(args.random_manifest)),
                       "rows": len(rand_rows)},
            "extra": ({"path": str(args.extra_manifest),
                       "sha256": sha256_file(_resolve(args.extra_manifest)),
                       "rows": len(extra_rows)} if extra_rows else None)},
        "ckpt": str(Path(args.ckpt).resolve()), "ckpt_sha256": sha256_file(args.ckpt),
        "decode": dmeta,
        "mp3": {"bitrate": MP3_BITRATE, "total_bytes": total_mp3},
        "scorer_sha256": sha256_file(REPO / "bnasr/scoring.py"),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    sample_file.write_text(json.dumps(
        {"meta": meta,
         "clips": [{k: v for k, v in c.items() if k not in ("mp3_b64", "diff_html")}
                   for c in clips]},
        indent=1, ensure_ascii=False), encoding="utf-8")

    page = render_html(meta, clips, len(randoms))
    # Two hard properties of the artefact, asserted rather than hoped for.
    assert "/mnt" not in page, "the board must not carry an absolute box path"
    mb = len(page.encode("utf-8")) / (1 << 20)
    if mb > args.max_mb:
        raise SystemExit(f"board is {mb:.1f} MB, over the {args.max_mb} MB budget; "
                         f"lower --n-random/--n-extra or the bitrate")
    out.write_text(page, encoding="utf-8")
    meta["html_mb"] = round(mb, 2)
    meta["html_path"] = str(out)
    meta["html_sha256"] = sha256_file(out)
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def score(args):
    """Apply the SPEC §3 G0.6 pass rule to a verdict file from the owner."""
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    clips = {c["clip_id"]: c for c in sample["clips"]}
    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    if not isinstance(verdicts, dict):
        raise SystemExit("--verdicts must be a JSON object {clip_id: verdict}")

    unknown = sorted(set(verdicts) - set(clips))
    bad = sorted({v for v in verdicts.values() if v not in VERDICTS})
    if unknown:
        raise SystemExit(f"verdicts for {len(unknown)} clip(s) not in this sample, "
                         f"e.g. {unknown[:3]} — the verdict file and the sample "
                         f"file do not describe the same board")
    if bad:
        raise SystemExit(f"unrecognised verdict value(s) {bad}; expected {VERDICTS}")
    missing = sorted(set(clips) - set(verdicts))

    strata = {}
    for cid, c in clips.items():
        s = strata.setdefault(c["stratum"], {v: 0 for v in VERDICTS})
        v = verdicts.get(cid)
        if v:
            s[v] += 1
    rand = strata.get("random", {v: 0 for v in VERDICTS})
    n_rand_judged = sum(rand.values())
    n_rand_total = sum(1 for c in clips.values() if c["stratum"] == "random")
    catastrophic = sum(s["catastrophic"] for s in strata.values())

    # The rule is "≤5 material / 100 random". Expressed as a rate so a board of
    # a different size (a smoke board, or a bucket with fewer clips) is judged on
    # the same standard rather than on a count that silently means something else.
    material_per_100 = (100.0 * rand["material"] / n_rand_judged) if n_rand_judged else None
    complete = not missing
    passes = None
    if complete and n_rand_judged:
        passes = bool(material_per_100 <= MAX_MATERIAL_PER_RANDOM
                      and catastrophic <= MAX_CATASTROPHIC)

    rep = {
        "tool": "bnasr/listening_board.py score",
        "spec": "experiments/SPEC.md §3 G0.6",
        "sample": {"path": str(args.sample), "sha256": sha256_file(args.sample),
                   "label": sample["meta"]["label"],
                   "is_registered_g06_board": sample["meta"]["is_registered_g06_board"]},
        "verdicts": {"path": str(args.verdicts), "sha256": sha256_file(args.verdicts),
                     "n": len(verdicts)},
        "by_stratum": strata,
        "random_stratum": {"n": n_rand_total, "judged": n_rand_judged,
                           "material": rand["material"],
                           "material_per_100": (round(material_per_100, 2)
                                                if material_per_100 is not None else None),
                           "limit_per_100": MAX_MATERIAL_PER_RANDOM},
        "catastrophic_anywhere": catastrophic,
        "catastrophic_limit": MAX_CATASTROPHIC,
        "unjudged_clips": missing,
        "complete": complete,
        # None, never True, on an incomplete board: an audit with 40 of 170 clips
        # judged has not passed G0.6, it has not finished it.
        "passes_g06": passes,
        "consequence": ("SPEC §3 G0.6: a failed bucket is excluded whole"
                        if passes is False else
                        "bucket admitted subject to the remaining Gate-0 checks"
                        if passes else
                        "verdict withheld: the board is not fully judged"),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="decode the sample and write the HTML board")
    b.add_argument("--random-manifest", required=True,
                   help="the bucket under audit (SPEC §3 G0.6: macro_filtered)")
    b.add_argument("--extra-manifest", default=None,
                   help="the +50 random clips (SPEC §3 G0.6: madasr)")
    b.add_argument("--ckpt", default=str(REPO / "experiments/fastconformer_ctc_bn_1kh/final.nemo"))
    b.add_argument("--n-random", type=int, default=N_RANDOM)
    b.add_argument("--n-worst", type=int, default=N_WORST)
    b.add_argument("--n-extra", type=int, default=N_EXTRA)
    b.add_argument("--worst-pool", type=int, default=500,
                   help="clips ranked by the baseline disagreement to pick the worst from")
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--device", default="cuda")
    b.add_argument("--label", default=None)
    b.add_argument("--max-mb", type=float, default=MAX_MB_DEFAULT)
    b.add_argument("--out", required=True)

    s = sub.add_parser("score", help="apply the G0.6 pass rule to owner verdicts")
    s.add_argument("--sample", required=True, help="the .sample.json written by build")
    s.add_argument("--verdicts", required=True, help="{clip_id: verdict} from the board")
    s.add_argument("--out", default=None)

    a = ap.parse_args()
    if a.cmd == "build":
        if a.device.startswith("cuda"):
            pin_gpu()
        build(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
