# Phase-4 end-of-run checklist — the main-loop runbook

Written 2026-09-15 ~04:30 UTC at update ~118,000, ~3 h before H=120,000, so that
the sequence survives a context compaction. Every command below has been
rehearsed on the real tool against the real run (SPEC §8w.14, §8w.16). Nothing
here is delegated to a subagent except the recompute/review/draft phases in
`experiments/p4/workflows/end_of_run.js`.

Owner standing rules that bind every step: never touch a co-tenant process;
never `pkill -f`; every subagent is opus/medium; commit as `ehzawad`, no
co-author or AI trailer; report each step, do not go quiet.

## 0. Confirm the normal ending (do not proceed on a log line alone)

```bash
cd /mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline
R=experiments/p4/runs/N_O+macro_filtered_H120k
grep -c 'COMPLETE at update 120000' $R/supervisor.log        # must be 1
python3 -c "import json;print(any(json.loads(l)['update']==120000 for l in open('$R/val.jsonl')))"   # True
ls -la $R/milestones/step_120000.pt                           # must exist
bash experiments/p4/monitors.sh                               # trainer will now be DOWN by design; note it
```

## 1. Report validation 120,000 and the SPEC §8v final window

`python3 tools/p4/p4_progress.py | tail -3`, then the 8v numbers: best eligible S
through 110,000 (baseline **16.94087**, fixed 2026-09-14 17:53 UTC) vs through
120,000; gain ≥ 0.20 pp undermines convergence. Report both the eligible
minimum and the tie-selected checkpoint's gain.

## 2. Ask the owner the four pending questions BEFORE anything irreversible

He said: "if you have a question, you should ask me when the full training is
there." These are the questions, with the council-recommended answer first:

1. **Licence.** CC-BY-SA-4.0 (recommended: commercially usable, ShareAlike
   from OpenSLR-53 propagates, what Phase 1 shipped as, what council advised)
   vs MIT/Apache (only defensible if weights are held not to be a derivative
   of training data, which is legally unsettled).
2. **MaCro licence record.** Council AA says the Kaggle rules state
   CC-BY-4.0 with commercial use permitted; NOT independently verified (page
   is JS-rendered, API unauthenticated, bundle has no licence file). Owner can
   close it in two minutes by viewing the rules page logged in. Until closed:
   Phase 4 must NOT ship commercially. See
   `experiments/p4/licences/MACRO_BengaliAI_licence_OPEN_ITEM.md`.
3. **"New private repo for the pipeline":** reuse
   `ehzawad/bengali-fastconformer-phase4` (already private, holds exactly the
   pipeline) or create a separate one? One word resolves it.
4. **Card scope:** he wants "dataset, benchmark, output" only. Attribution is
   a CC-BY-SA licence obligation, and SPEC §8w.14 requires disclosing TWO
   FLEURS events. Propose: the minimal card plus a linked public supplement
   carrying credits and the seven-panel numbers.

## 3. Mint the qualified selection (the ONE real mint)

```bash
export CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567
nice -n 19 .venv/bin/python tools/p4/p4_qualify.py --device cuda
```
Refuses unless the ending is normal; selects by `select_best`; qualifies per
§8w.2 item 4; exports `.nemo` and requires exact dev-WER reproduction; refuses
to overwrite. Output: `experiments/p4/qualified_selection.json`. **Read it back
and quote selected_update, selected_S_pp, qualified, export_verified.match.**

## 4. Run the review workflow (subagents: recompute, review, draft — nothing else)

Pass `experiments/p4/workflows/end_of_run.js` inline to the Workflow tool.
It gates on step 3, recomputes selection/qualification/parity/8v, adversarially
reviews the two commands below, and drafts the card, launcher and report.
Resolve every problem it raises before step 5.

## 5. Event two — the second sealed FLEURS decode (main loop only)

```bash
export CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567
export P4_FINAL_TEST_SECOND_EVENT=1          # the ONLY override; documented in SPEC 8w.14
NEMO=$(python3 -c "import json;print(json.load(open('experiments/p4/qualified_selection.json'))['selected_checkpoint'])")
nice -n 19 .venv/bin/python tools/p4/p4_final_test.py --ckpt "$NEMO" --out-dir experiments/p4/final_test --device cuda
```
Do NOT set `P4_ALLOW_UNQUALIFIED_FINAL`. Confirm afterwards that
`experiments/p4/final_test_EARLY_step88000/` is untouched and the early record
is preserved wherever `ensure_event` moves it. Report: WER/CER, Δ vs P2 16.536
and P1 19.672 with CIs, meets_success, meets_strong.

## 6. One-shot panels (main loop only, AFTER step 5)

```bash
nice -n 19 .venv/bin/python tools/p4/p4_panel_eval.py --ckpt "$NEMO" --panels <names per the Review verdict> \
  --out experiments/p4/eval/oneshot_panels_<step>.json --save-hyps experiments/p4/eval/oneshot_panels_<step>_hyps.json --device cuda
# both outputs are REQUIRED for a one-shot (the tool refuses otherwise) and must not already exist
```
Do NOT set `P4_ALLOW_PROTECTED_PANEL`. Gate now requires the qualified record
and binds the event sha to the selected sha. Audit log (SPEC §8w.19 format):
one `requested` record per invocation, then one `decoded` record per panel
written AFTER its decode; a prior `decoded` (or uncorrected legacy) record for
a panel is an ABORT unless the owner records why and sets
`P4_ALLOW_PROTECTED_REPEAT=1`. Partial results land in `<out>.partial.json`
after every panel; if the process dies, the panels already scored are there.

**Incident 2026-09-15 08:12Z:** the first invocation decoded holdout and
vaani_validation, then died on a post-decode assertion (3 path collisions in
the Vaani validation manifest) before writing anything. Correction record
appended; the five untouched panels decoded 08:26Z; the two spent panels are
re-decoded ONLY on the owner's ruling (council AB). See SPEC §8w.19.

## 7. Card, HF, benchmark, pushes (each an owner-visible step)

* Fill the drafted card; licence per Q1; MaCro line per Q2; both FLEURS events.
* HF: update `ehzawad/stt_bn_fastconformer` licence/card (Phase 1, no MaCro,
  clean) — and publish Phase 4 ONLY after Q2 is closed. Preserve the six
  baseline artefact hashes before touching the existing HF repo.
* Benchmark ×7 on the A5000 with the REVIEWED launcher, never
  `run_all_benchmarks.sh` unchanged. Declare it prospectively as a
  post-release reproducibility run and a THIRD FLEURS exposure.
* Commit + push both repos as `ehzawad`; final push of the Phase-4 repo.

## What must never happen
A second `qualified_selection.json`; a decode with `P4_ALLOW_UNQUALIFIED_FINAL`;
a protected panel before event two; `pkill -f` on the shared box; a co-tenant
process signalled; an HF licence flip before the owner answers Q1/Q2.
