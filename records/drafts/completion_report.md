Verified against `val.jsonl` (38 eligible of 60). Report skeleton follows.

```markdown
# Phase 4 — Expanded-Corpus FastConformer: Completion Report (DRAFT SKELETON)

Run `971d3f8245a4d3c3` · `experiments/p4/runs/N_O+macro_filtered_H120k` · H = 120,000 (27.88 epochs)
Ending: **NORMAL**, 2026-09-15T07:52:19Z · 60 validations, 38 eligible · 134.9 GPU-h (phase ceiling 350)

## 1. §8v final-window check (110k → 120k) — NOT TRIPPED
Re-derived from `val.jsonl` by the final-window-8v rule (best eligible `S_exact`; threshold ≥ 0.20 pp = convergence claim undermined):

| measure | through 110,000 | through 120,000 | gain |
|---|---:|---:|---:|
| best eligible S | 16.94087 (step 110,000) | 16.87973 (step 120,000) | **+0.06114** |
| tie-selected checkpoint | step_100000, 16.98500 | step_116000, 16.89126 | **+0.09374** |
| D WER (at best-eligible) | 16.871 | 16.813 | +0.058 |
| conversational WER (at best-eligible) | 19.297 | 18.980 | +0.317 |

Reading: converged by ~100,000; the last 20,000 updates (~25 GPU-h) bought little. Authorises no extension and does not prove a shorter schedule would have won.

## 2. Selection and qualification (`experiments/p4/qualified_selection.json`)
- Rule: lowest eligible S; ties within 0.05 pp → earlier. Tied set {116,000; 118,000; 120,000} → **step_116000**.
- Selected S = **16.89126 pp** (gate ≤ 22.0296 ✓); D = **16.831 pp** (gate ≤ 21.348 ✓); all guards pass; eligible ✓ → **qualified: true**.
- Baselines: −4.14 pp S vs Phase-1 composite 21.0296; D 20.348 → 16.831.
- Export: `export/step_116000.nemo`, sha256 `bea300ce…`; source `best/step_116000.pt`, sha256 `45351ab3…`.
- Export verification: recorded 0.16831 == exported 0.16831 on n = 5,385.
- Identity: init N (`1f2c819d…`), tokenizer `6288cf8e…`, train manifest `00d6506b…`, effective batch 252.
- Minted 2026-09-15T07:57:24Z by `tools/p4/p4_qualify.py` (SPEC §8w.2 item 4), dry_run false.

## 3. Disclosure: TWO FLEURS-test events (SPEC §8w.14)
Phase 4 has two FLEURS-test decode events, both disclosed here and on the model card, regardless of which number is better.

**Event one — early, owner-authorised deviation from §6** (update 88,000, 2026-09-13). Authorisation recorded in `EARLY_FLEURS_AUTHORISATION.json` before any decode; run incomplete, no qualified-selection record, decoded with `P4_ALLOW_UNQUALIFIED_FINAL=1` (stamped "not a registered Phase-4 result"). Checkpoint `export/step_88000.nemo`, sha256 `a73e6585…` (from `best/step_88000.pt`, `31d0ce4b…`). 920 test rows, frozen scorer, 20,000 paired sentence-cluster draws, seed 20260905:

| model | WER pp | CER pp | Δ vs P4 | 95 % CI of Δ < 0 | P(P4 better) |
|---|---:|---:|---:|---|---:|
| P4 step_88000 | 17.5441 | 5.2704 | — | — | — |
| P2 frozen (qwen3_adapter) | 16.5362 | 4.6781 | +1.0078 | no | 0.0074 |
| P1 frozen (ehzawad_fastconformer) | 19.6723 | 5.7915 | −2.1283 | yes | 1.0000 |

`meets_success` false, `meets_strong` false. Bundle: `experiments/p4/final_test_EARLY_step88000/`.

**Event two — the registered end-of-run decode on the qualified selection (step_116000).** Requires `P4_FINAL_TEST_SECOND_EVENT=1`. *[NOT YET RUN — placeholders below.]*

| model | WER pp | CER pp | Δ vs P4 | 95 % CI of Δ < 0 | P(P4 better) |
|---|---:|---:|---:|---|---:|
| P4 step_116000 | _TBD_ | _TBD_ | — | — | — |
| P2 frozen | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| P1 frozen | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

`meets_success` _TBD_ · `meets_strong` _TBD_ · baseline re-score gates _TBD_ · bundle path _TBD_.

## 4. Protected one-shot panels — PLACEHOLDERS (not yet decoded)
Gated on the same held-decode record as event two; each decode appends to `protected_panel_decodes.jsonl`.

| panel | rows / distinct sentences | WER pp | CER pp |
|---|---:|---:|---:|
| holdout | 6,561 sent. | _TBD_ | _TBD_ |
| Vaani validation | 10,071 sent. | _TBD_ | _TBD_ |
| Vaani test | 12,600 sent. | _TBD_ | _TBD_ |
| FLEURS validation | 150 sent. | _TBD_ | _TBD_ |
| MaDASR dev | 494 sent. | _TBD_ | _TBD_ |

Locked dev/selection panels (not one-shot), at step_116000: D 16.831 · M _TBD_ · K _TBD_ · R _TBD_.

## 5. Training corpus — arm N, seven sources (`experiments/p4/corpus/build_report.json`)
Seed 20260905 · **1,265,884 rows / 1,958.73 h** · new hours beyond O = 992.85 (min 500 ✓).

| source | rows | hours | share |
|---|---:|---:|---:|
| MaCro (macro_filtered) | 543,420 | 726.47 | 37.09 % |
| IndicVoices | 273,001 | 574.37 | 29.32 % |
| OpenSLR-53 | 162,144 | 158.96 | 8.12 % |
| MaDASR | 95,241 | 148.97 | 7.61 % |
| Vaani | 85,947 | 117.40 | 5.99 % |
| Kathbath | 45,965 | 81.46 | 4.16 % |
| Subakko | 33,800 | 105.81 | 5.40 % |
| Common Voice | 24,712 | 39.46 | 2.01 % |
| FLEURS train | 1,654 | 5.83 | 0.30 % |

*(Table lists the nine retained buckets in arm N; "seven sources" per build_report — reconcile the count wording before publishing.)*
Excluded: **SPRING-INX** (G0.9(a) script FAIL 46.18 %; G0.9(b) duplicate rate 35× ceiling) and **macro_validated** (PANEL_ONLY, 31.1 % PCM-duplicate rate). Fingerprint exposure: 5,296 train rows excluded, 135 M rows removed. Audio-defect re-verification: 2,019 flagged, 2,019 rescued, 0 still defective.

## 6. Open items — owner decisions pending
1. **MaCro licence record** — provenance/licence terms for the MaCro (competition) bucket not yet written to `experiments/p4/licences/`; required before any public release.
2. **Licence choice for the release** — owner's prior lean was NC-SA; not confirmed for this artefact.
3. **Repo: reuse vs separate** — publish into the existing `ehzawad/bengali-fastconformer` repo (new revision) or mint a separate Phase-4 repo. Affects HF model id, card scope and benchmark ×7 wiring.
4. *(Also outstanding from §8w.17)* card scope.
```