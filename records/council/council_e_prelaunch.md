**Verdict: NO-GO on the supplied packet.** The retained hours pass G0.8, but the packet does not yet establish Gate-0 closure. Several items require corrected certification rather than another experiment.

**Q1. Launch blockers**

| Launch | Status | What must close first |
|---|---|---|
| O_A5000 | **BLOCKED** | Retained-corpus leakage certification and audit coverage; corrected M baseline; remaining Gate-0 evidence listed below. |
| O_A6000 | **BLOCKED** | Everything above, plus an explicit amendment acknowledging the failed registered 1% cross-card threshold. |
| Three bucket arms | **BLOCKED** | Common Gate-0 closure; corrected G0.6 report including board clips and Vaani’s reinstatement. Under the sequence below, finish the controls first. |
| N_final | **BLOCKED by design** | Single-bucket admission decisions, construction of the corresponding frozen union, and ≥500 unique new hours in that union. Its 8k test then determines continuation. |

The specific closeout items are:

1. **Certify leakage on the retained manifests.** The supplied `leakage_report` has `PASS=false`: 17,424 unresolved text matches, 380 PCM matches, and 296 confirmed plus 7,937 unresolved fingerprint matches. These may be findings against incoming rows subsequently excluded; they do **not** prove retained contamination. Conversely, removal counts do not prove zero residual matches.

   Produce a report joining every finding and component alias to the final manifests, showing zero retained prohibited matches. Preserve the incoming report. Explain the **665 `totals_fleurs` matches**, including which split they concern, and reconcile the earlier “0 matches” statement. This requires audit reconciliation, not headline decoding.

2. **Establish complete audit coverage.** The fingerprint index contains 19,174 clips. D, indexed M, R, K, FLEURS test, and Vaani validation/test alone total **34,805 manifest rows**, before holdout and SPRING panels. Index deduplication could explain some difference; a manifest-to-index coverage map must explain all of it.

   Also establish:
   - M’s comparison against **complete Phase-1/2/3 historical audio and text/tokenizer exposure**, not merely current O.
   - The registered **training-to-training fingerprint deduplication**; a train×panel report does not certify that.
   - Required candidate coverage, including the excluded SPRING bucket and its protected splits. Its `hash_coverage=0` is unfinished under the contract’s “EVERY candidate bucket” wording.

3. **Recompute the baseline on final M.** M’s **3.237 h / 1,829 groups passes** the registered 3–4 h and ≥500-group requirements. Keep that fixed panel; do not replenish it. However, the supplied P1 baseline uses **1,964 rows**, so **S(P1)=20.998 is stale**. Restrict existing P1 predictions to the surviving IDs and recompute corpus WER and S. Keep weights **0.60/0.20/0.10/0.10**.

4. **Attach the remaining required closeout evidence.** This extract does not certify:
   - G0.1/2 provenance and licence completeness;
   - unchanged tokenizer, retained `<unk>` checks, tokens/s p99 and the remaining G0.9 clauses;
   - incoming G0.9 rates for O at the registered bucket granularity;
   - the **21-longest-clip, microbatch-12** resource test on final admitted manifests. A 9.2 GiB qualification-run peak alone does not establish that test;
   - complete freeze coverage and exact O ID/target inclusion in every arm and NVMe remap.

   Reconcile the candidate-count changes—MaCro filtered **550,921→550,617**, validated **29,583→29,462**—with the exclusion ledger and registered denominators. These are missing certifications, not evidence that every gate failed.

5. **Correct contradictory statuses.** Preserve raw results, but issue a reconciled summary distinguishing resume semantics, numerical reproducibility, cross-card qualification, and G0.6 eligibility.

**CLOSED on the stated evidence:** fingerprint calibration; M’s size; the registered replay requirement of identical IDs/LRs across the interruption and epoch boundary; and the all-bucket union’s quantity requirement, **992.85 new hours**. Those do not close the other gates.

**Q2. Accept the amended G0.6 closure; no mandatory hypothesis-free re-audit.**

The owner explicitly replaced the listening-verdict gate before the machine results. Requiring the superseded human thresholds now would introduce another admission rule. The corrected owner statement is an informal spot-check; the 12 MaCro retranscriptions corroborate it without constituting an independent human audit.

The machine report nevertheless needs correction: its verdicts currently omit the board strata, although the registered audit includes them. For Vaani, assuming distinct sampled IDs:

- Random sample: 18 consensus disagreements, zero catastrophic-like.
- Board: two consensus disagreements, **one catastrophic-like**.
- Combined: **20/1,060 = 1.887%** disagreement and **1/1,060 = 0.0943%** catastrophic-like.

Both pass **4.27% / 0.70%**. Report the board’s **1/60 = 1.67%** catastrophic-like rate separately. Verify sample IDs and assign the MaCro/RESPIN board clips to their respective buckets.

Exact reconciliation wording:

> G0.6 uses the owner-amended machine-consensus rule, including each bucket’s random sample and assigned board clips, with both strata reported separately. The earlier hypothesis-directed owner verdicts are withdrawn. The corrected owner assessment is recorded as an informal reference spot-check. The superseded listening audit’s zero-catastrophic criterion does not apply. MaCro-filtered, MADASR and Vaani pass G0.6 and remain eligible for their individual pilots.

Delete the stale sentence excluding Vaani for six owner “catastrophic” verdicts.

**Q3. Neither exposure issue invalidates the registered numerical passes. Both limit their interpretation.**

Phase-1 exposure can depress the old baseline and tighten the new-bucket threshold. Hishab exposure can suppress new-bucket disagreements and make the audit less sensitive to label defects. These effects are not quantified and cannot be assumed to cancel. Low WER alone also does not establish training provenance.

Exact card wording:

> Label quality was machine-audited with Phase-1 FastConformer and hishab/titu_stt_bn_conformer_large and informally spot-checked by the owner; labels were not independently human-verified. The recognisers are separate models, but their judgments are not independent of training exposure. Phase 1 trained on the old-corpus baseline, and hishab is treated as training-exposed for MaCro/RESPIN. These exposures can bias the comparison in different directions. Passing the preregistered thresholds establishes recogniser/reference agreement under this procedure, not a measured label-error rate. Random and board-sample results, including Vaani’s one catastrophic-like board case, are reported separately.

If Hishab exposure is only suspected, replace “training-exposed” with “potentially training-exposed.”

**Q4. Keep RESPIN’s cap and its separate arm.**

Retain **95,241 recordings / 148.97 h**, with five distinct speakers per sentence×dialect cell and the registered cross-source sentence ceiling. No re-raise.

**O+madasr remains meaningful:** it directly tests whether this retained material helps at equal update exposure. Folding it into N_final would prevent the required individual admission decision. The 500-hour minimum applies to the corpus selected for the long run, not to each diagnostic bucket arm.

Consequently, RESPIN alone cannot support continuation. Neither can RESPIN+Vaani: **266.37 new hours**. **MaCro must pass for Phase 4 to continue under the current minimum.**

**Q5. Confirm the 8k control test as decisive, through an explicit pre-pilot amendment.**

Do not describe the original 1% threshold as merely proposed: it was registered and failed. Do not replace it with a threshold fitted just above 1.58%.

Exact amendment:

> The registered 200-update cross-card loss-agreement threshold of 1% was exceeded: the measured worst block difference was 1.582%. This failure is preserved in the qualification record. Before pilot training, loss-block agreement is changed to a diagnostic; identical cut IDs, identical LRs and finite losses/gradients remain mandatory. A6000 decision eligibility requires the 8,000-update controls to agree within 0.10 pp S, 0.20 pp D and 0.30 pp for every D source. Every A6000 bucket arm must pass against both controls. If control agreement fails, A6000 results cannot determine admission and affected decision arms run on the A5000.

The same-card **1.233%** drift supports treating bitwise reproducibility separately from resume semantics. It does **not** establish that “most” cross-card variation is ordinary run-to-run noise, or identify its kernel-level cause. Remove those causal claims. No additional replay is required solely to satisfy the registered IDs/LR criterion.

**Q6. No additional Vaani treatment is registered.**

Keep the registered annotation transformations and partial-marker exclusions. Do not insert model-heard false starts, remove further clips by ASR disagreement, or retune its mixture.

Its arm decides admission. A conversational-D improvement is plausible; aggregate D and FLEURS effects remain uncertain. Omitted false starts could encourage deletion, but the supplied audit does not establish an expected signed change. Vaani’s official validation/test remain one-shot panels after final checkpoint selection.

**Q7. O stays frozen; disclose the OpenSLR finding.**

The **37/500 = 7.4%** result is consensus disagreement, not 7.4% verified label corruption. It does not authorize ASR-ranked filtering or a new source-removal experiment.

Suggested wording:

> In the old OpenSLR-53 audit sample, 37 of 500 clips met the two-recogniser consensus-disagreement condition. This may reflect reference problems or shared recognition errors. No rows were selected or removed using this result; the registered O control remained fixed.

This does not exempt O from closing its already-required G0.9 checks.

**Q8. Launch sequence**

Use this sequence after the common Gate-0 blockers and cross-card amendment close:

| Wave | A5000 | A6000 | Decision before advancing |
|---|---|---|---|
| 1 | **O_A5000**, 8k | **O_A6000**, 8k | Compare controls using all three agreement tolerances. |
| 2 | **O+macro_filtered**, 8k | **O+madasr**, 8k | MaCro compares with O_A5000; MADASR compares with both controls. |
| 3 | **O+vaani**, 8k | Idle | Vaani compares with O_A5000. Determine passing buckets and remaining new hours. |
| 4 | **N_final**, 8k from init A, if it is a distinct multi-bucket corpus | Idle | Apply N_final admission and best-single-arm safeguard. |
| Continuation | Chosen admissible arm resumes its own 8k state | Idle | Preserve optimizer, scheduler, sampler and RNG; no second warmup. |

If the controls disagree, retain O_A5000 and run all decision bucket arms on the A5000 sequentially. Do not repeat the canonical control merely because the A6000 control failed.

Before wave 1, freeze the deterministic construction of every possible eligible N_final subset and its manifests/checksums. The supplied all-three-bucket `N` is only one candidate. The possible new-hour totals are:

| Passing corpus | Unique new hours |
|---|---:|
| MaCro alone | 726.47 |
| MaCro + MADASR | 875.44 |
| MaCro + Vaani | 843.87 |
| All three | ≈992.85 |
| Any combination without MaCro | ≤266.37 — insufficient |

Two useful pre-pilot clarifications:

> If the admitted union is identical to an already tested single-bucket arm, reuse that arm’s 8k checkpoint; do not conduct another selection attempt on the identical corpus.

> If a distinct N_final fails, the fallback is the lowest-S passing single-bucket arm that independently satisfies G0.8. With the frozen inventories, only O+macro_filtered can qualify.

For a distinct union, do not initialize N_final from a single-bucket 8k checkpoint. It receives its own registered schedule prefix from init A.

The exact 8k admission tests are:

| Test | Requirement |
|---|---|
| Endpoint | Evaluate the **8,000-update checkpoint** on final D/M/R/K. Earlier checkpoints do not substitute for the admission endpoint. |
| Score | `S = 0.60 D + 0.20 M + 0.10 R + 0.10 K`, using corpus WER in percentage points. R stays included even if MADASR later fails its arm. |
| Required comparator C | O_A5000 for A5000 arms; **both controls separately** for A6000 arms. |
| Minimum improvement | `S(arm) − S(C) ≤ −0.30 pp`. |
| Marginal uncertainty | Paired 95% interval for `S(arm) − S(C)` has upper bound **strictly below zero**. |
| Simultaneous bound | With **m=3**, upper bootstrap quantile at **`1 − 0.025/3 = 0.9916667`**, the **99.1667th percentile**, is strictly below zero. Keep m=3 despite early stopping or exclusion. |
| Bootstrap implementation | **10,000 draws**, seed **20260905**; paired sentence-cluster resampling, coupled weights for shared sentence clusters across panels; recompute each panel’s corpus WER before forming S. |
| D safeguard | `D(arm) ≤ D(C) + 0.30 pp`. |
| Source safeguard | Every D source: `WER(arm) ≤ WER(C) + 1.00 pp`. |
| Conversation safeguard | Conversational D: `WER(arm) ≤ WER(C) + 0.50 pp`. |
| Absolute P1 eligibility | Every D source also remains within **P1 +1.00 pp**; conversational D within **P1 +0.50 pp**, giving **22.075%** from the supplied baseline. |
| Empty outputs | **≤0.1%**, using the frozen evaluator’s denominator. |
| Additional N_final tests | Same admission tests against O_A5000; **`S(N_final) ≤ S(best passing single arm) + 0.20 pp`**; ≥500 retained unique new hours. Its interval is a conditional selection statistic. |

The runtime and stopping rules are:

| Trigger | Required action |
|---|---|
| At validation: non-finite loss, empties **>1%**, or D **>25.346%** using the supplied full-precision P1 baseline | Stop that arm. |
| Ordinary selection-guard failure below those stopping thresholds | Mark the checkpoint ineligible; this alone is not a newly introduced early-stop rule. |
| Any required 8k admission condition fails | Exclude the bucket whole. An interval including zero is inconclusive, not evidence of equivalence or bad labels. |
| No passing corpus has ≥500 unique new hours | **STOP Phase 4.** With these inventories, MaCro failure makes continuation impossible. |
| N_final fails and no qualifying single-arm fallback exists | **STOP Phase 4.** O is not promoted; headline stays sealed. |
| Foreign compute appears | Poll at **≤60 s** intervals; save a resumable checkpoint and release the card within **5 minutes**. Do not wait for the next scheduled 2k checkpoint or interfere with the foreign process. |
| Resource limit reached | Pause/release safely. A5000 qualification: ≤16.5 GiB, watchdog 17.0 GiB. A6000: 20 decimal GB cap, watchdog 18.0 GiB. One arm per card. |

Keep hourly loss and latest available WER reporting, with the checkpoint step identified. No FLEURS or protected one-shot decoding during these decisions.

Finally, correct the planning arithmetic: six distinct 8k runs are approximately **54.5 GPU-h at 880 steps/h**, and continuation approximately **161.4 GPU-h**, before unmeasured evaluation overhead. At the frozen all-three-bucket row count, H corresponds to approximately **29.9 nominal epochs**, not 16–18. Preserve H and the registered stopping rules; correct the explanatory text.
