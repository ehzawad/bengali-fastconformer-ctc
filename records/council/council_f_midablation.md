**Q1 — Pick (a): keep S frozen for admission and final selection.**

Keep `S = 0.60D + 0.20M + 0.10R + 0.10K`, eligibility guards, and the 0.05 pp earlier-checkpoint tie rule.

Switching to D after observing that MaCro’s benefit concentrates on M would be post-hoc selector shopping. Option (c) is also ill-defined: D and K represent domains already present in O; R represents admitted training data if MADASR passes. Same-domain evaluation is intentional here; verified transcript/audio separation addresses a different concern.

S measures the registered panel mixture. It does **not** establish that the selected checkpoint is optimal for FLEURS. Do not change it.

**Q2 — Pick: retain checkpoint ineligibility; add no conversational abort threshold.**

The observed trajectory argues against adding a three-consecutive-failures rule: it would have terminated MaCro at 6k, before its registered 8k pass.

The pass is nevertheless fragile:

- Against O: **21.89 − 21.47 = +0.42 pp**, leaving **0.08 pp**.
- On 6,628 reference words, that margin is approximately **five additional edits**.
- Admission uses O as comparator; main-run eligibility uses the frozen P1 baseline. P1’s reported **21.57/21.58** discrepancy must be reconciled from its exact edit count. Its main-run ceiling is approximately **22.08**, not the admission ceiling of **21.97**.

Admission also does not automatically establish main-run eligibility for every source: “within +1.0 pp of O” and “within +1.0 pp of P1” are different tests.

Checkpoint ineligibility is sufficient to protect the shipped model **provided** the implementation preserves the selected eligible checkpoint independently of checkpoints with better but ineligible S. Ineligible checkpoints must never reset plateau patience. The existing **20k progress gate** and **50k/70k plateau mechanism** provide the registered protection against wasted computation.

**Q3 — Pick: retain conditional selection; make the fallback explicitly feasibility-first.**

(i) Permissible one-sentence claim:

> “Among the frozen corpus candidates evaluated at 8,000 updates, this corpus was selected by the registered weighted-dev, regression-guard and minimum-hours rules; the selected-union comparison is conditional on those development results and does not independently establish an optimal corpus or transfer to FLEURS.”

Repeated use of development results can bias apparent selection gains; that is why the conditional qualification matters. [Cawley and Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html)

For N_final, retain **both** point-estimate requirements and the bootstrap/guard requirements. If MaCro remains the best passing single:

- Against O: `S(N_final) ≤ 21.073 − 0.300 = 20.773`.
- Against the best single: `S(N_final) ≤ 20.531 + 0.200 = 20.731`.

Thus **20.731** is the tighter point threshold. If another passing single has lower S, use that lower S for the comparator—even if that single cannot independently meet the hours requirement.

(ii) **Clarify the fallback; do not tighten statistical thresholds.** Rank only passing single arms whose own frozen corpus satisfies **≥500 unique new hours**. Currently that set can contain only MaCro:

- MaCro: **726.5 h**.
- MADASR: **149.0 h**.
- Vaani: approximately **117.4 retained h**.
- Both non-MaCro additions together: approximately **266 h**.

If N_final fails, use the admitted MaCro single. Do not search additional unions, raise repetition caps, or admit portions of failed buckets. A lower-S but undersized MADASR arm cannot be the fallback.

**Q4 — Pick: keep H = 150,000 and the existing plateau rule.**

The corrected nominal epoch counts are:

| Corpus | At 8k | At 70k | At 150k |
|---|---:|---:|---:|
| O + MaCro, 1,084,696 rows | 1.86 | 16.26 | **34.85** |
| All three, 1,265,884 rows | 1.59 | 13.93 | **29.86** |

Thirty-five nominal epochs warrant monitoring; they do not justify changing a registered schedule after seeing pilot results. Repetition count alone does not establish overfitting.

The verified 8k scheduler state is valuable. Preserve it, the optimizer, sampler and RNG state. **No second warmup, compressed cosine, LR reset or epoch-based replacement horizon.**

Keep the 50k anchor, cumulative **0.20 pp** improvement requirement, and **10** non-reset validations, with earliest plateau termination at **70k**. An external cancellation could truncate this unchanged schedule, but would need to be reported as a departure from the stopping plan.

**Q5 — Pick: finish MADASR and Vaani.**

At MADASR update 700, approximately **15,300 pilot updates** remain across those two arms: **18.4–24.7 GPU-h** at 620–830 updates/h, plus evaluation overhead. A distinct N_final adds another **9.6–12.9 GPU-h**.

Skipping loses:

- Evidence about the two remaining allowlisted buckets under this recipe.
- Evidence about whether dialect-rich read speech or spontaneous speech improves the current weak areas.
- The opportunity to test the registered union against the best single.
- The ability to claim that all three proposed additions received their required ablation.

Neither bucket’s smaller size establishes that it is unhelpful. Vaani’s spontaneous speech is particularly relevant to the observed conversational regression, although whether it helps remains unmeasured.

**Skipping is reasonable if the owner has a concrete deadline or resource constraint that makes approximately 20–25 GPU-h of remaining ablations unacceptable.** Then proceed with MaCro’s existing 8k state, record MADASR and Vaani as **untested**, exclude them, and retain **m=3**. Do not reduce the multiplicity penalty after seeing MaCro’s result.

Absent that constraint, complete the experiment the owner requested.

**Q6 — Pick: yes, the measured admission gain is predominantly “MaCro helps MaCro.”**

Using the rounded panel WERs:

| Panel | WER change, pp | Weighted contribution to ΔS, pp |
|---|---:|---:|
| D | −0.11 | −0.066 |
| M | **−2.54** | **−0.508** |
| R | +0.31 | +0.031 |
| K | −0.03 | −0.003 |

M supplies approximately **93–94%** of the net improvement. The rounded contributions sum to −0.546; the authoritative prediction-level result is **−0.542**. Outside M, the weighted contribution is only **−0.038 pp**.

Honest one-sentence characterization:

> “At equal 8,000-update budgets, the MaCro arm improved the registered composite by 0.542 pp, driven predominantly by a 2.54 pp improvement on held-out MaCro, while D improved 0.11 pp and conversational WER worsened 0.42 pp within the admission tolerance.”

The result supports admission under the chosen objective. It does not demonstrate broad improvement, label superiority, or a reproducible training-run effect.

Also, equal updates do not hold old-data exposure constant: adding MaCro roughly halves O’s share of presentations. This comparison measures the **whole corpus-mixture intervention**, including reduced repetition of O.

For FLEURS, **temper expectations; do not assign a numerical forecast**. Beating Phase 2 requires more than **3.13 pp** improvement from P1’s 19.67%, plus the required confidence interval. A −0.542 pp change in S supplies no calibrated estimate of that transfer.

**Q7 — Pick: correct the following lines; preserve their historical versions.**

| Exact passage | Required correction or qualification |
|---|---|
| §8j: **“Δ = −0.517 pp, 95 % CI [−0.62, −0.42]”**, with simultaneous upper **“−0.44”** | Latest measured record says **−0.542**, **[−0.736, −0.352]**, upper **−0.314**. These differences are not rounding. Identify the authoritative artifact and preserve the superseded record. |
| §8j: **“the old corpus is saturated under this recipe, so any arm's gain is attributable to its new data”** | Unsupported. O is **+0.043 pp versus P1 at 8k**, but its S is still declining: **21.54→21.32→21.17→21.07**. Neither saturation nor isolation from training/hardware variation has been established. |
| §5: **“on the A5000 — canonical control”** | The completed control ran **A5000 then A6000**. Preserve its run identifier, but describe it as the completed mixed-card reference control. |
| §8i: **“the binding cross-card evidence is the 8k control agreement, which we no longer have”** | The agreement test is **uncompleted**, not passed. Continuing under the owner’s amendment accepts a protocol deviation; partial agreement within **0.08 pp S** does not certify the missing 8k D/per-source thresholds. |
| §8d: **“Same-card replay is exactly deterministic (Δloss 0.0)”**; §8e: **“same-card runs are deterministic from update 1”** | These overgeneralize short probes and conflict with the 1,000-step result: divergence from update **285**, maximum **0.022**. Report determinism only for the tested probe lengths. |
| §8e: **“Cause (known PyTorch behaviour): CUDA dropout/mask kernels derive their Philox offsets…”** | Treat this as the proposed mechanism unless directly traced in this environment. General PyTorch reproducibility limitations do not diagnose this particular run. [PyTorch documentation](https://docs.pytorch.org/docs/2.9/notes/randomness.html) |
| Q2 premise: **“MaCro is ~43 % of presented rows”** | True for all three: **42.93%**. For MaCro alone it is **50.10%**, assuming uniform row sampling. Report the selected corpus’s actual sampling share. |
| Measured results: **“New hours: … vaani 118.1”** | Frozen arm totals imply **1,083.3−965.9 = 117.4 h**. The 118.1 h figure is pre-freeze. Use exact duration sums; the 992.8/992.9 total difference can be rounding. |
| §4: **“≈16–18 epochs of the final corpus”** | Already superseded: **29.86 epochs** for all three; **34.85** for MaCro alone. |
| §7: **“Pilots: 2 controls + 4 buckets + N_final”** and continuation **“≈194”** GPU-h | SPRING is excluded; the second control stopped at **4,038**. Remaining continuation is **171–229 GPU-h**, before additional overhead or queue delays. Charge abandoned work. GPU-hours do not guarantee calendar completion. |
| §8d: Vaani **“is excluded by the owner's listening verdict”**; §3 G0.9: **“the human audit (G0.6)”** | Superseded by §8f: Vaani passes the amended machine-consensus gate; earlier hypothesis-directed verdicts were withdrawn. Claims must say machine-audited and informally owner-spot-checked. |
| §8d: hishab **“has almost certainly trained on those corpora”** | Low WER raises an exposure concern; it does not establish training provenance. State suspected exposure and unverified independence. |
| §1: **“fingerprint dedup across all corpora”**; §3 coverage over **“EVERY candidate bucket”** | §8f records two exceptions: no train×train fingerprint dedup and no Vaani protected-split fingerprint index. Gate 0 closed with those disclosed deviations; do not describe the originally specified coverage as completed. |
| §8g: **“completion = last logged update ≥ target”** and **“a crash → re-queue”** | Insufficient for scientific termination. Progress-gate failure, plateau completion and numerical aborts must produce terminal states that the supervisor does not restart. |

Also reconcile conversational P1 **21.57 versus 21.58** using exact counts before any threshold comparison. Keep final M at **1,829 rows / 3.24 h** and its baseline at **20.74**, S(P1) **21.030**.

**Q8 — Pick: the likeliest scientific waste is continuing domain-specific improvement while broad D progress remains inadequate.**

The cheapest registered check is the **20,000-update progress gate**, enforced automatically:

- **S ≤19.530**
- **D ≤19.35**
- **Every main-run eligibility guard passes**

From MaCro’s 8k result, this requires another **1.001 pp S** and **0.86 pp D** improvement. Improving M alone cannot satisfy the D requirement.

Reaching that decision costs approximately **14.5–19.4 additional GPU-h** from 8k. Enforcing failure there avoids the remaining **130,000 updates**, approximately **157–210 GPU-h**.

Verify the supervisor’s handling of a scientific STOP before continuation. A correct evaluator coupled to an “always resume” loop that restarts rejected runs defeats the entire gate. Passing 20k still does not guarantee FLEURS success.

The **required SPEC amendments**, with exact wording, are:

1. **Authoritative result and claims**

   > “The authoritative MaCro admission result is ΔS = −0.542 pp, paired 95% CI [−0.736, −0.352], and simultaneous upper quantile −0.314 pp at 1−0.025/3. The prediction, manifest, evaluator and result artifacts are identified by checksum; earlier numerical summaries remain archived as superseded. O’s +0.043 pp change from P1 at 8,000 updates does not establish saturation.”

2. **Actual control and uncertainty**

   > “O_ref denotes the completed 8,000-update control, trained on the A5000 and subsequently the A6000; its card/update segments are recorded. Under the owner’s §8i amendment, all admission comparisons use this fixed O_ref. The original 8,000-update cross-card agreement test remains uncompleted. Bootstrap intervals describe evaluation-sampling uncertainty conditional on the fitted models and do not include training-run or hardware variation.”

3. **Eligibility and preservation**

   > “Admission regression guards use O_ref. Continuation and final-selection regression guards use the frozen, full-precision P1 baseline: every D source ≤ P1 +1.0 pp, conversational WER ≤ P1 +0.5 pp, and empty-output rate ≤0.1%. Admission alone does not certify continuation eligibility. Ineligible checkpoints cannot be selected or reset plateau patience. The checkpoint chosen by the registered eligible-S selector, including its earlier-checkpoint tie rule, is preserved independently of resumable-checkpoint rotation.”

4. **Fallback feasibility**

   > “If N_final fails admission, the fallback is the lowest-S passing single arm among those whose own frozen corpus contains ≥500 unique new hours. With the frozen inventories, only O+macro_filtered can qualify. The N_final best-single comparator still ranges over all passing singles. No additional union search or partial bucket admission follows a failed union. The multiplicity count remains m=3 if an owner-authorized arm is skipped.”

5. **Terminal stop semantics**

   > “The supervisor distinguishes resumable infrastructure interruption from scientific STOP and successful training termination. A failed 20,000-update progress gate or registered numerical/divergence abort terminates Phase 4 without automatic restart or FLEURS decoding. Plateau termination and reaching H terminate training and invoke registered final selection. ‘Always resume’ applies only to resumable infrastructure interruptions.”

6. **Current-record precedence**

   > “The current register uses final M = 1,829 rows / 3.24 h and S(P1) = 21.030; retained hours come from frozen-manifest duration sums. §8f governs the amended machine-consensus audit and its explicitly limited fingerprint coverage. Earlier contradictory audit, corpus-size, epoch and budget statements are historical. Current rates are 620–830 updates/hour; queue delay and evaluation overhead are reported separately.”

The **explicit main-run STOP conditions** are:

- **Before continuation:** no admissible corpus satisfying **≥500 unique new hours**. If MaCro does not remain validly admitted, no frozen alternative qualifies.
- **At any validation:** non-finite loss, empty-output rate **>1%**, or D **>25.35%** → scientific abort; no automatic restart.
- **At 20k:** S **>19.530**, D **>19.35**, or any main-run guard failure → terminate Phase 4. Do not salvage it by changing selectors, thresholds or corpus.
- **After the 50k anchor:** **10 consecutive non-reset validations** → plateau termination, earliest **70k**; select the preserved eligible checkpoint.
- **At 150k:** terminate training and apply the registered selector.
- **At final selection:** no eligible checkpoint → no shipment and no headline decode.
- **After the single headline event:** if WER is not **<16.54%**, or the paired 95% CI against P2 is not entirely below zero, report failure of the headline objective; no second checkpoint attempt.

An ordinary conversational guard failure makes that checkpoint ineligible. It is **not** an additional abort rule.
