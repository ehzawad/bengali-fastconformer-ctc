**STOP pilots.** The current evidence does not close Gate 0, including for O. Continue ingestion, reconciliation and explicitly designated Gate-0 GPU checks. My picks are: **two cards with an additional O control; RESPIN cap 5 per sentence–dialect; run Vaani if it passes admission; exclude SPRING at 5–15% script loss under the current contract; H=150k with longer plateau patience; accept M’s design provisionally.**

These are recommendations for registration before pilots. I have not verified the queued audits or underlying manifests.

**Q1 — Allow two-card pilots, with a full control on each card.**

Identical seeds and bf16 settings do not guarantee identical numerical trajectories. PyTorch explicitly limits reproducibility guarantees across platforms and configurations. [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.9/notes/randomness.html)

Register:

- **O_A5000 remains the canonical control. Add O_A6000 for 8,000 steps.** Freeze arm-to-card assignments before results. Run these two controls first.
- Run a **200-update sanity check** using the identical O input stream on both cards: **50,400 presentations each**. Require exact cut order, target hashes, augmentation parameters and LRs. Record software versions, GPU UUIDs, precision/kernel settings, initial weights and BN buffers.
- Require finite losses and gradients, and **≤1% relative difference between corresponding 20-step mean losses**. This is an engineering tripwire; it cannot establish equivalence at a 0.3-pp WER admission margin.
- At 8k, require controls to agree within **0.10 pp S, 0.20 pp D, and 0.30 pp for every D source and the conversational slice**. Evaluate decision checkpoints through the same A5000 evaluator configuration.
- Every A6000 bucket must pass the registered admission comparisons against **both** O_A5000 and O_A6000. Keep the common O_A5000 comparison for cross-arm reporting. If control agreement fails, conduct decision-making pilots on the A5000.
- Run N_final on the A5000. An A6000 fallback winner transfers its complete 8k resumable state to the A5000; verify that transfer path beforehand.

These margins are proposed policy tolerances, not established hardware equivalence bounds. A sentence bootstrap also measures evaluation-sample uncertainty conditional on the trained checkpoints; it does **not** capture training-seed variance or hardware-by-corpus interactions.

The extra control costs approximately **10.91 GPU-h including four validations**, using the A5000 rate. It does not increase the four-bucket multiplicity count.

**Q2 — Pick 5 distinct speakers per `(normalised sentence, official dialect)`, with an aggregate ceiling of 25 recordings per sentence.**

Repeated prompts across different speakers and dialects provide acoustic variation. A text-only cap of five can discard that variation without identifying a label-quality defect. This is a defensible pre-pilot amendment because its rationale is corpus structure.

Register the exception narrowly:

- Non-RESPIN recordings retain the existing global allowance of **five**, with O reserved first.
- RESPIN may add at most **five distinct speakers per sentence–dialect cell**.
- Total retained recordings across all sources for such a sentence must remain **≤25**.
- Select deterministically by hash, with no WER ranking. Missing dialect metadata receives no expanded allowance.
- All protected-text, protected-audio, historical-exclusion and reference-conflict rules apply before this cap.

The arithmetic is:

| Rule | RESPIN row ceiling | Hours at the incoming mean duration |
|---|---:|---:|
| Five per sentence | 100,190 | 138.7 h |
| Five per sentence–dialect, five dialects | 500,950 | 693.7 h |

**Neither hour figure is a hard bound or a forecast.** The second requires prompts to recur across dialects with enough distinct speakers. If dialects mostly use different sentence inventories, the dialect cap may retain little more than the original cap.

Publish the sentence × dialect occupancy table, distinct-speaker counts per cell, and retained hours per dialect before accepting the estimate. **32.2 recordings per sentence does not establish 32.2 distinct speakers per sentence.** Protected matches, O reservations and other exclusions reduce these ceilings further.

Do not raise the cap again if the result misses 500 new hours.

**Q3 — Run O+vaani if it passes Gate 0.**

My subjective prior, conditional on accurate retained labels, is:

| Outcome versus O at 8k | Expected direction and rough magnitude |
|---|---|
| D conversational slice | Modest improvement: approximately **−0.3 to −1.0 pp** |
| Overall D | Small improvement: approximately **−0.1 to −0.3 pp** |
| FLEURS | Approximately flat to slightly worse: **0 to +0.3 pp** |

These are judgmental expectations, not measured estimates or confidence intervals. Image-description monologues, regional orthography and verbatim false starts provide only partial transfer to IndicVoices conversation. Their target conventions may transfer less well to read Wikipedia.

Vaani represents approximately **13.8% of rows in O+vaani**, before further exclusions. That is enough to affect training. Its arm costs about **10.91 GPU-h**, and the owner explicitly requested checking each addition. An owner prior is not an exclusion ruling.

Keep faithful false starts if they satisfy the registered convention. Do not silently rewrite them into fluent text. FLEURS remains sealed regardless of this prior.

**Q4 — Drop whole Latin-bearing rows; at 5–15% loss, exclude SPRING under the current contract.**

Deleting only the Latin words would create incomplete references. Transliteration, translation and model-written substitutions would introduce unregistered targets.

There is a consequential threshold conflict: council C fixes admission-rate denominators to incoming candidates, while G0.9(a) requires **≥99% compliant rows**. Therefore:

- At **≤1% incoming script violations**, remove those rows and continue other gates.
- At **>1%**, the bucket fails the stated 99% criterion.
- At **5–15%**, it fails; the additional owner-review flag does not itself authorize admission.

My pick is to preserve the strict rule for this phase. An owner decision to allow more would require an explicit amendment before freezing, stating the replacement threshold and the resulting bias toward speech without Latin-script code-switching. Do not report “100% compliant after filtering” as evidence that the incoming 99% gate passed.

Freeze R1+R2 as the declared SPRING bucket; report releases separately, but do not invent a passing release-only bucket after seeing losses.

**Q5 — Keep H=150,000; extend plateau patience.**

At 2.3 million rows and batch 252:

- One nominal epoch: **9,127 updates**.
- 150k: **16.43 epochs**.
- Matching 18 epochs: **164,286 updates**, not 200k.
- 200k: **21.91 epochs**.

Under the illustrative RESPIN retention of 500,950 rows, and holding everything else constant, the corpus falls to approximately **2.155 million rows**, making 150k about **17.54 epochs**. Further exclusions increase that number. The current evidence does not justify 200k.

Register:

> Keep the 1,000-step warmup and cosine horizon H=150,000. At step 50,000, initialise a plateau anchor to the best eligible S observed so far. Reset the anchor and patience counter when an eligible checkpoint improves the anchor by at least 0.20 pp cumulatively. Otherwise increment the counter at each scheduled validation. Stop after 10 consecutive non-reset validations.

That gives **20k-step patience** and an earliest plateau stop at **70k**. Small successive improvements accumulate toward the reset; each individual validation need not improve by 0.20 pp. Keep the 20k progress gate unchanged.

Increasing H by 50k would cost approximately **68.2 additional A5000 GPU-h including validation**.

Also correct “2.3× Phase 1”: **2.3M / 552,932 = 4.16× the rows**. The provisional 3,370 hours are approximately **3.42×** Phase 1’s 984.1 hours.

**Q6 — M’s design is acceptable; its cleanliness and detectable effect remain unestablished.**

**1,964 distinct sentences, one recording each, and 3.50 hours** satisfy the specified size requirements. Multiple recordings per sentence are not required. Having more distinct sentences can be preferable for text coverage.

However, “historically unexposed” is still provisional until the audio audit finishes. The 7,602 sentence-clean groups are not yet proven audio-clean groups.

There is **no defensible numerical minimum detectable WER difference from hours and recording count alone**. You need reference-word counts and paired error differences.

For sentence \(i\), let \(d_i\) be the difference in edit counts, \(w_i\) its reference-word count, and \(\delta=\sum d_i/\sum w_i\). A useful approximation is:

\[
SE_{\rm pp}
=
100\frac{\sqrt{\frac{n}{n-1}\sum_i(d_i-\delta w_i)^2}}
{\sum_i w_i},\qquad n=1964.
\]

For planning, suppose the corresponding length-adjusted sentence influence has SD \(s\), measured in percentage points:

| Assumed \(s\) | SE of ΔM | Observed improvement needed for a marginal 95% interval below zero | Approximate 80%-power detectable improvement |
|---:|---:|---:|---:|
| 10 pp | 0.226 pp | 0.44 pp | 0.63 pp |
| 20 pp | 0.451 pp | 0.88 pp | 1.26 pp |
| 30 pp | 0.677 pp | 1.33 pp | 1.90 pp |

At the four-arm critical level, **99.375th percentile**, the middle scenario becomes approximately **1.13 pp observed improvement** or **1.51 pp for 80% power**. These are illustrative calculations, not measurements of M; the actual admission test concerns S.

If only M improves, a **0.30-pp S gain** requires **1.50 pp on M at weight 0.20**, or **1.00 pp at weight 0.30**.

Ten thousand bootstrap draws do not create finer experimental resolution; they approximate the sampling distribution. With one recording per sentence, sentence resampling is utterance resampling. Repeated speakers can still create dependence and overly narrow intervals. Report speaker concentration where identities exist, and qualify unknown-speaker inference. [ASR blockwise-bootstrap research](https://www.isca-archive.org/interspeech_2020/liu20c_interspeech.html)

**Q7 — Exact clauses requiring reconciliation before launch**

| Current text or clause | Required replacement or clarification |
|---|---|
| §1: “`MAX_PER_TEXT = 5` … across all sources” | Add the explicit RESPIN **5-per-sentence–dialect / 25-total** exception from Q2, including O precedence and distinct-speaker selection. |
| §2: “R … selection (only if MADASR admitted)” and S conditional on admission | Fix R inclusion and S weights **once after Gate 0, before the first pilot**, based on Gate-0 eligibility. Keep them fixed even if MADASR later fails its arm. Different arms must never be compared using different S definitions. |
| G0.6: “the … 50 madasr clips count only towards the 0-catastrophic clause” | Delete this exemption. Macro random: **≤5/100 material**. Each other new bucket: **≤2/50 material**. **Zero catastrophic anywhere**, including the 20 flagged macro clips. |
| G0.9(a): “≥99%” together with “losing >5% … flagged for owner review” | State **incoming candidate denominator**, >1% failure, and that >5% triggers an additional report without overriding failure. |
| G0.9(b): incoming cross-source duplicates, together with O precedence | Separate duplicate **incidence** from which copy survives. Keeping O does not make an incoming duplicate disappear from the rate. Specify the counting unit and count each flagged row once. Apply the declared rule to O as well. |
| §5: validations “at 2k/4k/6k/8k … (+ Vaani/SPRING official validation splits as one-shot reports)” | Delete the parenthetical. Those splits are decoded **only after final checkpoint selection**. |
| §2 one-shot row includes “conversational slice of D” | Move it to repeatedly evaluated selection/guard diagnostics. It cannot simultaneously be a pilot guard and a one-shot report. |
| Kathbath K: “no training rows are removed for it” | Replace with “No rows are removed **solely for shared speaker identity**; all protected sentence/audio exclusions still apply.” |
| §2 protected-panel table omits Vaani/SPRING official splits | Add Vaani validation/test and SPRING dev/eval explicitly, with manifest hashes and their one-shot roles. Remove unavailable Kathbath test from scheduled evaluation. |
| §3: “Pre-GPU gates … CPU only” | Replace with “Pre-pilot gates: CPU audits plus explicitly listed Gate-0 GPU qualification.” Include restoration, disagreement, profiling, cross-card checks and replay. |
| §1: “final arm manifests … frozen” before any pilot, versus N_final depending on winners | Freeze O and immutable per-bucket retained shards; define N_final as their deterministic selected union. Precompute/hash possible unions if necessary. **No post-result backfilling of unused cap slots.** |
| O priority versus exclusion of entire conflicting audio components | Resolve every candidate alias/reference conflict **before freezing O**. A later conflict that changes O invalidates existing arm comparisons. |
| §2 P1 regression guards versus §5 O-relative guards | State that bucket/continuation eligibility requires **both** the applicable P1-relative guards and O-relative pilot guards. |
| §4: “≈27 epochs of 1.4 M rows” and existing plateau sentence | Replace with final-manifest epoch accounting and the Q5 plateau rule. |
| §5: “equal exposure” | Replace with **equal optimizer updates and 2.016M presentations**. These are unequal epochs and unequal O exposure: O gets approximately **3.71 epochs**, versus **0.88** for the uncapped provisional N_final. |
| §1/§2 source and M counts presented as final | Publish staged counts: original inventory → registered cleaning → leakage removal → deduplication → cap → retained. Reconcile **29,583 → 29,462** explicitly. Keep O and M provisional until audio closure. |
| §7 budget and §0/§5 competing totals | Replace with the consistent ledger below and actual device-specific measurements. |

The bootstrap implementation also needs one explicit sentence: **use paired resamples across systems; shared sentence clusters across panels must receive coupled resampling weights before forming S.** Do not assume panels are independent without checking their overlap.

The revised illustrative ledger is:

| Item | GPU-h |
|---|---:|
| Gate-0 qualification allowance | 8.0 |
| Seven 8k runs: two controls, four buckets, N_final; including validation | 76.4 |
| Continuation 8k→150k, including 71 validations | 193.7 |
| Final evaluation/export/latency | 8.0 |
| Reserve | 12.0 |
| **Soft planning total** | **298.1** |

This assumes **745 training steps/h on both cards**, plus 2.6 minutes per validation. Replace it with measured rates. A 1,000-update trajectory costs **1.34 GPU-h** at that rate; independently executing both reference and replay trajectories costs **2.68 GPU-h**, before other work.

Finally, resolve memory units: **20 decimal GB = 18.626 GiB**, whereas **19 GiB = 20.401 decimal GB**. Until clarified, enforce the stricter 20-billion-byte ceiling, with a lower watchdog threshold, including process overhead.

**Q8 — Ordered launch checklist and concurrency**

1. **Register the amendments and close the inventory.** Finish SPRING or formally exclude/defer it before freezing this experiment. Lock source versions, protected sets, counting denominators, caps, score weights, card assignments and transformations.

2. **Complete CPU admission and leakage work across the entire declared universe.** Run every text key, full PCM hashing, fingerprint matching, original-reference conflict checks, tokenizer/feasibility checks and incoming G0.9 rates. Include Vaani/SPRING protected splits and complete historical Phase-1/2/3 exposure for M. Hash already-complete files while SPRING ingestion proceeds; the final coverage reconciliation must include everything.

3. **Close the fingerprint uncertainty.** The **17,920/29,462 = 60.82% unresolved queries** are neither confirmed leakage nor clearance. Calibration must recover **at least 198/200 positives for each of six transforms**, with zero false confirmations in the prescribed negative sets. Do not tune thresholds on protected examples. Resolve candidates or quarantine their implicated components/groups. Unused quarantined MaCro rows need not become clean, but **no retained training component or selected M group may retain an unresolved relevant match**.

4. **Build retained shards and establish feasibility.** Apply protected exclusions before caps, reserve O, and publish actual unique hours. Verify M still meets its hours/group requirements. Verify the Gate-0-passing candidate pool has **≥500 unique new hours**. The 749 D and 645 holdout sentence matches require counts of removed recordings and hours; those sentence counts cannot simply be added because their sets may overlap.

5. **Complete the human boards and GPU qualification.** The specified listening work is **270 judgments**: macro 100 random +20 flagged, and 50 each for RESPIN, Vaani and SPRING. Sample from retained shards. Obtain owner judgments and apply the corrected material-error rules. In parallel, finish zero-update restoration, P1 baselines including clean M, longest-clip profiling, cross-card checks and the **1,000-update interrupted/uninterrupted replay across an epoch boundary**. Verify full-state resume and any proposed A6000→A5000 transfer.

6. **Freeze and issue an explicit Gate-0 pass record.** Hash manifests, exclusion ledgers, protected panels, code, tokenizer, initial state, baseline hypotheses and configuration. Assert that every arm contains exactly the frozen O rows and targets. With R included, the supplied baselines imply  
   **S(P1) = 16.883 + 0.20·M(P1)**; M’s missing baseline prevents completing that gate today.

7. **Launch the two O controls concurrently.** After their registered agreement check, run the four bucket arms in two fixed waves, skipping Gate-0 failures. Then run N_final on the A5000. Recheck **≥500 unique new hours for the actual selected union or fallback**, not merely the earlier candidate pool.

CPU audits and owner listening can overlap Gate-0 GPU qualification. Cross-card checks naturally use both cards; other qualification tasks can occupy one card each. **No 8k pilot, including O, starts while Gate 0 remains open.**

Keep one job per card. Check foreign compute before launch and poll during execution. Hourly-only detection plus a 2k checkpoint cadence could leave approximately **3.7 hours** before release of the card. Register **≤60-second polling and a resumable checkpoint/release within five minutes of detection**, without touching the foreign process. Hourly reports can show the latest scheduled WER, clearly labelled with its checkpoint step.

The immediate blocker is evidence: the retained hours, clean M, incoming duplicate rates, listening decisions and replay qualification are still unverified.
