**STOP pilot training for now.** The restoration and memory results are encouraging, but the supplied evidence does not close Gate 0. Audio coverage is incomplete, the human audits are outstanding, and a 40-step replay does not satisfy the registered 1,000-step replay. There are also consequential contradictions in the admission rules.

This is an audit of the supplied measurements, not an independent reproduction.

**Q1 — Keep G0.9 on O. Do not restore the literal Phase-1 corpus.**

The measured change removes **8,901 rows, 1.61% of rows and 1.17% of hours**. The 3.9 hours of low-density OpenSLR prompts are only **0.40% of the original hours**. Those prompts are not necessarily incorrect, but their removal follows the registered density rule. Removing arithmetic expressions and unsupported target conventions is also acceptable.

O answers: “What does continued training on the cleaned old corpus achieve under this training recipe?” The untouched P1 checkpoint supplies the historical comparison. O is not a replication of original Phase-1 training.

**However, freeze O identically across arms.** The present global hash ordering and `MAX_PER_TEXT=5` can let newly added rows displace old rows. Then O+bucket changes both additions and removals. Reserve O’s existing recordings and sentence slots before admitting new recordings; resolve all exclusions before freezing O.

**Q2 — Pick (a): keep digits as registered.**

Do not drop valid numeric speech. Do not change target conventions in response to newly inspected headline-test properties.

**18.3% of sentences containing digits does not imply an 18.3-point WER opportunity.** The relevant quantity is the number of word errors attributable specifically to digit-script differences divided by total reference words. For otherwise unchanged hypotheses, correcting \(k\) isolated script-only substitutions among \(N\) reference words saves approximately \(100k/N\) WER points. For example, 100 such substitutions among 10,000 words would be 1 point—not an estimate for this dataset.

Mapping training digits to ASCII might reduce some substitutions, but could introduce mismatches on Bengali-digit panels and changes tokenization. Its net training effect is unmeasured. **Budget no numeric WER gain from the unchanged policy.** Produce the registered diagnostic after the single scoring event.

**Q3 — Prefer the existing admissible CV row when targets agree; quarantine conflicting references.**

Decoded-audio identity establishes recording identity, not transcript correctness. For every CV–MaCro audio component:

- Compare the **original human references after the frozen surface**, retaining both originals in the audit ledger.
- If targets agree, keep the already-admissible CV training row and reject its MaCro copies. Count **zero additional unique hours**.
- If targets disagree materially, do not choose by source prestige, Whisper WER, P1 disagreement, or arbitrary hash order. My conservative recommendation is to **exclude the whole conflicting component for this phase**.
- If any member matches a protected recording, exclude every training alias. Check every reference attached to the component against protected texts before discarding aliases.
- Enforce historical exclusions across audio aliases; a new MaCro ID must not re-admit an excluded CV recording.

Finish the comparison against **all CV training audio**, all other sources, and protected panels. Include re-encoded, trimmed and partially overlapping recordings; whole-file PCM hashes cannot establish absence of those overlaps.

There is also a contract problem: **362 / 29,583 = 1.22%**. If protected-audio duplicate removals count under G0.9(b), `macro_validated` already exceeds its **<0.5%** threshold. It cannot be called a clean pass by reporting zero duplicates *after* removing them. See the required ruling below.

**Q4 — Keep K in S at 0.10.**

Known-speaker, unseen-text recognition is a legitimate component of the registered objective. It does not establish generalization to new speakers.

The overlap affects **58.9% of the original Kathbath training rows**. Furthermore, K has **17 speakers**, not 2,236 independent speaker observations. Sentence-clustered intervals do not establish uncertainty over unseen speakers.

The weight has a meaningful consequence: a **3-point K improvement contributes 0.30 points to S**, potentially satisfying the point-improvement requirement with other panels unchanged. Accept that interpretation explicitly.

Delete the contradictory instruction that K’s speakers are excluded from training. Keep the measured P1 K baseline, **14.67%**, and label K consistently.

**Q5 — Keep the Latin-row filter. Measure prevalence; do not invent an estimate.**

“doggy” and “pt” establish presence, not prevalence. They cannot support a defensible numerical loss estimate.

For capacity planning only, if affected rows have representative duration:

| Latin-row loss | Approximate SPRING hours removed |
|---:|---:|
| 5% | 32.4 h |
| 10% | 64.8 h |
| 20% | 129.5 h |

Run the full transcript scan separately for R1 and R2, reporting both row and hour loss. Code-switching can be correctly transcribed speech; exclusion here expresses the chosen target convention, not proof of bad annotation.

**Above 5%, my recommendation is to jettison SPRING for this phase.** Precisely, the current contract says **owner review**, not automatic exclusion. Therefore admission must stop pending that ruling; the orchestrator cannot silently waive it. Do not delete only the Latin words and retain their audio—that would create incomplete labels.

**Q6 — Keep 8,000 steps for every eligible bucket. No direct admission of `macro_validated`.**

Seven is a maximum, not a requirement to run failed or unavailable buckets. A small bucket still requires its own arm if admitted.

Different corpus sizes do **not** invalidate the comparison. They define its interpretation: benefit under the same optimizer-step budget and the specified sampling policy.

Under uniform row sampling, using the preliminary counts:

- O receives **3.71 corpus passes** in 2.016 million presentations.
- O+`macro_filtered` receives approximately **1.84 passes**, before further exclusions.
- Approximately **1.014 million presentations** in that arm come from MaCro.

Thus, the experiment combines new-data benefit with reduced rehearsal of O. It does not isolate intrinsic label quality, equal epochs, equal audio hours presented, or eventual converged performance. Shortening everything to 6k would change the registered question without evidence that the shorter endpoint is adequate.

Three statistical issues remain:

1. **Multiplicity.** Five marginal 95% intervals do not provide joint 95% coverage. Illustratively, five independent favorable 2.5% tails give an 11.9% chance of at least one occurrence; shared O and the −0.3-point requirement change the actual rate. Add a simultaneous admission bound before seeing results.
2. **Training variability.** Sentence bootstrapping conditions on the trained checkpoints. It does not measure sensitivity to training seeds.
3. **Adaptive selection.** N_final is constructed from observed arm outcomes. Its ordinary CI is a selection statistic, not an independent confirmation of the selection procedure. These distinctions matter when interpreting model comparisons and selected results. [Dror et al.](https://aclanthology.org/P18-1128/), [Cawley and Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html)

A CI containing zero means **benefit was not demonstrated at 8k under this protocol**. It does not prove equivalence, harmful labels, or uselessness at longer training. Nevertheless, the registered admission decision is **exclude**.

The revised arithmetic is **75.2 pilot GPU-h + 190.6 continuation GPU-h + 26 other/reserve GPU-h = 291.8 GPU-h**, before any evaluation overhead absent from the throughput measurement. The 236-hour ledger is stale.

**Q7 — Conservative removal is acceptable; adjudication is not required to discard candidates.**

The 1,752 candidates are only **0.32% of the original old-corpus row count**, although the candidate list spans more than that corpus. Removing them is a reasonable conservative decision. Publish their actual source distribution, hours and reasons.

Call them **candidate-based exclusions**, not 1,752 confirmed leaks. Do not add the per-panel counts: one row may match several panels.

The audit still needs coverage for:

- Remaining new buckets and newly constructed M—not only old train and `macro_validated`.
- Vaani and SPRING protected splits.
- Short texts below the 20-character fuzzy threshold.
- Split/merged sentences, embedded excerpts and local containment.
- Digit words versus digit strings, transliteration, spelling variants and joiner differences.
- Every original transcript attached to duplicate audio.

The char-5-gram bound also needs an exhaustive check on manageable fixtures showing that it does not prune qualifying matches. Report retrieval coverage separately from confirmation accuracy.

**Q8 — Most likely remaining defect: historical audio exposure hidden by a different dataset ID or transcript.**

Specifically, **M may contain recordings already consumed by the Phase-1 warm start through CV**, despite passing a historical sentence lookup. This is a risk inferred from the demonstrated MaCro–CV overlap, not an already measured M contamination count.

Removing current training rows does not erase prior exposure from initialized weights.

The check is to compare candidate M recordings against the **complete historical P1/P2/P3 training inventories**, using decoded PCM and calibrated fingerprints, while comparing their original reference variants against historical sentence inventories and tokenizer-training text. Carve M only from the surviving components.

Also distinguish historical exposure on D and holdout from current-training leakage. The 163 and 721 fuzzy candidates warrant disclosure and investigation; they are not automatically confirmed historical duplicates.

**REQUIRED changes to SPEC before pilots**

Replace conflicting clauses rather than appending another interpretation. The following is proposed exact wording.

1. **Freeze admission artifacts and O.**

   > “Before any pilot optimizer update, freeze the source inventories, protected-panel manifests, transformation code, exclusion ledger and final arm manifests with checksums. Report raw counts, overlapping reason counts and disjoint removal counts with a fixed precedence. The reported 544,031 rows / 972.6 h is provisional until all audits finish. Every O+bucket arm contains exactly the frozen O rows and targets. Deduplication and MAX_PER_TEXT reserve O’s retained recordings first; new rows occupy only remaining eligible slots.”

2. **Resolve duplicate-reference identity before retention.**

   > “Preserve all source aliases and original human references for each confirmed audio component. Check every alias and reference against protected sets and historical exclusions. Any protected-audio match excludes every training member. For equivalent frozen-surface references, retain an already-admissible O recording before a new-source recording. Components with conflicting frozen-surface references are excluded from all training arms for this phase. ASR agreement does not select the retained reference.”

3. **Resolve G0.9(b) explicitly. My conservative ruling is to enforce the incoming defect rate.**

   > “G0.9(b) uses the ingested allowlisted candidate count before defect removal as its denominator. Its numerator is the union of empty-target, decode-failure and duplicate-removal rows under the registered retention rules, including duplicates against protected panels and other sources. Count each removed row once. The rate must be <0.5% per bucket; post-removal rates cannot substitute for this admission rate. On the supplied counts, macro_validated has at least 362/29,583 = 1.22% duplicate-removal rows and is excluded from training.”

   If the owner instead intends cross-corpus overlap to be exempt, that requires an **explicit post-measurement amendment**. It is not a pass under the conservative interpretation above. Excluding its training bucket need not discard an independently audited, unexposed M panel.

4. **Define other admission-rate denominators and SPRING handling.**

   > “Compute duration, script, tokens/s and feasibility admission statistics before removing failures of the corresponding rule; publish both pre-removal and retained distributions. Identify arithmetic expressions from the original NFC reference before punctuation removal destroys their operators. Measure script-filter losses for SPRING R1 and R2 separately and combined. A script-filter loss >5% blocks admission pending an explicit owner ruling; no automatic waiver or deletion of spoken Latin words from retained targets is permitted.”

5. **Close current and historical protection coverage.**

   > “Run text and audio audits over every candidate training bucket against every protected panel, including Vaani validation/test and SPRING dev/eval. Audit candidate M against historical Phase-1/2/3 training audio and sentence inventories and tokenizer-training sentence text. M must satisfy its registered hours and sentence-group requirements after these exclusions; otherwise STOP. Record historical exposure separately from current-training overlap. Test-set audits return only identifiers and counts.”

6. **Make panel roles and weights invariant across comparisons.**

   > “K is a known-speaker, text-disjoint selection panel with weight 0.10; no Kathbath training rows are removed solely for sharing K speaker IDs. Freeze the selection weight vector before pilots. Include R according to MADASR’s Gate-0 eligibility, and use the same vector for every arm and checkpoint comparison regardless of which buckets later pass their ablations.”

7. **Give every conditionally admitted new source a meaningful listening gate.**

   > “Retain the registered macro_filtered audit of 100 uniform clips plus 20 disagreement-selected clips. For MADASR, Vaani and SPRING, audit 50 uniformly sampled clips per bucket from the final retained candidate manifest. Each 50-clip stratum passes with at most 2 material errors and zero catastrophic errors. All targeted listening strata require zero catastrophic errors. Publish selections, judgments and counts before the bucket’s pilot. Machine disagreement is not a human audit.”

   This explicitly strengthens the present MADASR rule, which otherwise permits **50/50 material errors provided none is classified catastrophic**.

8. **Specify the pilot estimand and simultaneous bound.**

   > “Retain the fixed 8,000-step endpoint and all registered point-effect and regression thresholds. Freeze and publish the row-sampling policy and seeds; log presentations and presented audio hours by source. Bootstrap paired sentence clusters and recompute each panel’s corpus WER before forming weighted S; do not average utterance WERs. Preserve shared-cluster dependence across panels and shared-control dependence across arms. Report marginal 95% intervals. For m predeclared eligible single-bucket comparisons, additionally require the bootstrap upper quantile at 1−0.025/m to be below zero. Use 10,000 draws and seed 20260905. N_final intervals are conditional selection statistics, not independent confirmation. An inconclusive bucket is excluded without a claim of equivalence or poor label quality.”

   With five single-bucket comparisons, that additional upper bound is the **99.5th percentile**.

9. **Close the small-fallback loophole.**

   > “G0.8 must hold both before pilots and for the corpus chosen for continuation, including a single-arm fallback: at least 500 unique new hours remain after all exclusions and deduplication against O. If the selected corpus fails this condition, STOP Phase 4 unless the owner explicitly amends the minimum before continuation.”

   A roughly 45-hour `macro_validated` fallback cannot independently satisfy this condition.

10. **Complete the actual replay and resource gates.**

   > “The 40-step smoke replay does not satisfy resume acceptance. Before pilots, complete the registered interrupted-versus-uninterrupted 1,000-step replay spanning an epoch boundary, verifying identical sample IDs and LRs and reporting loss and state comparisons. Repeat resource profiling on the final admitted manifests’ longest clips. Record all GPU gate work as explicit exceptions to the ‘CPU only’ heading and charge it to Gate 0.”

11. **Remove contradictory one-shot language and preserve the numeral policy.**

   > “The conversational slice of D is a repeatedly evaluated selection guard, not a one-shot report. Vaani/SPRING one-shot evaluation results are produced only after final checkpoint selection and cannot influence admission or selection. Keep digits as written in every training arm. No further headline-test-driven target or scorer changes are permitted. Produce the digit-error diagnostic from the single registered final scoring event.”

12. **Replace the stale budget forecast.**

   > “At 745 optimizer steps/hour, seven 8k pilots require approximately 75.2 GPU-h and continuation from 8k to 150k requires approximately 190.6 GPU-h. Including the existing 26 GPU-h for other work and reserve gives approximately 291.8 GPU-h before any unmeasured validation overhead. Re-measure throughput on final arm manifests, include evaluation time, and record the owner’s revised budget ruling. The orchestrator does not independently increase the authorized plan.”
