**Recommendation: C—finish the original ablation, evaluate its prescribed union, then face the 20k gate. Defer CV17 to a separately registered experiment.** Do not spend 25–40 GPU-h filtering it now.

The experiment numbers below come from your packet; I have not independently verified the admission JSONs.

**Q1 — Legitimate development work; not a prospectively selected hypothesis.**

Using D/M/R/K to discover promising training data is legitimate. Calling CV17 “registered before use” does not make its discovery independent of those panels.

Distinguish three things:

- **Training leakage:** requires the registered sentence/audio audits. Shared source provenance alone does not establish leakage.
- **Adaptive development:** CV17 was selected after inspecting errors on the admission panels. This happened even if every protected recording is successfully excluded.
- **Confirmation:** CV17’s bootstrap interval would measure evaluation-sample uncertainty conditional on the chosen systems. It would not account for the process that selected CV17, or training/card variation.

Increasing m corrects the number of named comparisons under the applicable assumptions; it does **not** account for adaptive hypothesis discovery. This is the relevant distinction in work on adaptive data analysis. [Dwork et al.](https://arxiv.org/abs/1506.02629)

The quantiles are:

| Family | Required upper quantile |
|---|---:|
| m=3 | 99.1667th percentile |
| m=4 | 99.375th percentile |
| m=5 | 99.5th percentile |

**Keep m=4 now, including if CV17 is deferred.** Its slot was announced in §8l. Recompute MaCro’s upper quantile from the registered bootstrap—not a normal approximation to the published CI. Its m=3 admission remains a historical result; current eligibility requires the m=4 check. The supplied −0.314 value does not establish the new quantile.

If CV17 proceeds on these panels, label its result—and any CV-containing union—**exploratory development evidence**. Independent confirmation requires genuinely unconsulted evaluation data and a prospectively fixed comparison. Another seed, a later phase number, or splitting already-inspected D does not provide that. Do not repurpose the protected one-shots into admission panels.

Exact disclosure wording appears below.

**Q2 — Pick C: complete the existing experiment before paying for expansion.**

Specifically:

1. Finish MADASR and Vaani.
2. Run the one prescribed union if distinct.
3. Resume the selected corpus to 20k.
4. Apply the existing gate unchanged.
5. Keep CV17 outside this Phase-4 training path, whether that gate passes or fails.

Starting MaCro continuation immediately would pre-empt the running arms and union rule.

The economic comparison is **15–19 GPU-h for the selected corpus’s 20k decision versus 35–50 GPU-h for CV filtering plus one arm, or 45–60 GPU-h with two arms**, before additional auditing or a changed union.

The current gate already limits a failing continuation’s expenditure. Therefore, CV expansion cannot be justified simply as avoiding 165–190 wasted GPU-h: the contract does not permit spending that whole amount before checking progress.

Nor is the proposed benefit established:

- MaCro’s observed D gain was only 0.11 pp, but extrapolating that linearly would also be unjustified.
- Removing the remaining 0.87 pp of D error means approximately **480 fewer edit operations** at the supplied denominator.
- Literary/poetic SUBAK.KO and encyclopedic FLEURS have related written-language characteristics; they are not established as the same error distribution.
- Equal median word counts do not diagnose lexical difficulty: substitutions and cancelling insertions/deletions can produce that statistic.
- **73.8% of `other` sentence types already occur in training text.** The 244,284 clips carrying unseen sentences are **24.6% of its clips**, before exclusions.

The expected benefit of CV17 is therefore uncertain at several successive stages: admission, unique usable volume, label quality, and transfer. There is no defensible numerical success probability in this packet.

The owner’s quality preference supports preserving CV17 for a properly audited experiment. It does not require buying its entire filtering bill before using the cheaper decision already registered.

**Q3 — One model is sufficient to implement the filter; insufficient to validate its interpretation.**

The AND rule is a permissive disagreement screen. It has no demonstrated sensitivity or specificity for audio/reference defects.

| Phase-1 WER | Phase-1 CER | Decision |
|---:|---:|---|
| 90% | 20% | Keep |
| 45% | 65% | Keep |
| 80% | 50% | Drop |
| 60% | 90% | Keep: inequality is strict |

All three possibilities matter:

- A materially wrong prompt can survive.
- A correct short utterance containing unfamiliar words can fail both thresholds.
- A model can produce an apparently matching reference despite defective audio.

Thus, **“defect filter, not an easy-example filter” is too strong**. It is model-conditioned selection with a loose cutoff. The AND reduces rejection relative to OR; it does not eliminate difficulty, dialect, vocabulary, or recording-condition bias.

Phase 1 also trained on older Common Voice. Its independence from CV17 must not be asserted before historical identity checks. Adding hishab would not automatically solve this, and using Phase 1 again in the machine-consensus audit creates dependence between selection and audit.

For any later use, publish:

- Immutable model, decoder, scorer, threshold and candidate-manifest identifiers.
- Counts and hours through every exclusion stage, with the original incoming denominators preserved.
- All four WER/CER quadrants, distributions, and strict-boundary behavior.
- Retention by duration, prompt familiarity, speaker and available dialect metadata; missing metadata reported explicitly.
- Retained unique sentences, speakers and hours—not nominal release volume.
- A model-hypothesis-blinded listening audit: **200 random retained clips, 100 random rejected clips, and 50 retained clips from each discordant quadrant**, or all available clips if a stratum is smaller. Report stratum-specific material/catastrophic errors and uncertainty; do not pool this deliberately stratified sample into an unweighted overall rate.
- Original human prompts remain targets; recognizer outputs never become labels.

Zero catastrophic errors among 200 random retained clips still permits an approximately **1.49% one-sided 95% upper bound**. It is useful evidence, not certification of near-zero noise.

**Without independent listening evidence, I would withhold `other`.** Investigate validated first in the later experiment, but do not assume its 73.7 gross hours are new or that community validation guarantees expert-level accuracy.

There is also a concrete ingestion discrepancy: this unofficial mirror’s loader exposes `train/dev/test/other/invalidated`, not a `validated` split, and its download construction includes `invalidated`. Reconcile the claimed 44,121 validated recordings against an actual inventory and use explicitly allowlisted ingestion. Do not assume train+dev+test exhaust validated. [Dataset card](https://huggingface.co/datasets/fsicoli/common_voice_17_0), [loader implementation](https://huggingface.co/datasets/fsicoli/common_voice_17_0/blob/main/common_voice_17_0.py)

**Q4 — Two evidence buckets when CV is evaluated; zero CV arms in this Phase-4 recommendation.**

Choose separate `cv17_validated` and `cv17_other` buckets, each with its own admission audit and arm.

Pooling them makes approximately **94.1% of nominal hours** come from `other`. A pooled pass would establish the behavior of that mixture, not independently support either evidence tier. A pooled failure would likewise be unattributable.

The extra **~10 GPU-h** is justified once the programme commits to the larger 25–40 GPU-h filtering expense. Separate gates also prevent one tier from diluting the other tier’s incoming defect rate.

If the owner instead keeps both tiers in this Phase-4 family, register **m=5 before their outcomes**, and retest every admission at the **99.5th percentile**. Preserve all earlier results; do not reduce m when a tier fails or is skipped. That is a contingency, not my recommended current spending path.

**Q5 — Preserve the current mixture; correct the premise and investigate conversation next.**

The statement that all candidates are read speech is false: **Vaani is spontaneous image-description speech**. It does not equal dialogue or extempore conversation, but its running arm is directly relevant to this concern.

Also, error share is not marginal training value. Using the rounded decomposition:

| D component | Approximate share of reference words | Effect on D of a 1 pp source-WER improvement |
|---|---:|---:|
| IndicVoices | 52.9% | −0.529 pp |
| SUBAK.KO | 18.9% | −0.189 pp |

If everything else stayed constant, meeting the remaining 0.87 pp D requirement would need approximately **1.64 pp improvement on IndicVoices alone**, or **4.62 pp on SUBAK.KO alone**.

That makes conversational coverage important. It does not establish that more conversational hours will deliver that improvement.

My choice:

- Finish Vaani.
- Keep sampling weights frozen; upweighting IndicVoices is another intervention requiring a separate arm.
- Conduct a bounded conversational-data survey for the next registration.
- Disclose the current coverage limitation and report the conversational guard prominently.

MaCro has only about **0.08 pp admission-guard headroom** against O’s conversational baseline. That is a reason to preserve the guard, not to alter mixture weights after seeing it.

**Q6 — Keep the prescribed union-of-passers rule, with exactly one interaction test.**

Let \(P\) contain every bucket passing its complete registered single-arm rule. Build:

\[
U = O \cup \bigcup_{b\in P} b
\]

using the frozen deduplication, precedence and cap rules. Do not rank subsets or drop an individually passing bucket because its trajectory looks inconvenient.

“Add everything that passed” becomes a principled **candidate-construction rule** when fixed in advance and followed by the separate union test. It is not a claim that improvements must combine.

For this Phase-4 recommendation:

- Reuse the single-arm checkpoint if U equals that arm.
- Otherwise train **one** distinct U from init A to 8k.
- Require the complete admission rule against O and  
  **S(U) ≤ S(best passing single) + 0.20**.
- If MaCro remains best, the binding point threshold is **20.731**, tighter than O−0.30 = **20.773**.
- On union failure, use the lowest-S passing single whose own corpus supplies **≥500 unique new hours**.
- With CV deferred, only MaCro’s **726.5 h** can supply that fallback. Otherwise STOP.

No second union, bucket-removal search, weight adjustment, or mixture-cap rescue.

If CV is later admitted into an expanded experiment, recompute the eligible-hour logic from frozen marginal inventories: “MaCro must pass” would no longer necessarily hold. Do not carry that conclusion forward mechanically.

**Q7 — Recommend a 300 GPU-h spending ceiling, including the 45 already spent.**

| Item | GPU-h |
|---|---:|
| Already spent | ~45 |
| Finish MADASR and Vaani | 18–20 |
| One distinct union, if required | 0–12 |
| Selected corpus, 8k→150k, including the 20k decision | 165–190 |
| Final evaluation/export | ≤8 |
| Reserve | 12 |
| **Full-horizon total** | **~248–287** |

The 20k decision should arrive at approximately **78–96 cumulative GPU-h**, including the remaining original pilots and possible union. Its 15–19 GPU-h continuation cost is a **subset** of the continuation allocation, not an additional charge.

Two CV arms plus filtering add roughly **45–60 GPU-h**, moving the range to approximately **293–347**, before extra audits or resource slowdowns. That expansion has no demonstrated fit within the existing plan.

Recommended allocation: **0 further Phase-4 GPU-h for CV17**, and at most one CPU worker-day of admission reconnaissance off the training critical path. Only the owner raises the spending ceiling.

Two operational caveats affect that estimate:

- Shared-card throughput must be measured; dedicated-card smoke throughput is not sufficient.
- The reported **2.3 GB free is below the registered 2,500 MiB yield floor**—even 2.3 GiB is only 2,355 MiB. Reconcile actual MiB with any explicit owner exception. Sharing authorization does not supply a new numerical floor in this packet.

Keep the incomplete 8k cross-card qualification disclosed. Bootstrap intervals do not compensate for the missing hardware/run replication.

**Q8 — Biggest expansion-specific risk: counting recycled campaign recordings as new, independently screened training evidence.**

CV17 may substantially overlap older CV and MaCro in audio, not merely prompts. That threatens admission, unique-hour accounting and the supposed independence of the Phase-1 filter. **The packet does not yet establish the audio-overlap rate.**

The cheapest registered check is **G0.9(b), using incoming identity/PCM duplicate accounting before full GPU filtering**.

Using the supplied counts as denominators:

| Candidate bucket | Incoming clips | Duplicate count sufficient to fail the strict <0.5% rule |
|---|---:|---:|
| Validated | 44,121 | **221** |
| Other | 993,696 | **4,969** |
| Combined | 1,037,817 | **5,190** |

Other counted defects can cause failure sooner. Use the actual frozen ingested denominator when available.

Join stable identifiers first to locate candidates, then confirm using the registered audio identity procedure and existing digest inventory. Once confirmed duplicates alone cross the ceiling, **stop that bucket without buying the ASR pass**. A small clean sample cannot establish a pass.

Shared transcripts are not duplicate audio. Conversely, removing duplicate recordings does **not** repair a failed *incoming-rate* gate. Do not redefine the candidate set as “new CV only” after measuring failure.

**REQUIRED SPEC amendments — proposed exact wording**

1. **CV17 status and scope**

> “CV17 was proposed after inspection of Phase-4 development results. Its Phase-4 ingestion/filtering/ablation plan is deferred. No CV17 recordings or targets enter Phase-4 training, and no CV17 ASR filtering or pilot GPU work is charged to this phase. CPU admission reconnaissance may proceed without delaying the registered runs or changing their frozen artifacts. A failed Phase-4 progress gate does not activate a CV17 rescue run.”

2. **Multiplicity and discovery disclosure**

> “The Phase-4 admission family retains m=4 because CV17 was announced in §8l, although its arm is deferred. Every current admission requires the registered paired-bootstrap upper quantile at 1−0.025/4 = 0.99375 to be below zero, together with all existing point and regression criteria. Earlier m=3 results remain historical records. Skipping or deferring an arm does not reduce m.”

> “The CV17 proposal was generated from error analysis of the same development panels used for admission and checkpoint selection. It was not prospectively chosen independently of those results. No CV17 data were used in the selected Phase-4 model. Any subsequent CV17 evaluation on these panels constitutes adaptive development evidence; increasing the number-of-comparisons correction alone does not make that evaluation independent confirmation.”

3. **Claims correction**

> “The SUBAK.KO analysis weakens several proposed explanations but does not establish a unique cause of its errors. Equal median hypothesis/reference word counts do not establish correct alignment or exclude compensating insertion/deletion errors. Transfer from Common Voice prompts to SUBAK.KO or FLEURS is a hypothesis. Vaani contains spontaneous image-description speech. Zero detected CV17/FLEURS transcript matches does not certify absence of audio leakage.”

> “The proposed Phase-1 WER>0.60 AND CER>0.40 rule is an ASR-disagreement filter over human-written prompts. It may reject correct difficult recordings and retain defective pairs. Its label-quality discrimination and selection biases require empirical auditing; it is not described as difficulty-neutral.”

4. **Original-arm completion and union**

> “Complete the frozen MADASR and Vaani arms unless a registered abort occurs. Construct exactly one final candidate from O and all passing original buckets using the frozen union rules. A distinct union receives one 8,000-update run. It must pass the complete admission rule against O_ref and satisfy S(union)≤S(best passing single)+0.20 pp. If it fails, use the lowest-S passing single with at least 500 unique new hours, or STOP if none exists. No subset search or second union is permitted.”

5. **Admission reconnaissance and freeze integrity**

> “Source authorization does not waive G0.9(b), historical exclusions, or protected-set rules. Incoming defect denominators and audio-duplicate findings are preserved before ASR filtering and removal. A failed incoming-rate gate cannot be repaired by relabelling the surviving rows as a new bucket. CV17 release inventories and the actual available validated recordings must be reconciled before any later admission proposal.”

> “New evidence that invalidates a frozen corpus or panel suspends the affected comparisons for reconciliation. Frozen rows, targets or panel membership are not silently changed while completed-arm results are retained as comparable.”

6. **Continuation and spending**

> “The selected corpus resumes its own verified 8k state without changing H=150,000, the scheduler, selector, plateau policy or eligibility guards. At 20k, require S≤19.530, D≤19.35 and every main-run guard, including conversational WER≤22.075%. Failure terminates Phase 4 without a FLEURS decode or corpus-substitution restart.”

> “Plan total Phase-4 expenditure within 300 GPU-h, including approximately 45 GPU-h already spent. Reforecast from observed shared-card throughput before each remaining spending tranche. Only the owner may increase this ceiling. CV17 receives no further Phase-4 GPU allocation.”

**Recommended execution order**

1. Reconcile the reported free-memory state with the registered yield rule; preserve resumable state.
2. Recompute MaCro’s m=4 admission statistic from the authoritative predictions and bootstrap procedure.
3. Finish MADASR and Vaani; apply m=4 and all existing guards.
4. Select the prescribed union; run it once if distinct. Reuse MaCro’s 8k state if it is the selected corpus.
5. Reach the unchanged 20k gate at roughly **78–96 cumulative GPU-h**. On failure, terminate.
6. On pass, continue under the existing schedule and stopping rules within the **300 GPU-h** plan.
7. Register CV17 separately, with two evidence tiers, incoming-overlap audits before filtering expenditure, and an explicit distinction between adaptive development and independent confirmation.
