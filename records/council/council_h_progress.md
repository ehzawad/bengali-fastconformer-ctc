**Keep the registered science.** Finish both pilots, evaluate exactly one distinct union if required, and retain the existing admission, selection, horizon and stopping rules. The packet contains several factual errors and one important resume-boundary risk to check before continuation.

This is a review of the supplied evidence. The implementation files and checkpoints were not present in the review workspace, so I cannot independently certify the reported test execution.

1. **Q1 — The described fixes implement the contract; none requires a new scientific ruling.**

   | Fix | Verdict and required interpretation |
   |---|---|
   | D4: eligibility-aware retention | Correct. Preserving additional already-eligible checkpoints changes storage, not the candidate population or selector. |
   | D5: unchained tie band | Correct: find the minimum **unrounded eligible S**, then select the earliest checkpoint within **0.05 pp of that minimum**. |
   | D6: forced 20k validation | Correct. The gate cannot depend on an optional validation flag. This does **not** authorize abandoning the registered 2k validation cadence elsewhere. |
   | D7: recorded ABORT instead of `assert` | Correct. Horizon/batch incompatibility is a terminal protocol refusal, including under `python -O`; the supervisor must not classify it as an infrastructure failure. |
   | D11: refuse terminal STOP on fresh invocation | Correct and necessary. **Using** `P4_OVERRIDE_TERMINAL_STOP=1` requires the recorded owner ruling already specified. Merely implementing the disabled override does not. |

   D1–D3 restore the registered plateau and eligibility rules; D8–D10 restore the registered watchdog behavior.

   Three qualifications:

   - **“At most one extra entry in the tie band” is false** if every eligible checkpoint in that band is retained. There are **75 scheduled validations through 150k**; all could lie within 0.05 pp. At 1.39 GB each, that is approximately **104.3 GB** of validation checkpoints. Affordable against 9.5 TB free, but report the correct bound.
   - Plateau anchoring must use the **minimum eligible S**, independently of the earlier checkpoint selected by the tie rule. Those values can differ by 0.05 pp.
   - D1–D11 names **11 defect IDs**. Reconcile “ten,” and identify the **one skipped test**. Neither discrepancy invalidates the fixes, but neither should disappear inside “all tested.”

   Preserve the original freeze and append the implementation revision and tested hashes; do not overwrite the historical freeze to suggest the corrected code ran the earlier pilots.

2. **Q2 — Confirm the sequence, but distinguish the best single for comparison from the eligible fallback.**

   Finish MADASR and Vaani at **8,000**. Apply the complete admission test against **O_ref at 8,000**, with **m=4**, quantile **0.99375**, the marginal interval, point threshold and all admission guards.

   Then form exactly:

   > “ONE union U = O ∪ (all passing buckets)”

   If U is MaCro+MADASR, train its own **8,000-update prefix from init A**. Cost is approximately **9.5–11.1 GPU-h** at 846–723 updates/h. Do not splice two trained states together.

   Its point threshold is:

   \[
   S(U)\le \min\{S(O_{\rm ref})-0.30,\ \min_{\text{passing singles}}S+0.20\}.
   \]

   Thus **yes**, if MADASR finishes below MaCro’s **20.531**, MADASR supplies the best-single comparison—even though it cannot satisfy the long-run hours minimum. For example, MADASR at **20.200** makes the union threshold **20.400**, tighter than the control threshold of approximately **20.7735**.

   If U fails, §8k governs:

   > “the lowest-S passing single arm whose own frozen corpus has ≥500 unique new hours”

   | Candidate | New hours | Can be the fallback? |
   |---|---:|---|
   | MaCro | 726.5 | **Yes** |
   | MADASR | 149.0 | **No** |
   | Vaani | 117.4 | **No** |

   Therefore the fallback is **MaCro’s own 8k state**, even if MADASR has lower S. No second union or subset search.

   The live MADASR trajectory is **not yet a pass**. At 4k, against O_ref’s required 8k endpoint, its **D delta is +0.335 pp**, exceeding +0.30, and conversational delta is approximately **+0.770 pp**, exceeding +0.50. Its eventual result remains unknown.

3. **Q3 — In-domain-dominated admission is legitimate under this registered objective. The stated D improvement is wrong.**

   At matched **4k** updates:

   | Panel | MADASR − control, pp | Weighted contribution, pp |
   |---|---:|---:|
   | D | **+0.076** | +0.0456 |
   | M | +0.444 | +0.0888 |
   | R | **−9.547** | **−0.9547** |
   | K | +0.608 | +0.0608 |
   | **S** | | **−0.7595** |

   R contributes **125.7%** of the net improvement; deterioration elsewhere offsets part of it. D did **not** improve by 0.09 pp. The three worsening D sources are IndicVoices **+0.29**, Kathbath **+0.22**, and OpenSLR **+0.01**.

   Exact current card wording:

   > “At 4,000 updates, O+MADASR improved the registered composite by 0.7595 percentage points relative to the control at the same update. The improvement was concentrated on RESPIN dev: R improved by 9.547 points, while D worsened by 0.076, M by 0.444 and K by 0.608. These are interim development results; the registered admission decision occurs at 8,000 updates.”

   Wording accompanying any eventual admission:

   > “Admission establishes improvement under the registered weighted-development objective and regression tolerances. It does not establish broad improvement, superior labels, or transfer to FLEURS. Bootstrap intervals condition on the evaluated trained systems and do not quantify training-run or hardware variability; the registered 8,000-update cross-card agreement test was not completed.”

   **Both guard systems are required.** Admission uses **O_ref**, including every D source ≤ control +1.0 pp and conversation ≤ control +0.5 pp. Continuation eligibility uses **P1**, including conversation ≤ **22.075%**. Neither substitutes for the other. These tolerances permit some deterioration; they do not promise universal improvement.

4. **Q4 — Run Vaani to 8k. No salvage within this registration.**

   None of its registered abort conditions has fired. Early per-source guard failures do not authorize early termination.

   The remaining compute is approximately:

   \[
   (8000-2904)/846=\mathbf{6.02\ GPU\!-\!h},
   \]

   rather than five hours, before any overhead omitted from that throughput estimate.

   If it fails at 8k, exclude it whole. A capped mixture, conversational upweighting or partial admission requires a **new registration**. Its early conversational benefit is useful research evidence, not permission to rescue the arm.

5. **Q5 — The relative thresholds govern. The 0.001 pp check must never become gate slack.**

   Keep:

   \[
   S\le21.0296-1.5=\mathbf{19.5296},\qquad D\le\mathbf{19.35}.
   \]

   A **0.001 pp consistency tolerance** against the printed constants is reasonable: 19.530 differs from 19.5296 by **0.0004 pp**.

   But **S=19.5300 fails**, even with D and guards passing. Do not round the observed S or add 0.001 to its permissible threshold.

   The consistency check must reject non-finite values and survive `python -O`. It supplements the frozen baseline’s provenance and identity checks; it does not replace them.

6. **Q6 — Keep the instrument limitation, with narrower wording. No CV17 experiment needs to be recreated.**

   Cryptographic hashing operates on the input representation; its purpose is to distinguish changed messages. The recording-identity limitation follows from that design, rather than requiring the destroyed reconnaissance as evidence. [NIST Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)

   Proposed exact wording:

   > “PCM SHA-256 detects equality of the canonical decoded sample representation. It does not establish recording independence: the same underlying recording can produce unequal PCM representations after decoding, re-encoding, resampling or other transformations. This is an instrument limitation, not an empirical finding about Common Voice 17. Future overlap audits must combine source-identity checks and calibrated audio fingerprinting with exact PCM hashing. No cross-decoder miss rate is claimed.”

   Avoid saying hashing misses **every** cross-decoder copy: different decode paths can also produce identical canonical PCM. Likewise, the claim that the MaCro matches were found **only because** they shared a decoder path needs surviving evidence of that mechanism.

   The withdrawal is not consistently reflected in the supplied prose. Remove the pre-launch card’s withdrawn release-17 hour estimate and ensure the old identity-join and overlap paragraphs in §8l cannot be cited as current findings.

   Also correct §8n(iii): Vaani validation/test **are protected one-shot evaluation sets**. Their absence from the fingerprint index remains the disclosed coverage limitation; calling them “neither … a panel” does not eliminate it.

7. **Q7 — Keep G0.9(b), but reject “counts grow, therefore strict containment is proved.”**

   Mozilla documents release-time exclusion of problematic clips and train/dev/test partitions that can change across releases. Consequently, increasing aggregate counts do not establish a clip-level subset relation. [Mozilla’s scripted-speech documentation](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md)

   The supplied registry shows **net growth of 618 validated clips and 0.81 hours**. That is not an identifier-level proof that precisely 618 recordings are newly available.

   The consequences are:

   - **G0.9(b): unchanged.** Its literal requirement is **<0.5%**, using incoming candidates and including cross-source copies. Re-release status does not exempt duplicates, and removing them does not retrospectively improve the incoming rate.
   - **MaCro finding: unchanged.** The **1,047 exact copies**, approximately **0.190%** of 550,617 ingested candidates, remain positive evidence backed by the surviving audit. They are not proof that every non-identical recording is independent.
   - **Future re-releases:** screen publisher, collection lineage, release, canonical recording IDs and historical exclusions **before audio ingestion**. Register a precisely defined novel delta prospectively; report the full inventory and already-held share so this is not denominator laundering. Then apply protected-text, identity, hash and calibrated fingerprint checks.

   The identity assertion to verify on the **surviving CV26 files** is:

   \[
   \text{ingested IDs}=\text{validated IDs}\setminus(\text{dev IDs}\cup\text{test IDs}).
   \]

   Matching the arithmetic **44,739−9,402−9,402=25,935** alone is insufficient. This check does not require reacquiring CV17. Keep CV17 withdrawn and make no forecast of how many useful rows a future release will contain.

8. **Q8 — Change no hyperparameter. Correct the epoch arithmetic and card language.**

   At **37.8 million presentations**:

   | Corpus | Nominal epochs at H |
   |---|---:|
   | MaCro | **34.848** |
   | MaCro+MADASR | **32.036**, not 30.6 |
   | All three | **29.861** |

   These exposures create an overfitting risk, but the risk was accepted within the registered horizon, plateau and eligible-checkpoint selection design. Phase 1’s 18 epochs do not justify changing the schedule after observing these pilots.

   **Keep H, LR, warmup, regularization, batch and sampling unchanged.** Resume the selected arm’s weights **and buffers**, optimizer, scheduler position, sampler and RNG. A continuation LR reset would break the literal-prefix design.

   Correct the card’s “2,000 hours target is met” statement: the maximum is **1,958.7 h**, which is **41.3 h short of 2,000**. What MaCro satisfies is the registered **≥500 new-hour** minimum.

9. **Q9 — Both projected paths fit, subject to measured throughput and complete accounting.**

   Using the supplied end-of-pilots estimate and allowing eight hours for final evaluation/export:

   | Path | Estimated total | Margin below 300 |
   |---|---:|---:|
   | Reuse MaCro | **248.3 GPU-h** | **51.7** |
   | Distinct union first | **258.2 GPU-h** | **41.8** |

   At **723 updates/h** for both the union pilot and continuation, the second path becomes approximately **280.5 GPU-h**. Thus 250–260 is a reasonable central estimate, not a guaranteed completion cost. GPU-hours across simultaneous cards must be added.

   Insert one CPU preflight, targeting **under 30 minutes, zero GPU-h**:

   - Verify current code hashes against the passing test record; identify the skipped test. Freshly launch the continuation process so it actually loads the corrected code.
   - Reissue admission records from the existing frozen predictions with the hardened tool. Require **m=4**, **quantile=0.99375**, exactly the 8k validation record, and explicit P1 continuation eligibility. The supplied `null` fields do not document those checks.
   - Verify the selected corpus/checkpoint association and full resume state. Check prediction/checkpoint/panel identity; aggregate WER agreement is a strong consistency check, not a complete ordering proof.
   - Run the crash-boundary test in Q11.

   No additional cross-card training experiment is needed merely to repeat already-completed qualification. Record the accepted hardware deviation.

10. **Q10 — Pre-register the IndicVoices inventory audit; prepare SPRING as the first next-phase candidate.**

   For **IndicVoices**, register a CPU-only identity and exclusion-ledger diff. The outcome is **measured unique admissible delta hours**, currently unknown. A delta of **zero** closes that candidate. A positive delta justifies preparing an arm; it does not itself admit data or promise a useful effect. There is no evidential basis here for inventing a 50-hour or 100-hour cutoff.

   For **SPRING**, avoid choosing a whole-bucket Latin-loss ceiling just above the observed 46–48%. The defensible next-registration amendment is a semantically defined subset:

   > “For the next registration only, SPRING-INX R1/R2 may form a Bengali-script subset. After the frozen registered surface transformation, retain a row only if it contains zero disallowed characters under the existing script predicate. Reject the entire offending row; do not delete spoken words or transliterate. The whole-source script-loss ceiling is waived for this explicitly defined subset. Report exclusions against the full source inventory. This is a content filter that removes code-switched speech. All other admission requirements remain applicable.”

   **Exact retained-row threshold: zero disallowed characters.** The waiver requires owner authorization. No adaptive retained-fraction threshold.

   Its design is already informed by observed script rates; it cannot honestly be described as chosen before any retained-rate information was known. Register it before further screening and model outcomes, with that disclosure. **≈280 h remains an estimate**, since character share is not duration-weighted retained audio.

   Keep **Ben-10** in the research queue. Neither its usable hours nor its anticipated machine-gate failure is sufficiently established here to prioritize a training registration over those two checks. None enters Phase 4.

11. **Q11 — My highest-priority remaining failure hypothesis is a resume that silently skips the 20k gate.**

   Mechanism: update **20,000** is checkpointed, then the process dies before the validation or gate decision is durably recorded. Resume starts at **20,001**. An equality-triggered gate never fires again. The listed tests explicitly verify that the gate does not fire at 20,001, but do not establish recovery from this intervening crash.

   This is **not a demonstrated implementation bug** from the packet; it is the most consequential uncovered boundary I would test first.

   The revealing invariant is:

   > Before executing update 20,001 or later, exactly one valid PASS decision must exist for this run’s update-20,000 checkpoint, frozen baseline and evaluator.

   Cheapest check: a **CPU fault-injection test** that interrupts execution between the 20k checkpoint write and gate-decision write, then exercises the real resume path. The resumed run must evaluate the saved 20k state or refuse continuation. It must not use a later checkpoint retrospectively.

   Include **S=19.5300, D=19.35, guards passing**: the expected result is **terminal failure**, because S exceeds 19.5296. Verify a fresh supervisor invocation remains stopped.

   Finally, preserve the current terminal-outcome wording: §8k explicitly includes **plateau STOP** among outcomes after which no FLEURS decode follows. If the intended behavior is to score the selected checkpoint after a plateau stop, that requires an **owner amendment**; it is not an implementation fix.
