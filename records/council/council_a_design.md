**Verdict: NO-GO for GPU work today; GO for the design below once provenance, publication rights, and leakage checks pass.** Keep the Phase-1 Bengali weights and tokenizer. Exclude Shrutilipi. Default to the cleaned natural mixture. Register **150,000 additional optimizer steps maximum**, with **176 GPU-h allocated and 24 GPU-h reserved**.

This is a review of your supplied measurements and public documentation, not an inspection of the downloaded audio or manifests. No GPU was used.

**Q1 — CORPUS**

**Source decisions**

| Source | Recommendation | Admission rule |
|---|---|---|
| MaCro filtered train, 736.8 h | **IN** | Verify original human prompt/reference lineage; structural cleaning, protected-set carving, global deduplication, human quality gate below |
| MaCro validated, 48.9 h | **IN** | Target **3.5 h dev**, accepting 3–4 h; approximately **45.4 h train before other removals** |
| MADASR train, 603.3 h | **OUT until exact release terms are established** | Admit only if training and the intended model distribution are permitted **before manifest freeze**, and quality gate passes |
| MADASR official dev, 1.9 h | Same licence block | Evaluation rights also need establishing |
| Shrutilipi wer15, 142.6 h | **OUT; zero pilot budget** | Filtering does not establish human verification of segment/transcript alignment |
| OpenSLR release | **Merge, never concatenate** | Preserve approved originals; reconsider only genuinely additional recordings with recoverable provenance and exclusion history |
| Kathbath train | **Merge, never concatenate** | Same rule; official evaluation splits remain excluded |
| Existing IndicVoices, SUBAK.KO, Common Voice, FLEURS train | **KEEP** | Preserve Phase-1 cleaning, subject to global leakage/deduplication checks |
| Pseudo-labels, TTS, “long” concatenations | **OUT** | Explicit source allowlist; no recursive directory ingestion |

**(a) MaCro satisfies the owner’s definition, subject to lineage verification.** Human-authored prompts spoken by humans are human labels, even where a person did not subsequently verify every recording. Call them **human-labelled, model-filtered prompted speech**; do not call all 736.8 h manually verified.

Require evidence that the retained text is the supplied human reference after documented normalization—not a Whisper replacement. The filtering model’s use of TTS during its own training does not make the retained recordings synthetic.

The filter imports these biases:

- Preference for acoustically easy, canonical, familiar speech and orthography.
- Underrepresentation of dialect, spontaneous deviations, noise, disfluency, names, and code-switching.
- Complete removal of the specified Bengali-digit rows.
- Repeated selection toward the teacher’s strengths across three rounds.
- Duration/length bias that **must be measured, not assumed to mean “shorter.”** At strict WER `<15%`, a six-word reference permits **zero** errors, whereas a seven-word reference permits one. Very short references can face harsher selection.

Compare retained versus original distributions where the original index exists: duration quartiles, reference word count, dialect/speaker metadata, numerals, and vocabulary. Without the rejected population, disclose that selection bias cannot be quantified fully.

**(b) Shrutilipi stays out.** The unfiltered ablation does not prove the filtered subset would fail. It establishes a high burden of proof, and WER filtering alone does not meet it. Its **142.6 h is only about 6%** of the larger candidate pool. Spending an ablation slot on it is worse than testing whether the clearly eligible additions help.

**(c) Use Phase 1 for disagreement triage, humans for noise estimation. Do not re-filter everything with Phase 2.**

The arithmetic is decisive:

- Qwen: `1.4M × 1.461 / 3,600 = 568 GPU-h`.
- Phase-1 CTC: `1.42M / 50 / 3,600 = 7.9 GPU-h` before overhead.
- A **22,000-clip** CTC audit: **0.122 GPU-h** of nominal decoding.

Register this protocol:

1. Audit five possible new buckets separately: MaCro filtered, MaCro validated train-side remainder, licensed MADASR, additional OpenSLR, additional Kathbath. Keep six original-source buckets as controls.
2. Decode **2,000 clips per bucket**, or the whole bucket if smaller: **1,600 simple-random clips**, plus **400 duration-stress clips**, 100 from each interval `[0.5,3), [3,6), [6,12), [12,23]` seconds. Reallocate unfilled quotas by a fixed seeded rule.
3. Use frozen Phase-1 greedy decoding, no augmentation. Flag **WER >30%, CER >15%, or an empty hypothesis**. These are review flags, not automatic rejection thresholds.
4. For each new bucket, have Bengali annotators independently transcribe **400 randomly selected clips** from the random sample, without seeing ASR hypotheses. Separately review **100 flagged/stress clips**. Review **100 random clips per original-source control**.
5. Define a material label error as **>10% reference WER against adjudicated human transcription**, under the frozen scorer. Define catastrophic errors as wrong recording/reference pairing, non-speech with a speech transcript, or gross segmentation mismatch.
6. **New-source pass:** at most **12 material errors / 400 random clips**, **zero catastrophic errors**, and at most **20 material errors / 100 targeted clips**, with zero catastrophic errors there too. Twelve of 400 gives a **one-sided 95% Wilson upper bound of 4.75%**. Zero catastrophic errors in 400 gives an exact upper bound of approximately **0.75%**.
7. Second-review every proposed error and a random **20% of proposed clean clips**. Unresolved cases count as errors.
8. A failed new bucket is **excluded whole under this registered rule**. No threshold sweep, automatic transcript repair, or repeated sampling until it passes.

Report random-sample estimates separately from targeted-review yields. Report duration-stratified disagreement and hour-weighted diagnostics; the 400-clip acceptance bound concerns **clip prevalence**, not corrupted hours.

Maximum human workload is about **3,100 clips**, roughly **5 h of audio** and an estimated **20–40 annotator-hours**. That CPU/human work is essential. Without it, call the result an **ASR-disagreement audit**, not a label-noise estimate.

Allocate **≤6 GPU-h total** for audit decoding, baseline selection-panel inference, and resource profiling. Do not spend the allowance merely because it exists. Phase 1 has trained on the old corpus; its lower disagreement there cannot establish superior old-label quality.

**(d) Deduplication and carving**

Apply in this order:

1. Restore canonical source recording IDs and original human references where available.
2. Deduplicate canonical paths and identical decoded audio.
3. Detect re-encoded, trimmed, gain-adjusted, and overlapping-segment duplicates.
4. Remove every training recording belonging to protected text/audio groups.
5. Apply **global `MAX_PER_TEXT=5` across all sources**, retaining distinct recordings and preferably distinct known speakers.

**Keep the cap at five.** More corpus volume is not a reason to increase repeated-prompt exposure. Use deterministic selection favouring verified references and speaker/source diversity; never rank candidates by the lowest ASR WER.

Different people reading the same sentence are useful acoustic examples, not duplicate audio. Retain up to five globally. A protected sentence, however, excludes **all** corresponding training recordings.

Your OpenSLR counts need reconciliation:

- `166,585 − 8,603 = 157,982` shared recordings.
- `166,585 + 49,398 = 215,983` union recordings before exclusions.

Thus “we already train on 166,585 of these” cannot literally mean that all 166,585 occur in his release. Recover the ID-level ledger. Historical exclusions for evaluation membership, speaker separation, corruption, or bad labels remain exclusions.

**(e) Digits and punctuation**

Use the frozen normalization consistently: **punctuation → space, NFC, whitespace collapse**. Preserve existing human-labelled Bengali and ASCII numerals. Do not delete digit-bearing rows or invent spoken-number expansions.

For shared recordings, retain the original approved human reference. Punctuation deletion can join words, so his cleaned text is not necessarily equivalent to punctuation-to-space normalization. Recover originals for new recordings; quarantine unresolved lexical changes. Do not change the headline scorer to accommodate training text.

**(f) Duration and CPS**

For **new recordings**, register:

- Duration **0.5–23.0 s**.
- **2–35 CPS**, defined as NFC non-whitespace Unicode code points—including combining marks—per audio second.
- Reject corrupt, non-finite, all-zero audio and empty normalized references.
- Verify every file’s actual decoded sample rate, channels, and duration. Convert legitimate format differences deterministically to 16 kHz mono.
- Apply the tokenizer/CTC feasibility checks in Q3.
- No blanket SNR rejection, aggressive VAD cropping, or new speed augmentation.

These are conservative engineering limits, not assertions that speech outside them is mislabeled. Publish removal counts and hours per reason. Preserve existing Phase-1 bounds for old recordings.

**Hours:** the listed components, excluding Shrutilipi and allowing 3.5 h for MaCro dev, total approximately **2,428.8 h with MADASR** or **1,825.5 h without it**, before carving/caps and before accounting for retained old-only OpenSLR/Kathbath recordings. Neither is an audited final total. **Do not weaken cleaning to advertise “2,000 h.”**

**Q2 — MIXTURE**

**Use the natural mixture after all cleaning and the global text cap.** No FLEURS upweighting, no automatic IndicVoices cap, no quality weights derived from teacher WER.

Excluding Shrutilipi already changes the proposed proportions substantially. Without MADASR, the nominal MaCro contribution, including validated training audio, is approximately **43%**, and IndicVoices approximately **32%**. Recompute from the final manifest.

“Natural” means each retained recording appears once per audited epoch. Hour shares are descriptive; they are not automatically minibatch probabilities or loss weights.

Register exactly **one check that can overturn natural sampling**:

- Build a second static manifest limiting **MaCro-filtered + MADASR-filtered together to 30% of retained hours**.
- Keep all other eligible recordings, including MaCro validated, unchanged.
- Downsample the two filtered sources proportionally with a fixed hash selection.
- After the matched 6,000-step pilots, choose this capped mixture only if it passes the eligibility gates in Q5 and beats natural by **≥0.5 percentage points in selection score**, with the paired **95% CI upper bound below zero**.

Otherwise keep natural. No alternate caps.

**Q3 — INITIALISATION & TOKENIZER**

**Choose A. Expected final WER at a fixed compute budget: A < B < C.** This ranking is an engineering judgment, not a measured ablation.

| Initialization | Warmup | Peak LR | Recommendation |
|---|---:|---:|---|
| **A: all Phase-1 weights, existing tokenizer/head** | **1,000 steps** | **2e-4 throughout** | Registered choice |
| B: Phase-1 encoder, fresh tokenizer/head | 2,000 steps | Encoder **1e-4**, head **5e-4** | Reject unless the existing tokenizer fails CPU checks; would require a new registration |
| C: English initialization, fresh Bengali head | 2,000 steps | **5e-4** | Reject for this Phase 4 |

At 1,100 steps/h, **10,000 steps cost 9.1 h** under any of these initializations, absent token-length differences. A saves optimization progress, not inherently forward-pass time. C discards the adaptation that already cost **36.4 GPU-h**. B discards a trained label mapping without evidence that it is inadequate.

A has no persuasive plasticity problem here: the language and architecture remain unchanged, and all layers remain trainable. The plausible risk is forgetting from a distribution shift plus excessive LR. Address that through the lower peak LR, retained old data, and source-specific regression gates—not a fresh head.

Load A’s weights but start **new AdamW state and a new scheduler**. A weight initialization is not a resume from Phase 1’s old cosine position.

Before accepting its tokenizer, demand:

- Exact tokenizer file and token-ID mapping hashes.
- **Zero UNKs in retained labels** and lossless encode/decode round trips modulo documented tokenizer normalization.
- Quarantining unsupported legitimate text must affect **≤0.1% of clips in every source**; otherwise **STOP A**.
- Report tokens/s p50, p95, p99 and rejection rates by source.
- Require **tokens/s ≤11**, plus the stronger alignment-margin check below.
- If CTC/token-rate rules reject **>1% of legitimate clips in any source**, stop rather than quietly remove the dialect or text type.

For each transcript let \(L\) be token count, \(R\) the number of adjacent equal-token pairs, and \(T\) the **actual encoder output length**. Require:

\[
L+R\le T,\qquad (L+R)/T\le0.90.
\]

The first is CTC alignment feasibility; the second is the registered safety margin. At roughly 12.5 frames/s, an 11-token/s limit alone does not account for repeated-token blank requirements. Do not let `zero_infinity` conceal impossible examples; PyTorch documents that it zeros infinite losses and associated gradients. [PyTorch CTCLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html)

**Q4 — HORIZON & BUDGET**

Register **150,000 additional optimizer steps**, including the selected 6,000-step pilot.

- Effective batch **252**.
- At 1.4M recordings: **5,555.6 steps/epoch**.
- Maximum selected-run exposure: **37.8M recording presentations**, or **27 epochs** at that corpus size.
- Recalculate epochs from the final retained count. A smaller capped corpus will see more epochs.
- Warmup **1,000**, peak LR **2e-4**, cosine minimum **1e-5**, endpoint **150,000**.
- AdamW weight decay **1e-3**, clipping **1.0**, bf16 and the Phase-1 SpecAug settings.
- Validate every **2,000 optimizer steps**, including pilot endpoints.
- Resume the selected **step-6,000 endpoint** with its optimizer, scheduler, sampler and RNG states. No second warmup.

**Progress gate at step 20,000:** selection score improves by **≥1.5 pp** versus frozen Phase 1; locked-dev WER improves by **≥1.0 pp**; source regression guards still pass. Failure means stop.

**Plateau rule:** starting at step 30,000, stop after **five consecutive validations—10,000 steps—without a ≥0.2 pp improvement** in the best eligible selection score. Keep an explicit “meaningful-improvement anchor”; tiny improvements cannot reset patience indefinitely.

**Retention:** immutable Phase-1 parent; selected pilot endpoint; last two resumable checkpoints; best three eligible checkpoints by selection score; immutable step-20k gate checkpoint. Store predictions and metrics for every validation. Final choice is the lowest eligible score, breaking ties within **0.05 pp** toward the earlier checkpoint. No unregistered averaging or decoding search.

The supplied **19-point** internal-dev sequence does not establish a per-1k slope through 40k. Recover actual global steps and score Phase-1 `final.nemo` with the external scorer. Do not extrapolate “−0.3 pp/1k” into a promised endpoint.

**Shared-box launch gate:** profile worst-duration batches and optimizer-state allocation. Target **≤16.5 GiB process VRAM**, stop the process at a **17.0 GiB watchdog threshold**, and never exceed the **17.5 GiB practical line**. Select among predeclared `12×21`, `6×42`, or `4×63` batching configurations before pilots.

A VRAM ceiling does **not** prevent compute or I/O interference. Use an established resource reservation that protects the co-tenant. If both workloads need the same GPU concurrently and no isolation arrangement exists, **do not launch**. Never reset the GPU, alter shared clock/power settings, or kill another process.

**Q5 — PRE-REGISTERED GATES**

Spend the pilot budget on **new-data value and mixture risk**, not three initialization experiments.

Define externally scored WERs:

- \(D\): unchanged locked dev.
- \(M\): new MaCro validated dev.
- \(R\): licensed MADASR official dev.
- \(K\): Kathbath official validation.

Use:

\[
S=0.60D+0.20M+0.10R+0.10K
\]

when MADASR evaluation is licensed. Otherwise fix:

\[
S=0.60D+0.30M+0.10K.
\]

These are weighted panel-level corpus WERs, expressed in percentage points. Freeze the applicable formula before pilots. Keep \(D\) separately visible for historical comparison.

| Arm | Initialization | Static corpus | Steps |
|---|---|---|---:|
| O: continued-training control | A | Original eligible Phase-1 corpus | **6,000** |
| N: natural expansion | A | All newly admitted data plus old corpus | **6,000** |
| C: capped expansion | A | Same, with the registered 30% cap | **6,000** |

All arms use the same LR schedule and effective batch. Each processes **1.512M recording presentations**. This matches optimizer updates and recording exposure, **not necessarily audio hours**; report actual audio hours and GPU-h.

**Before pilots:**

- All provenance, publication-rights, structural, leakage and tokenizer gates pass.
- MaCro filtered passes the human quality gate.
- Final natural manifest adds **≥500 nonduplicate hours** beyond eligible Phase-1 recordings.
- Resource profile passes.
- Audit/profile allocation stays within **6 GPU-h**.

**During pilots:** stop an arm immediately for non-finite loss/gradients or a resource violation. At validation from step 2,000 onward, stop it for **>1% empty outputs** or locked-dev regression **>5 pp** versus Phase 1.

**At exactly step 6,000**, a new-data arm is eligible only if all hold:

1. \(S\le S_0-0.5\) pp.
2. \(M\le M_0-1.0\) pp.
3. \(D\le D_0+0.3\) pp.
4. No original dev source worsens by **>1.0 pp**.
5. The preidentified conversational portion of locked dev worsens by **≤0.5 pp**.
6. Empty-output rate is **≤0.1%**.
7. It beats control O by **≥0.3 pp in \(S\)**, with paired sentence-clustered **95% CI upper bound below zero**.

Then apply Q2’s **single cap-overturn rule**. If natural is ineligible, capped must still meet that rule against natural and all its own eligibility criteria. If neither qualifies, **STOP Phase 4**. Do not promote O into a long Phase-4 run or open FLEURS to seek encouragement.

Bootstrap comparisons use **10,000 fixed-seed draws**, preserving sentence clusters, including cross-panel cluster membership where present. These are operational selection gates, not final confirmatory claims.

Pilot training is **18,000 / 1,000 = 18 GPU-h** at the conservative admitted throughput. Including the **6 h** preflight allocation gives **24 GPU-h**, below the 25 h limit. No B, C-initialization, Shrutilipi, or additional dataset ablations.

**Q6 — EVALUATION SURFACE**

| Panel | Role |
|---|---|
| Locked 5,385-row dev | Selection, stopping, historical comparison |
| MaCro validated 3–4 h slice | Selection; transcript/audio disjoint, **not claimed speaker-disjoint** |
| MADASR official dev, 1.9 h | Selection only with established rights and historical-disjointness audit |
| Kathbath official validation, 5.0 h | Selection |
| Kathbath official test, 3.2 h | One-shot report |
| Kathbath test_known, 5.0 h | Optional registered one-shot report; identify its known-speaker nature |
| Phase-1 speaker-disjoint holdout | One-shot report |
| Existing separate conversational holdout | One-shot report |
| FLEURS official test | Sealed Phase-4 headline |
| FLEURS validation | **Never a selection or reporting instrument** |

Do **not** combine Kathbath validation and test into an 8.2 h selection panel.

For MaCro, target **3.5 h and ≥500 distinct sentence groups**, selecting at most three evaluation recordings per group. Quarantine every other recording with those sentences from training. Select groups by a fixed seed after eligibility checks, not by baseline WER.

**Historical exposure matters.** Audit new panels against **every Phase-1/2/3 training manifest and tokenizer-training text**, not only Phase 4. Removing a sentence now cannot undo exposure inherited in A. Choose MaCro dev from historically unseen groups. An official split with historical exposure must be disclosed as such and cannot silently be treated as an unseen selection instrument.

Run these checks before training:

- Exact canonical IDs and normalized sentence strings.
- Secondary audit keys covering punctuation deletion, Unicode formatting differences and digit-script equivalence. These keys do not replace the scorer.
- Conservative fuzzy-text candidate retrieval, e.g. normalized character edit similarity **≥0.95 for strings ≥20 characters**; adjudicate or quarantine candidates.
- Hash decoded PCM, not only WAV files.
- Robust audio fingerprint retrieval for re-encoding, trimming, gain changes and shared segments; verify candidates acoustically.
- Exact speaker-ID exclusions wherever reliable IDs exist. The official MADASR description explicitly specifies unseen dev/test speakers and sentences; recover and verify that metadata. [MADASR official dataset description](https://sites.google.com/view/respinasrchallenge2023/dataset)

**Pass rule: zero retained train/protected-set text matches, zero confirmed audio overlaps, zero unresolved overlap candidates, and zero forbidden known-speaker intersections.** Validate the fingerprint pipeline on **200 controlled transformed duplicate pairs**, requiring **≥99% recovery** across the declared transformation suite. This validates detection performance; it does not prove that every possible hidden overlap is detectable.

Run sealed-set checks through a fixed audit process that exposes match decisions, not examples or model performance. Preserve the official 920-recording test. Do not “repair” contamination by deleting test rows.

Known FLEURS-validation overlap with FLEURS train remains documented historical contamination. Its absence from new files does not rehabilitate that validation set.

**Final headline procedure**

Freeze one checkpoint, tokenizer, decoding configuration and scorer **before** report-only evaluation. Then:

- Decode all **920 FLEURS recordings** once.
- Bootstrap **349 sentence clusters**, sampling clusters with replacement and retaining all recordings in each selected cluster.
- Use **50,000 fixed-seed draws**.
- In each draw, recompute corpus WER as total errors divided by total reference words—not an average of sentence WERs.
- Pair the same draws with frozen Phase-2 and Phase-1 hypotheses.

Primary difference:

\[
\Delta_2=\mathrm{WER}_{P4}-\mathrm{WER}_{P2}.
\]

My registered success criterion is **WER ≤16.04%**, at least **0.5 pp** below Phase 2, and a paired two-sided **95% CI for \(\Delta_2\) entirely below zero**. This is stricter than a rounded point estimate just below 16.54%.

Compare against Phase 1 second in a fixed testing hierarchy; if the primary test fails, report the Phase-1 comparison descriptively. Report all endpoints regardless of outcome. No checkpoint switching after seeing any one-shot results.

Call this **a Phase-4 test lock**. The benchmark has already influenced programme-level decisions through earlier results; it is not an untouched test for the entire programme.

**Q7 — CLAIMS & LICENCES**

**MADASR is blocked now.** The official 2023 site establishes its data collection and split design, but the material I inspected did not establish the licence for your particular release. Its official repository links the audio and metadata without resolving that question. Public availability and a Kaggle mirror are insufficient evidence of permission. Archive applicable terms covering training, evaluation and the intended weight distribution before admission. [Official MADASR repository](https://github.com/bloodraven66/RESPIN_ASRU_Challenge_2023/)

**MaCro:** the competition rules allow commercial and noncommercial data use, specify CC-BY-4.0, and restrict redistribution to nonparticipants. Establish authorized access and retain the applicable rules. Publish pipeline code and aggregate provenance; do not bundle competition audio or reconstructible transcript datasets into either repository. A private repository still distributes material to its members. [Competition rules](https://www.kaggle.com/competitions/bengaliai-speech/rules)

**The NC requirement needs a rights decision before training.**

OpenSLR-53 is explicitly CC-BY-SA-4.0. That fact alone does not establish that learned weights necessarily inherit ShareAlike. However, your **Phase-1 weights themselves were released under CC-BY-SA-4.0**. If Phase 4 relies on that licence for adapting those weights, an NC restriction is incompatible with its commercial-use freedoms and prohibition on additional restrictions. [OpenSLR-53](https://www.openslr.org/53/), [CC-BY-SA terms](https://creativecommons.org/licenses/by-sa/4.0/)

The owner may be able to license rights they control separately, but must establish that they possess the necessary rights, including any inherited obligations. Existing CC grants cannot simply be revoked. Document the rights basis for the required NC release; obtain qualified review where that basis is unresolved. **An HF access gate is not a licence cure.** [Creative Commons licensing FAQ](https://creativecommons.org/faq/)

The NVIDIA starting model identifies CC-BY-4.0; preserve its attribution and provenance separately. [NVIDIA model card](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large/blame/main/README.md)

Defensible card wording:

> Continued fine-tuning of a Bengali-adapted FastConformer-CTC Large on X unique hours of human-labelled Bengali speech after documented cleaning, deduplication and evaluation exclusion. Y hours were selected upstream using ASR/reference agreement filtering. Phase-4 optimization used no pseudo-label targets or TTS recordings.

Also disclose:

- Exact retained hours, recordings and unique transcripts by source.
- Phase-1 training history separately; inherited exposure is not additional unique data.
- Upstream Whisper filtering, reported threshold/rounds, digit exclusion, normalization, and the filter model’s reported OpenSLR+TTS training.
- Which provenance details were verified and which remain upstream reports.
- Human audit protocol/results and every exclusion.
- MaCro speaker-disjointness limitation.
- Benchmark exposure history, selection panels, scorer, decoding, confidence procedure and all GPU-h.
- Base-model pretraining is outside the claim that **Phase-4 supervised targets** are human-labelled.

Do not claim fully manually verified data, uniformly dialect-representative data, globally speaker-disjoint evaluation, commercial usability, or “state of the art.”

The 14.37% hishab result is a useful external comparison, but its training exposure differs. A private pipeline also does not support calling the programme fully open-source or independently reproducible without access.

**Q8 — MOST LIKELY FAILURE**

**The most likely performance failure is that extra easy, repeatedly filtered read prompts produce attractive development gains without improving unseen FLEURS sufficiently.** Continuing training on the old corpus might have delivered comparable gains.

The one preventive spending check is the **6,000-step expanded-data versus continued-old-data control**, scored on historically unseen, human-verified MaCro plus the protected legacy surface. Require **≥0.3 pp composite advantage with a paired CI below zero**, alongside the conversational regression guard.

This prevents an unjustified long run. It cannot guarantee transfer to the sealed headline; the one-shot paired test decides that.

**REGISTERED PLAN — compact execution sheet**

| Item | Registered decision |
|---|---|
| Corpus | Existing eligible human-labelled corpus; audited MaCro filtered; MaCro validated minus 3–4 h dev; eligible OpenSLR/Kathbath additions. MADASR only with documented rights before freeze. Shrutilipi, pseudo-labels, TTS and concatenations excluded. |
| Corpus quality | New-source random audit: **≤12/400 material errors, zero catastrophic errors**; targeted audit **≤20/100**, zero catastrophic errors. Failed source excluded. MaCro filtered must pass. |
| Cleaning | New duration **0.5–23 s**, CPS **2–35**, frozen text normalization, preserve numerals, exact and robust audio deduplication, global **five recordings/text**. |
| Admission | Zero unresolved protected-set leakage; inherited-training audit complete; **≥500 additional unique hours**; publication-rights basis established. |
| Mixture | Natural static manifest. One alternative: filtered MaCro+MADASR capped jointly at **30% of hours**. |
| Initialization | Phase-1 all weights and unchanged 1,024-piece tokenizer; fresh optimizer/scheduler. Zero retained UNKs; tokens/s **≤11**; \((L+R)/T≤0.90\). |
| Training | Batch **252**, bf16, Phase-1 AdamW/SpecAug settings. Warmup **1k**, peak **2e-4**, cosine to **1e-5** at **150k additional steps**. |
| Pilots | Old continuation, natural expansion, capped expansion: **6k steps each**. All use initialization A. |
| Pilot gate | Versus Phase 1: **S −0.5 pp**, MaCro **−1 pp**, locked dev no worse than **+0.3 pp**; per-source **+1 pp**, conversational **+0.5 pp** maximum; empties **≤0.1%**. New arm beats old control **≥0.3 pp**, paired CI below zero. |
| Mixture choice | Cap wins only with eligibility plus **≥0.5 pp** advantage over natural and paired CI below zero. Otherwise eligible natural wins; otherwise stop. |
| Main gate | At **20k**, **S −1.5 pp**, locked dev **−1 pp**, regression guards pass. |
| Stop/select | Validate every **2k**. From **30k**, stop after **10k steps** without **0.2 pp** meaningful improvement. Select best eligible checkpoint; no averaging/search. |
| Evaluation | Selection: locked dev, MaCro dev, licensed clean MADASR dev, Kathbath valid. All tests/legacy holdouts report once. FLEURS validation unused. |
| Headline | One frozen model; **920 recordings / 349 clusters / 50k paired bootstrap draws**. Success: **≤16.04%**, paired 95% difference CI versus Phase 2 below zero. |
| Resources | Reserved shared-box access; target **≤16.5 GiB**, watchdog **17.0 GiB**, hard practical line **17.5 GiB**. No interference with co-tenant. |

| GPU budget | Ceiling |
|---|---:|
| Audit inference, baseline selection scoring, resource profiling | **6 h** |
| Three 6k-step pilots, including validation at ≥1,000 steps/h | **18 h** |
| Selected trajectory: remaining 144k steps, including validation/checkpointing | **144 h** |
| Final inference, export verification and latency measurements | **8 h** |
| **Allocated work** | **176 h** |
| **Operational reserve** | **24 h** |
| **Absolute Phase-4 cap** | **200 h** |

At the historical 1,100 steps/h, total pilot-plus-main training is `162,000 / 1,100 = 147.3 h`; adding the audit and final-evaluation allowances gives **161.3 h**. The budget above uses **1,000 steps/h** conservatively.

Count all occupied GPU wall time, including failed jobs and operational replay. Reserve covers recovery within the registered trajectory—not new experiments, additional unique steps, or extensions after failed gates.

**Explicit STOP conditions**

- **Before any GPU:** unresolved rights for the required NC release; unauthorized data access; incomplete provenance or leakage audit; no co-tenant-safe resource reservation.
- **Source exclusion:** unresolved MADASR licence; failed source-quality gate; forbidden label/audio origin. No threshold relaxation.
- **Stop this Phase-4 registration:** MaCro quality failure, fewer than 500 eligible new hours, tokenizer coverage/feasibility failure, or historical leakage invalidating a required selection panel.
- **Immediate process stop:** non-finite training, resource violation, or co-tenant protection trigger. Never repair this by disturbing another workload.
- **Stop long-run admission:** no eligible pilot winner, or inability to complete pilots within their budget.
- **Stop training:** failed 20k gate, registered plateau, 150k-step ceiling, allocated main-run time ceiling, or exhausted cumulative GPU budget.
- **After sealed evaluation:** failure of the registered headline criterion means report failure. No alternative checkpoint, decoder, normalization, corpus, or extra training round within Phase 4.
