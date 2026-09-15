**NO-GO as written.** The plan needs a defensible NC release basis, an audit of the Phase-1 checkpoint’s training history, and a bounded experiment that distinguishes useful new data from merely doing more training.

This reviews your supplied measurements; it is not a file-level audit of the downloaded bundle. Probabilities below are subjective estimates of failure under the sketch without mitigation. WER costs are adverse scenarios, not measured predictions. Risks overlap; do not add their probabilities.

1. **The required NC publication has no resolved ShareAlike basis.**  
   **Mechanism/evidence:** OpenSLR53 is CC-BY-SA-4.0 and already exists inside the Phase-1 lineage. NC is not a compatible ShareAlike addition. Whether the weights themselves trigger those obligations requires a separate analysis.  
   **Probability:** 100% that this release question is unresolved today—not 100% that publication would be unlawful.  
   **Cost:** Publication blocked after up to **200 GPU-h**.  
   **Cheapest rule:** Before training, require **one written disposition covering the actual checkpoint ancestry and proposed NC weights release**, plus a licence record for **100% of retained sources**. A Hugging Face licence tag is insufficient. [OpenSLR53](https://www.openslr.org/53/), [CC compatibility rules](https://creativecommons.org/compatible-licenses/).

2. **The new MaCro dev slice can already be inside the warm-start checkpoint.**  
   **Mechanism/evidence:** MaCro and OpenSLR share prompts. Carving only against the new mixture cannot remove exposure accumulated during Phase 1. The tokenizer’s training text also counts.  
   **Probability:** 45%.  
   **Cost:** Invalid development evidence; potentially the entire long run.  
   **Cheapest rule:** Select the **3–4 h** MaCro slice only after checking every candidate group against **all Phase-1 training audio and text, tokenizer inputs, and Phase-4 training candidates**. Require **zero confirmed or unresolved matches**. An inherited FLEURS-test match means the clean-headline claim fails; later removal does not repair it.

3. **“17.5 GiB” does not establish that the co-tenant is protected.**  
   **Mechanism/evidence:** GPU compute, memory bandwidth, CPU workers, page cache and storage remain shared. A memory-safe process can still disrupt another workload.  
   **Probability:** 40% without resource admission controls.  
   **Cost:** Violation of a standing constraint, independently of WER.  
   **Cheapest rule:** Require an enforceable resource allocation before **any GPU profiling**. If the co-tenant uses the same GPU, require an exclusive scheduled window. Permit **zero foreign-process signals, GPU resets or global clock changes**. Without an enforceable arrangement, allocated training time is **0 GPU-h**.

4. **“Human-written prompt” gets promoted to “human-verified transcript.”**  
   **Mechanism/evidence:** MaCro train annotations were algorithmically cleaned; the competition explicitly distinguishes them from manually reviewed valid annotations. A speaker can omit, repeat or substitute words while reading. A plausible prompt is not proof of what was spoken.  
   **Probability:** 70% of overstating supervision quality.  
   **Cost:** Reputational damage; **+0.5–4 pp** if mismatches are sufficiently common.  
   **Cheapest rule:** Trace **100% of admitted rows** to an original human prompt/transcript and an allowlisted transformation. Admit **zero unexplained ASR-written replacements**. Use the terms *prompt-derived human labels* and *human-reviewed labels* separately. [Competition data description](https://www.kaggle.com/competitions/bengaliai-speech/data).

5. **Shrutilipi returns wearing a WER-filter badge.**  
   **Mechanism/evidence:** You measured a **24.6 pp** loss, blank collapse, and **91.8% WER / 22.5% empty** on its low-cps slice. Whisper agreement does not establish correct audio boundaries or human verification of the segment labels.  
   **Probability:** 60% of material harm **if reintroduced**.  
   **Cost:** **+5–25 pp**, or a failed run.  
   **Cheapest rule:** **0 Shrutilipi rows; 0 GPU-h auditioning it.** Keep the 142.6 h excluded.

6. **Path deduplication misses the same recording under different names.**  
   **Mechanism/evidence:** There are 165,873 duplicate CSV lines, several repackaged corpora, known shared prompt pools, and many recovered OpenSLR clips. Re-encoding, trimming and renaming defeat file/path equality.  
   **Probability:** 75% of additional duplicates or cross-split matches.  
   **Cost:** **0.2–3 pp apparent improvement** that may be leakage; redundant training.  
   **Cheapest rule:** Run canonical-PCM and robust partial-audio matching across **all corpora and protected panels**. Require **zero unresolved train/evaluation audio matches**. Calibration and pass rules appear below.

7. **The “speaker-disjoint holdout” loses that status after adding MaCro.**  
   **Mechanism/evidence:** MaCro has no speaker IDs. Different sentences from the same speaker evade text and recording-duplicate checks.  
   **Probability:** 100% that global speaker separation remains unverified.  
   **Cost:** An indefensible generalisation claim.  
   **Cheapest rule:** Require **zero shared IDs** for sources with recoverable speaker metadata. Report MaCro separation as **unknown**. Make **zero claims of globally verified speaker disjointness**; speaker embeddings cannot certify its absence.

8. **WER<15 retains correlated errors and removes useful difficulty.**  
   **Mechanism/evidence:** Selection rewards agreement with a particular Whisper lineage. Three rounds can reinforce the same accent, vocabulary and transcription preferences. They are not three independent quality checks.  
   **Probability:** 65% of meaningful distribution distortion.  
   **Cost:** **+0.5–3 pp** on FLEURS despite improving MaCro dev.  
   **Cheapest rule:** Before freezing manifests, audit **400 uniformly sampled retained clips per filtered MaCro/MADASR source**, using Bengali-speaking reviewers blind to teacher output. Require reference-label WER against adjudicated human transcripts **≤2%**, and **≤4/400** clips with material mismatch exceeding 20% WER. Also inspect **100 tail cases per source** spanning duration, dialect metadata and text length. A failed corpus gate stops the proposed experiment.

9. **More clean read speech dilutes the mixture that already worked.**  
   **Mechanism/evidence:** MaCro plus MADASR contributes approximately **1,389 h**, or **54%** of the nominal 2,570 h. IndicVoices drops from **59.0% to 22.6% by hours**. Your own capped-mixture experiment lost 4.7 pp.  
   **Probability:** 60%.  
   **Cost:** **+0.5–3 pp**; conversational degradation can be larger.  
   **Cheapest rule:** Preserve every eligible original row, freeze source counts and sampling multiplicities, and use the **one registered original-versus-expanded pilot** below. Report shares by **utterances, audio seconds and realised optimiser exposure**, not hours alone. Do not invent another source cap after seeing results.

10. **Previously rejected OpenSLR/Kathbath rows are silently rehabilitated.**  
    **Mechanism/evidence:** Some “new” rows were removed for cleaning or carving. Their rejection may encode leakage, bad labels or invalid audio. Moreover, the OpenSLR counts conflict:  
    **207,380 − 49,398 = 157,982 shared**, and **157,982 + 8,603 = 166,585 ours**. Thus 166,585 cannot simultaneously be the stated intersection.  
    **Probability:** 100% of a bookkeeping inconsistency; 50% of consequential reinstatement errors.  
    **Cost:** **+0.2–3 pp** or contaminated panels.  
    **Cheapest rule:** Reconcile the set identities exactly. Require an original rejection reason for **100% of recovered rows** and **zero reinstated split exclusions**. Unexplained rows remain excluded.

11. **Punctuation deletion changes the supervised word sequence.**  
    **Mechanism/evidence:** Your scorer replaces punctuation with spaces; his cleaning removes punctuation. Deleting a hyphen can concatenate two words. There are **9,363 shared OpenSLR rows** with these changes. Dropped digit-containing rows also narrow coverage.  
    **Probability:** 80%.  
    **Cost:** **+0.2–2 pp** from inconsistent targets and lost number coverage.  
    **Cheapest rule:** Recover original human text and apply the frozen normaliser. Check **all 9,363 differences** programmatically; permit **zero unexplained lexical changes**. Do not guess missing spaces from the cleaned CSV. Keep number-containing examples from the original eligible corpus; freeze **100 train-side normalisation fixtures**.

12. **A “cheap” full-corpus model audit consumes the budget.**  
    **Mechanism/evidence:** At the supplied Qwen latency, **1.42 million × 1.461 s = 576 GPU-h** for a serial pass. Batching may help, but no throughput measurement establishes the saving.  
    **Probability:** 50% of a serious underestimate if re-filtering is added casually.  
    **Cost:** **Tens to hundreds of GPU-h**.  
    **Cheapest rule:** Use existing filtering provenance and CPU/human checks. Cap **all GPU preflight work at 8 h**. Any proposed inference pass first gets a **1,000-clip timed sample**, then a full-cost forecast including failures and I/O. No unbudgeted filtering rounds.

13. **“40k steps” is mistaken for the old amount of training.**  
    **Mechanism/evidence:** At 1.42 million rows and batch 252, an epoch is **5,635 updates**. Forty thousand updates cover **7.10 passes**, versus Phase 1’s **18.23**. Matching Phase 1’s actual passes needs approximately **102,725 updates**.  
    **Probability:** 85% without explicit exposure accounting.  
    **Cost:** An undertrained run, or **30–100 GPU-h** of surprise extensions.  
    **Cheapest rule:** Freeze counts after auditing. Log optimiser updates, utterance presentations and audio seconds separately. Register the final update limit before launch; **zero extensions** based on “still improving.”

14. **Warm-start optimisation is neither a true resume nor a defined new run.**  
    **Mechanism/evidence:** Resuming the exhausted cosine schedule can pin learning near its floor; restarting at 5e-4 creates a **50× jump from 1e-5**. Old Adam moments also reflect a different mixture.  
    **Probability:** 45%.  
    **Cost:** **+1–5 pp**, or **10–100 GPU-h** spent learning ineffectively.  
    **Cheapest rule:** Load Phase-1 weights, initialise fresh AdamW, and register one schedule: **1,000-update warmup from 1e-5 to 1e-4; cosine to 1e-5 at update 100,000**. Keep the other Phase-1 optimisation settings. No LR sweep or schedule restart after a failed gate.

15. **The tokenizer loads successfully but the label space is wrong.**  
    **Mechanism/evidence:** Equal vocabulary sizes do not imply equal token IDs. A new Bengali BPE can silently invalidate the trained CTC head. Keeping the old BPE can instead expose previously unseen characters or inefficient segmentation.  
    **Probability:** 30%.  
    **Cost:** **+0.3 pp through total decoding failure**.  
    **Cheapest rule:** Require exact hashes for tokenizer files, token-ID ordering, blank ID and decoder configuration. Encode **100% of admitted transcripts**; require **zero unknown-token occurrences** and lossless round-trip text under the registered normalisation. No tokenizer replacement during Phase 4.

16. **Eightfold subsampling makes some CTC targets impossible.**  
    **Mechanism/evidence:** Short, dense or incorrectly paired clips can have too few encoder frames. Adjacent repeated target tokens need additional alignment frames. `zero_infinity` can conceal the resulting failures.  
    **Probability:** 30%.  
    **Cost:** **+0.5–10 pp**, silent exposure loss or collapse.  
    **Cheapest rule:** For every admitted row require  
    **T_encoder ≥ U_tokens + R_adjacent_repeats**, using the actual frontend’s length calculation. Require **zero infeasible examples**, **zero blank IDs in targets**, and **zero nonfinite training losses**. Instrument the loss before any infinity-zeroing. [PyTorch CTC documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CTCLoss.html).

17. **BatchNorm sees a different distribution from the advertised batch 252.**  
    **Mechanism/evidence:** NeMo’s Conformer convolution supports BatchNorm. Accumulating 21 microbatches does not make its statistics equivalent to one batch of 252. Source-sorted batches can rapidly change running statistics.  
    **Probability:** 30%, conditional on BatchNorm in the pinned configuration.  
    **Cost:** **+0.2–2 pp**.  
    **Cheapest rule:** Inspect the actual checkpoint configuration. Preserve **100% of normalisation buffers** on load; mix sources through a deterministic global shuffle; verify evaluation changes **zero buffers**. Run a **zero-update restoration check** before training. Do not add an unregistered “BN recalibration” rescue. [NeMo implementation](https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/submodules/conformer_modules.py).

18. **Adding MaCro changes the gate’s meaning.**  
    **Mechanism/evidence:** A 3.5 h supplement becomes approximately **29% of panel hours**. A pooled score can improve because the new domain improves while the original domains regress. Hishab already demonstrates a severe dev/FLEURS ranking reversal.  
    **Probability:** 80%.  
    **Cost:** **50–150 GPU-h** following a misleading gate.  
    **Cheapest rule:** Preserve the original **5,385-row score separately**. Freeze supplement membership without model scores. Use explicit panel weights and per-source regression limits; never compare the new pooled score with an old-panel historical number.

19. **Teacher benchmark exposure becomes invisible selection leakage.**  
    **Mechanism/evidence:** Later filtering rounds may have different training ancestry from the initial OpenSLR+TTS teacher. Zero overlap among retained labels does not establish that filtering was independent of evaluation data.  
    **Probability:** 25% of material undocumented exposure; ancestry is currently incomplete.  
    **Cost:** Invalid claims of benchmark-independent selection.  
    **Cheapest rule:** Record **all three reported filtering rounds**, known teacher training sources and available model/release identifiers. Any known use of protected evaluations for teacher fine-tuning or filter selection fails the independence claim. Unknown foundation-model ancestry must be disclosed, not converted into a “zero contamination” certificate.

20. **Additional training gets credited to dataset quality.**  
    **Mechanism/evidence:** Phase 1 was still improving. A successful 100k-update continuation cannot establish that MaCro/MADASR caused the gain. The supplied 19-point curve also lacks the exact step mapping needed to extrapolate its slope.  
    **Probability:** 90% of an unsupported causal claim without a control.  
    **Cost:** Reputational; possibly **100+ GPU-h** pursuing unnecessary data.  
    **Cheapest rule:** Register exactly **two 8,000-update pilot arms**, both starting from the identical Phase-1 checkpoint: original corpus versus accepted expansion. Use the same schedule prefix and decoder. This tests early incremental benefit; without a full-length control, the final claim remains **“expanded-data continuation improved this checkpoint,”** not “data quality caused the full gain.”

21. **Static manifests still fail audited resume semantics.**  
    **Mechanism/evidence:** Static input does not freeze shuffling, worker RNG, sampler cursor, accumulation state or dropped final batches. The duplicate-line history makes unintended sample weighting plausible.  
    **Probability:** 55%.  
    **Cost:** **5–30 GPU-h** of replay or unrecorded exposure differences.  
    **Cheapest rule:** Hash manifests and store row multiplicities. Compare uninterrupted versus interrupted execution across **two epoch boundaries**: require identical sample IDs and LR values for the next **1,000 updates**, with **zero unrecorded omissions or repetitions**. Checkpoint at optimiser boundaries with sampler and RNG state.

22. **The completed download is mistaken for a verified, affordable dataset.**  
    **Mechanism/evidence:** The bundle is only 77% downloaded. Headers, durations and decoding are unverified. A complete 2,570 h corpus stored as 16-kHz mono PCM16 occupies approximately **296 GB**, regardless of archive size.  
    **Probability:** 60% of an integrity or storage/throughput surprise.  
    **Cost:** **5–40 GPU-h**, plus shared-box disruption.  
    **Cheapest rule:** Decode **100% of admitted files on CPU**; reject empty/corrupt audio and inconsistent metadata. Verify sample counts and recalculate hours. Require free space of **1.2× remaining expanded bytes plus 10 GiB for checkpoints**. Never silently skip failed files during training.

23. **The confidence interval answers the wrong statistical question.**  
    **Mechanism/evidence:** There are 920 recordings but only 349 sentence clusters. Independent clip resampling, unpaired model resampling or averaging per-clip WER produces a different and potentially misleading estimate.  
    **Probability:** 60% without a frozen executable scorer.  
    **Cost:** A false superiority claim.  
    **Cheapest rule:** Freeze **20,000 paired bootstrap replicates**, seed **20260905**, resampling **349 sentence clusters** with replacement. Include every recording in each selected cluster; calculate each model’s corpus WER from summed edits/reference words. Primary success requires the **upper endpoint of the two-sided 95% CI for WER₄−WER₂ to be <0**.

24. **“Opened once” becomes “never previously used,” and a gated release becomes “open and reproducible.”**  
    **Mechanism/evidence:** FLEURS results already informed Phases 1–3 and external comparisons. A Phase-4 seal cannot undo that history. Private pipeline code also prevents independent public reproduction.  
    **Probability:** 100% if those stronger descriptions are used.  
    **Cost:** Credibility of the publication.  
    **Cheapest rule:** State **one final Phase-4 scoring event**, not a virgin benchmark. Select **one checkpoint and one decoding configuration** before that event. Describe the release as **gated, noncommercial weights with a private pipeline**. Make **zero SOTA, universal Bengali superiority or fully public reproducibility claims** from this experiment alone.

The GPU plan I would register is deliberately finite:

| Allocation | Maximum |
|---|---:|
| GPU preflight, restoration checks, profiling | 8 h |
| Original-corpus control: 8,000 updates | Included below |
| Expanded candidate: 100,000 updates total, including its pilot | Included below |
| Combined training at a verified ≥800 updates/h | 135 h |
| Validation and final evaluation | 15 h |
| Registered interruption/recovery allowance | 12 h |
| Unallocated safety margin | 30 h |
| **Hard total** | **200 h** |

The measured rate must apply to the actual batch configuration and mixture. Failed runs, inference, profiling and recovery all count. Start graceful shutdown by **195 cumulative GPU-h**, with an independent watchdog terminating only Phase-4 processes by **198 h**. The remaining time is margin, not another experiment.

For the pilot, let \(W_o\) be frozen-scorer WER on the original panel and \(W_m\) WER on the new MaCro panel. Register \(S=0.75W_o+0.25W_m\).

At **8,000 updates**, continue only if all conditions hold:

- Expanded \(S\) beats control \(S\) by **at least 0.30 pp**, with paired sentence-clustered 95% CI upper endpoint below zero.
- Expanded \(W_o\) is no more than **0.20 pp worse** than control.
- No original source is more than **1.0 pp worse** than its Phase-1 checkpoint result.
- MaCro WER improves by **at least 0.50 pp** versus the restored Phase-1 checkpoint.
- Original-panel empty-output rate is **≤0.1%**.

At **32,000 updates**, require another **0.20 pp reduction in \(S\)** against the fixed 8k checkpoint, retaining the regression guards. Failure stops the experiment. Validate every **4,000 updates**, select the eligible checkpoint with lowest \(S\), and break ties by earliest update. There is no extension past 100k.

These are resource-allocation rules. Passing them does not guarantee a FLEURS improvement, and failing them does not prove the corpus is intrinsically bad.

On **(i), yes: filtered MaCro/MADASR can make FLEURS worse than 19.57%.** There are four direct routes: residual wrong labels, teacher-induced selection bias, dilution of useful original domains, and optimisation damage during adaptation.

The WER threshold itself is length-dependent. Under strict WER<15%, a six-word reference must match the teacher perfectly, whereas a twenty-word reference can retain two errors. That selects different kinds of examples at different lengths. Removing digit-containing rows further changes coverage. Also, “dialect-rich before filtering” does not imply “dialect-rich after filtering.”

The target requires a **3.03 pp absolute / 15.48% relative** reduction from Phase 1. Neither the Kaggle win nor an increase to roughly 2,000–2,500 h establishes that such a gain transfers to this CTC model and FLEURS.

On **(ii), warm-starting is the preferred budget choice, but an undefined resume is a trap.** I would retain the Phase-1 weights and tokenizer, reset the optimiser, use the single conservative schedule above, and preserve normalisation buffers initially.

The strongest reasons to reject that checkpoint are **inherited evaluation contamination or an unresolved release-rights problem**. Forgetting, LR mismatch and normalisation drift are measurable optimisation risks. Tokenizer lock-in becomes a blocker if coverage or CTC feasibility fails; it is not a reason to casually rebuild a 1,024-piece vocabulary and reuse its old decoder.

On **(iii), the main unresolved publication issue is OpenSLR ShareAlike versus mandatory NC release—not an assumed research-only MADASR licence.**

- **MADASR/RESPIN:** IISc’s terms dated **19 May 2023** explicitly incorporate **CC-BY-4.0**. I found no research-only restriction in that document. Confirm that it governs the exact acquired release and preserve its attribution/modification notices. The licence is no longer simply “unknown.” [IISc terms](https://ee.iisc.ac.in/madasrdataset/RESPIN_Data_TAC.pdf).
- **MaCro:** Kaggle’s rules permit commercial and noncommercial use, but §7.B restricts redistribution to nonparticipants; the rules also address conflicts with the linked licence. Preserve lawful acquisition and do not mirror competition audio or transcript-bearing manifests into HF or shared GitHub access. A repost’s availability does not resolve its authority. [Competition rules](https://www.kaggle.com/competitions/bengaliai-speech/rules).
- **OpenSLR53:** Shared source material retains its CC-BY-SA rights; covered adaptations must follow ShareAlike requirements. Whether trained weights constitute such an adaptation is fact- and jurisdiction-dependent. Creative Commons explicitly distinguishes that legal question from its conservative recommendation to apply the training material’s SA licence to shared models. **Neither automatic inheritance nor automatic exemption is established merely by calling something “weights.”** [CC guidance on AI training](https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf).
- **NVIDIA base weights:** The named model card specifies **CC-BY-4.0**; retain the applicable attribution and modification notices. [NVIDIA model card](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large).

Since NC is non-negotiable, resolve the OpenSLR issue through a documented applicable legal basis or appropriate rights-holder permission **before committing GPU time**. Removing OpenSLR only from Phase 4 does not remove its Phase-1 ancestry. Gating is access control, not a licence cure.

The HF card should contain wording equivalent to this, with bracketed fields completed and claims verified:

> This model continues checkpoint [hash], trained using the Bengali corpus documented in [ancestry record]. Phase-4 training used [N] unique recordings / [H] hours after the registered provenance, deduplication and split-exclusion checks.
>
> Selected human-origin labels were obtained through tugstugi’s Bengali.AI 2023 data release. The author reports three rounds of Whisper-based filtering using WER<15% against supplied transcripts. These are model-filtered human labels; they are not all individually human-verified transcripts. The reported filtering teacher’s initial training included TTS. Synthetic audio and pseudo-label targets were excluded from our Bengali training manifests.
>
> Upstream processing included punctuation removal and exclusion of Bengali-digit-containing rows. Our additional changes were [exact transformations and exclusions]. Filtering provenance is reported from [sources]; [specific unreproduced or unknown details] remain unverified.
>
> OpenSLR53 source material is licensed under CC-BY-SA-4.0, with attribution to Google and the dataset authors. Upstream rights are retained; our NC release terms do not relicense that material. The documented basis for releasing these weights under [exact NC terms] is [specific permission or legal disposition].
>
> FLEURS-test was scored once for Phase 4 after checkpoint and decoder selection. Earlier programme phases had already reported results on this benchmark. MaCro speaker separation cannot be verified. Pipeline code is private.

Also publish a source-by-source table of versions, original creators, links, licences, final hours/counts, exclusions and known supervision limitations. **An unresolved bracket is a publication blocker, not acceptable final card text.**

On **(iv), I would require the following leakage audit beyond sentence equality.**

1. **Audit the whole accessible lineage.**  
   Include Phase-1 training manifests, tokenizer text, Phase-4 candidates, original and repackaged source metadata, locked dev/holdout panels, and the proposed MaCro supplement. Reserve official dev/test files explicitly; names such as `test_known` do not establish speaker separation.  
   **Pass:** Every admitted row has provenance and an assigned role; **zero unexplained assignments**.

2. **Compare several text representations.**  
   Keep the frozen scoring representation, plus audit-only forms for punctuation deletion, joiners, digit variants and spacing differences. Generate near-match candidates using **character edit similarity ≥0.90 for strings ≥20 characters**, **token 5-gram containment ≥0.80**, and substantial substring matches covering **≥80% of the shorter sentence**.  
   **Pass:** **Zero unresolved protected-panel matches**. These thresholds generate exclusions/review candidates; they do not alter scoring.

3. **Hash decoded audio and detect transformed copies.**  
   Compute canonical decoded-PCM hashes, then use indexed acoustic fingerprints for re-encoding, gain, resampling, offsets and partial clips. Check **≥2-second matching spans**, and **≥80% coverage of shorter clips** where full-span matching is needed. Avoid an all-pairs implementation: 1.42 million rows imply approximately **one trillion pairs**.  
   **Pass:** Remove confirmed training-side matches to protected evaluation recordings; retain **zero unresolved candidates**.

4. **Calibrate fingerprint detection before touching protected sets.**  
   Use **200 train-only recordings**, each with six transformations: compression, gain, offset, resampling and two speed changes. Require **≥99% recovery in each transformation class**. Test **10,000 unrelated pairs plus 1,000 same-prompt/different-recording pairs**; require no falsely confirmed duplicate recordings.  
   **Pass:** Calibration thresholds are frozen. Failure stops the audit; thresholds are not tuned using test matches.

5. **Separate recording identity from prompt identity.**  
   Multiple speakers legitimately reading the same prompt are distinct audio examples. Deduplicate recording copies; group shared prompts for carving. Build components linking original IDs, audio duplicates, transcript variants and known source passages.  
   **Pass:** **No component crosses a protected train/evaluation boundary**. Report unique recordings and unique prompt groups separately.

6. **Trace FLEURS to FLORES sentence identities and source passages.**  
   FLEURS uses sentences from FLORES dev/devtest. Check the corresponding IDs, alternate Bengali renderings, split/merged sentences and matching passages in any prompt collections or auxiliary text. Mere use of Wikipedia is not itself a match.  
   **Pass:** Account for **all 349 protected sentence clusters**, with **zero known matching passages admitted** to new training or auxiliary language-model text. Unknown foundation-model exposure remains a stated limitation. [FLEURS documentation](https://huggingface.co/datasets/google/fleurs).

7. **Keep the audit separate from model selection.**  
   Run protected-reference matching through a fixed audit process that returns training exclusion IDs and aggregate counts. Quarantine ambiguous candidates rather than using test examples to improve modelling decisions.  
   **Pass:** **Zero model hypotheses, WER values or test-driven threshold changes** before the final scoring event.

No finite fingerprint or string audit proves universal absence of leakage. The defensible claim is **“zero detected overlap under these specified checks, with these remaining ancestry and speaker limitations.”**

On **(v): I would forbid Shrutilipi outright, including the WER<15 subset.** It has the strongest directly measured adverse evidence in this programme. The permitted allocation is **zero rows and zero GPU-hours**.
