**Run one staged init-N arm on the final admitted corpus.** Use the Phase-1-informed cold-start recipe, hold the one-shots, and describe the result as a comparison of **two fixed training procedures**. It will not isolate initialization from learning rate, or establish the best achievable result from either initialization.

The proposed amendment is below. No experiment or SPEC change is being represented as already completed.

1. **Q1 — What warm starting buys, and my prediction.**

   Init A carries a Bengali-trained acoustic encoder, Bengali CTC alignments, a trained Bengali output head, and Bengali-adapted BatchNorm statistics. Fresh AdamW discards optimization history; it does **not** discard those learned capabilities.

   Init N retains substantial English ASR pretraining, so “cold” means **cold with respect to Bengali adaptation**, not random initialization. NVIDIA describes this checkpoint as an English model pretrained on multiple speech corpora. [NVIDIA model card](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large)

   A’s risks are inherited source preferences, transcription conventions and errors from the old corpus. The expanded corpus is still **42.8–49.9% O by rows**, so neither procedure escapes old-data problems. At 150k Phase-4 updates, A has received approximately **47.88 million Bengali presentations including Phase 1**, versus **37.80 million for N**. That is a legitimate advantage when reusing an available checkpoint, but it is not equal lifetime training expenditure.

   My prediction: **A learns faster initially; at a literal 150k budget I weakly favor A retaining lower eligible S, with 60% subjective confidence.** That is a judgment, not an estimated probability. Thirty to thirty-five passes over the expanded corpus give N substantial opportunity to catch up. Nothing supplied establishes a permanent advantage for A, or supports a numerical FLEURS-gain forecast.

2. **Q2 — Choose 5e-4 and 2,000 warmup updates for N.**

   Use peak **5e-4**, linear warmup **1e-5 → 5e-4 over 2,000 updates**, then cosine to **1e-5 at 150,000**. Keep A unchanged.

   This answers: **“Which of these two specified training procedures produces the better eligible model under the registered spending policy?”**

   It does not answer “same hyperparameters, different initialization.” Nor does one recipe per initialization establish “best achievable.”

   Two corrections matter:

   - **2e-4 has not been demonstrated to handicap N.** It might slow adaptation, but that remains a hypothesis.
   - The proposed N schedule is **Phase-1-informed, not a replay of the proven Phase-1 schedule**: its cosine horizon changes from 40k to 150k.

3. **Q3 — N gets its own 40k and 60k gates.**

   The premise that N *cannot* pass A’s 20k gate is too strong. The historical run used different data, scheduling and evaluation conditions. Nevertheless, applying that gate to N would impose an unjustified requirement.

   Register these thresholds; WER values are percentages and margins are pp:

   | Decision | Required conditions |
   |---|---|
   | N at **40,000** | `S_N ≤ min(22.0296, S_A40 + 2.00)`; `D_N ≤ min(21.348, D_A40 + 2.00)`; empties ≤0.1% |
   | N at **60,000** | `S_N ≤ 19.5296`; `D_N ≤ 19.35`; every existing P1-based final guard passes |
   | Failure of either | Terminate **N only**, exclude its checkpoints from final selection |
   
   Here, **22.0296 = S(P1)+1.00** and **21.348 = D(P1)+1.00**. The 40k comparison uses both arms’ **literal 40k checkpoints**, not a moving best.

   These are deliberately permissive spending thresholds, **not empirically calibrated predictions or noninferiority tests**. N need not satisfy the per-source and conversational release guards at 40k; it must satisfy them at 60k.

   Do **not** inherit the pilot’s `D >25.35` or early empty-output abort. A fresh Bengali head can initially produce poor or empty hypotheses. Numerical, configuration and resource failures remain immediate stops.

4. **Q4 — Choose option (a), with explicit limits on what it proves.**

   Hold FLEURS, holdout and Vaani’s protected one-shots until both procedures have resolved. **SPRING remains outside this phase under §8f**; the FACTS sentence must not reinstate its evaluation splits.

   Among qualifying runs, find the **global minimum unrounded eligible S**, then select the checkpoint with the fewest Phase-4 updates within **0.05 pp** of that minimum. If both update counts are equal, prefer A. Do not first apply separate arm-level tie selections and then compare those selections.

   Decode the selected winner once. This can confirm **the selected model against P2 on FLEURS**. It cannot confirm **N against A on FLEURS**, because the losing model receives no decode.

   A’s existing 20k failure remains fatal to Phase 4. **N is not a rescue branch.** I would launch N only after A passes that gate.

5. **Q5 — Finish the missing card control; do not assume the effect will be large.**

   Complete the original O_A6000 trajectory from its preserved resumable checkpoint to 8k. From 4k this costs approximately **4.7–5.5 GPU-h**. Require the original bounds against O_ref:

   **|ΔS| ≤0.10; |ΔD| ≤0.20; every D-source |ΔWER| ≤0.30 pp.**

   If it passes, concurrent N on the A6000 is acceptable for this **single-run procedure comparison**, with the remaining hardware uncertainty disclosed. If it fails, run N serially on the A5000 after A finishes. That reduces hardware differences; A’s mixed-card history still remains.

   The **1.58% cross-card versus 1.23% same-card loss-block deviations are not WER error bars**. Passing one completed control comparison also does not establish a universal hardware bound.

   There is a separate launch inconsistency: taking the supplied GB labels literally, the A5000 has **11.0 GB free before our allocation**, below the registered **12,000 MiB ≈12.58 GB** launch requirement. Reconcile live units and the operative launcher rule; “we fit” is not evidence that the registered launch condition passes.

6. **Q6 — Staged spending, with the complete conditional budget registered now.**

   Use 40k as the first spending decision, 60k as the qualification gate, and—if both pass—continue N to **150k without plateau stopping**.

   | N training extent | Additional update time at 855–723 updates/h | Existing 250–280 h plan plus that time |
   |---|---:|---:|
   | 40k | **46.8–55.3 GPU-h** | **296.8–335.3 GPU-h** |
   | 150k | **175.4–207.5 GPU-h** | **425.4–487.5 GPU-h** |

   These figures precede incremental control completion and any evaluation/qualification time not already counted. The claimed **430–440 h total is not a conservative upper bound**.

   Register **350 GPU-h through the 40k decision**, rising automatically to **525 GPU-h if N passes 40k**. Using the conservative 280 h existing plan, an additional 4k control completion, 2 h qualification, 8 h incremental evaluation and 12 h reserve gives approximately **515 GPU-h** for the full branch.

   The earlier “unbounded” quotation does not silently erase the later explicit **300 h ceiling**. These increases need recorded owner adoption before launch. Without it, continue the existing programme and leave this comparison untested.

   Sequence: finish pilots → resolve the final corpus, including union admission if necessary → resume A → complete the card check alongside A → launch N after A’s 20k PASS. A’s 8k→20k interval is approximately **14.0–16.6 update-hours**, plus validation and interruptions.

7. **Q7 — No new data-admission pilot and no second union.**

   N trains directly on the **identical final manifest used by A**. It does not compete against O_ref at 8k, and it does not trigger new bucket admission, cap changes or union construction. Data-admission multiplicity stays **m=4**.

   Identical: manifest bytes and order, decoded audio, targets, tokenizer bytes and token IDs, architecture, preprocessing, effective batch **252 =12×21**, AdamW settings other than the specified LR schedule, SpecAugment, scorer, panels, **2k** validation cadence, and `shard_seed=20260905`.

   Different: pretrained weights, complete CTC-head initialization, inherited BN statistics, LR/warmup, progress gates and the explicitly different stopping policy.

   **Critical timing correction:** if a distinct U needs its own 8k pilot, the selected training corpus is not settled at 03:00. N must wait for U’s admission or the MaCro fallback. Starting N on U and later changing its corpus would invalidate the comparison.

8. **Q8 — Conditions that would invalidate or substantially narrow interpretation.**

   | Problem | Number or check exposing it |
   |---|---|
   | Converting historical internal WER with a constant offset | Offset is **+0.631 pp** at restoration, **+1.465** on O at 8k, **+1.527** on MaCro at 8k |
   | Calling different recipes an initialization-only effect | N peak is **2.5×** A’s; warmup is **2×** |
   | Calling 40k N a completed replication of Phase 1 | Proposed N LR at 40k is approximately **4.245×10⁻⁴**, versus Phase 1’s terminal **1×10⁻⁵** |
   | Confusing equal updates with equal corpus passes or lifetime training | N has **7.96–9.29** final-corpus epochs at 40k; A also carries **40k historical Bengali updates** |
   | Silently changing the final corpus after union failure | Require identical manifest SHA-256 and matching per-update cut digests |
   | Claiming the corpus is optimal for N | Bucket admission and corpus selection were performed using **A** |
   | Promoting an ineligible checkpoint or counting it toward plateau improvement | Final conversational limit stays **22.075%**, with all P1 source guards unchanged |
   | Treating bootstrap precision as repeat-training precision | One run per procedure; sentence bootstrap conditions on these trained systems |

   For N, retain all eligible candidates correctly but apply **no plateau stop** after its 60k qualification. This prevents the 50k anchor/70k earliest-stop policy from truncating the cold arm during a materially different learning-rate phase. A’s plateau policy stays unchanged. Consequently, this is a comparison of procedures with budgets **up to** 150k, not necessarily equal realized training.

   Also, §8p overstates what a plateau establishes: ten past non-resets do **not** prove that future training would leave the selected checkpoint unchanged.

   The most dangerous implementation carryover is the trainer’s warm-specific resume invariant. **N at 20,500 without GATE20K must be permitted; A in that state must still ABORT.** N needs its own durable 40k and 60k decisions, owed-validation replay and resume invariants. Never manufacture a fake N “20k PASS.”

9. **Q9 — Permitted claims after selection.**

   If N wins:

   > “The selected checkpoint was adapted directly from NVIDIA’s English FastConformer-CTC model using the retained Bengali tokenizer and the frozen, admitted Phase-4 corpus. Under the registered development selection rule, this specified procedure outperformed the tested Phase-1 warm-start procedure.”

   Its **weight lineage** becomes NVIDIA → Phase 4. Phase 1 still supplied the tokenizer, historical corpus work, auditing instruments and the initialization used for corpus admission. Its contribution was not erased. Preserve the private repositories and existing provenance/licence disclosures; removing P1 weights does not remove the corpus’s licence history.

   If A wins after N completes:

   > “Reusing the Phase-1 checkpoint produced the selected eligible model among the two tested procedures under the registered development objective and spending policy.”

   If N is screened out:

   > “The cold-start procedure failed the registered continuation screen. Its performance after 150,000 updates is unknown.”

   Neither outcome establishes an initialization-only causal effect, universal warm-start superiority, or a sealed A-versus-N comparison.

10. **Q10 — The most likely wrong conclusion.**

    **Stopping N at 40k and reporting that warm starting gives better final accuracy.**

    The revealing number is **42.45×**: N’s proposed LR at that point is approximately 42.45 times Phase 1’s terminal LR. The comparison would conflate a spending decision during ongoing adaptation with a completed optimization comparison.

    The cheapest check is CPU-only: print the realized LR sequence and endpoint, alongside the frozen-surface trajectory, and enforce the “screened out; full-budget outcome unknown” reporting state. This catches the invalid inference. It cannot establish the unobserved 150k outcome. Stopping at 40k saves **128.7–152.1 update-hours**; accepting that uncertainty is the reason to stage the experiment.

Append the following as the operative amendment **after owner adoption of its budget terms**:

```markdown
## 8q. Initialization-procedure comparison — staged init N

Status: proposed; effective upon recorded owner adoption, before any init-N
training update. This amendment does not retrospectively establish that any
procedure, gate or budget was registered before its adoption.

Purpose: compare two specified training procedures on the final admitted
Phase-4 corpus. This is not an initialization-only causal experiment and does
not establish the best achievable performance from either initialization.
The proposal follows inspection of development results, and the corpus was
selected using init A. Results are conditional on that adaptive development.

1. Corpus and launch

Complete the existing bucket and union decisions without alteration. Init N
uses exactly the final manifest selected for A, including any MaCro fallback.
A candidate union is not the final selected corpus until its admission is
resolved. No speculative init-N training on an unresolved union is permitted.

No new 8k admission comparison against O_ref, union search, data filtering,
sampling change or tokenizer training is performed for N. The data-admission
family remains m=4.

A resumes its existing 8k state with optimizer, scheduler position, sampler,
buffers and RNG intact. Its 20k progress gate and terminal-stop rules remain
binding. Launch N only after A has a durable 20k PASS. A failure of that gate
terminates Phase 4; N cannot rescue it.

2. Initialization and recipe

Use distinct initialization and corpus identifiers in every artifact; init N
must not be confused with corpus N_final.

A remains unchanged.

N loads the archived NVIDIA stt_en_fastconformer_ctc_large checkpoint.
Record the full checkpoint hash and revision; verify its identity against the
Phase-1 parent artifact where available and disclose any unresolved identity.
Use the same Phase-4 architecture and preprocessing configuration.

Replace the entire English CTC head, including blank parameters, with a fresh
1,025-output head for the existing 1,024-piece Bengali tokenizer. Equal head
shapes do not authorize retaining English token-indexed weights. Record the
head initialization seed and resulting hash. Preserve NVIDIA encoder weights
and BN buffers; import no Phase-1 model tensors. Verify tensor loading and
confirm all intended encoder parameters remain trainable.

N starts fresh AdamW and fresh sampler/scheduler state at update zero:
betas (0.9, 0.98), weight decay 1e-3, gradient clip 1.0, bf16,
effective batch 252 = 12 x 21, and the existing SpecAugment configuration.
Warm up linearly from 1e-5 to 5e-4 over 2,000 updates, then cosine to 1e-5
at H=150,000. Use that 150k schedule from update one, including the probe.
No 40k cosine endpoint, second warmup, LR reset, head freeze or later retuning.

Manifest bytes, targets, tokenizer, preprocessing, panels, scorer, batch,
augmentation configuration and shard_seed=20260905 are identical to A.
Validate D/M/R/K every 2,000 updates with weights .60/.20/.10/.10.
Verify per-update cut order against A's corresponding prefix. Identical seeds
and cut order do not assert identical cross-card augmentation realizations.

3. N progress gates

All decisions use unrounded frozen-surface values. The existing 1e-9 pp
floating-representation tolerance is retained; it is not substantive slack.

At N update 40,000, require ALL:
  S_N40 <= min(22.0296, S_A40 + 2.00)
  D_N40 <= min(21.348, D_A40 + 2.00)
  D empty-output rate <= 0.1%.

Use exactly one validation record for each literal 40k checkpoint.
If A's comparator is unavailable, N waits at its resumable 40k checkpoint;
no later, earlier or best-so-far comparator is substituted.

The 40k thresholds are resource-screening policies, not noninferiority tests
or estimates derived from the historical cold-start curve. Full P1 source
and conversational eligibility is not required to pass this screen.

At N update 60,000, require ALL:
  S_N60 <= 19.5296
  D_N60 <= 19.35
  every D-source WER <= its frozen P1 baseline + 1.0 pp
  conversational WER <= 22.075%
  D empty-output rate <= 0.1%.

Failure of either N gate terminates N, excludes all its checkpoints from
production selection, and does not terminate an otherwise valid A run.
Report the outcome as failure of the registered continuation policy, with
N's unobserved full-budget performance unknown.

The warm-only 20k gate and the data-pilot D/empty-output abort rules do not
apply to N. Numerical, configuration, checkpoint-integrity and resource
failures remain terminal for N under the applicable infrastructure policy.

Each N decision is durable, bound to its run, initialization, corpus,
configuration and checkpoint. Resume must replay any owed validation and
decision. N resumes past 40k/60k require the corresponding N PASS records.
Never create a fictitious N GATE20K PASS.

4. Stopping and selection eligibility

A's horizon, plateau, retention and final guards remain unchanged.

After passing 60k, N continues to H=150,000 without plateau stopping.
A qualifying normal ending for N is reaching H after passing both gates.
A terminal N outcome cannot authorize headline selection from N.

Both procedures use the existing P1-based checkpoint eligibility guards.
Ineligible checkpoints remain diagnostic only and cannot become selected.
No eligible N checkpoint by 60k implies failure of its qualification gate.

Required acceptance cases include:
- N resume at 20,500 without GATE20K is accepted;
- A resume at 20,500 without its required PASS is refused;
- crashes between each N gate checkpoint and decision replay that decision;
- missing or duplicate gate records are refused;
- terminal N failures are not restarted;
- eligible-checkpoint retention and global tie selection remain correct.

5. Hardware qualification

Complete the original O_A6000 control to 8,000 on the A6000 from its
preserved original resumable state, as a qualification continuation.
Do not reset its optimizer, sampler or schedule.

Require absolute differences against O_ref of:
  S <= 0.10 pp; D <= 0.20 pp; every D source <= 0.30 pp.

If qualification passes, N may run concurrently on the A6000.
If it fails, N runs serially on the A5000 after A ends normally.
An unavailable qualification state is not a PASS.

Record card assignment, migrations, software versions, realized throughput
and interruption history. Existing memory, polling and co-tenant demand
rules remain binding. The completed control is evidence about these control
realizations, not a bound on training-run or hardware variability.

6. Budget

Upon owner adoption, replace the Phase-4 aggregate ceiling of 300 GPU-h
with 350 GPU-h through N's 40k decision. If N passes that decision, the
aggregate ceiling automatically becomes 525 GPU-h, subject to its 60k gate.
Authorize both limits before observing any N result.

Charge all Phase-4 spending, including prior work, control completion,
qualification, both procedures, evaluation and reserve. Count each cost once.
A keeps scheduling priority. Do not shorten or alter A to finance N.

Without recorded adoption of these ceilings, N is not launched.
A resource-truncated N run is not evidence of full-budget inferiority.

7. One-shots and final selection

Hold FLEURS, holdout and Vaani's protected one-shots until both procedures
have resolved. SPRING remains excluded from this phase.

Only checkpoints from procedures that qualify and end normally enter final
selection. A's existing phase-terminal outcomes do not authorize automatic
promotion of N or a headline.

Across all eligible checkpoints of qualifying procedures, find the minimum
unrounded S. Select the checkpoint with the smallest Phase-4 update count
within 0.05 pp of that global minimum. If update counts also tie, select A.
Do not apply chained or separate arm-level tie bands.

Freeze the selected checkpoint, tokenizer, decoder and scorer before any
one-shot decode. Decode only that model on each authorized one-shot.
The FLEURS success rule remains WER <16.54% with the registered paired
95% CI against P2 entirely below zero; strong remains <=16.04%.

Report paired sentence-clustered development comparisons with the existing
coupled-panel bootstrap. These intervals condition on the trained systems
and do not quantify training-run variability or independently confirm the
development selection. A winner-only FLEURS decode does not test A versus N.

8. Claims and provenance

Report matched-update trajectories, eligibility, panel contributions,
actual spending and stopping reasons for both procedures. Distinguish
equal Phase-4 updates from unequal lifetime Bengali training exposure.

If N is selected, its weight lineage is NVIDIA English -> Phase 4.
Disclose the retained Phase-1 tokenizer, corpus provenance and A-based
admission process. If A is selected, describe selection among the two tested
procedures, not universal superiority of warm starting.

A screened-out N run supports no claim about its 150k performance.
Repository privacy, owner-controlled visibility and existing provenance
and licence disclosures remain unchanged.
```
