**Council L: finish U’s admission pilot; run N only; drop the control completion; retain N40; make N60 diagnostic; retain H=150k without plateau stopping; replace 350→525 with a fixed 350 GPU-hour ceiling.**

This review uses the supplied training records. I independently checked the schedule and cost arithmetic; I have not verified or changed the running jobs. The N-only directive is settled. The additional gate and hardware changes below are **prospective policy amendments requiring recorded adoption**, not implementation fixes.

**Q1. Reading — yes.**

N is the sole production training procedure. The A-vs-N comparison is deferred to a future registration, with no automatic launch or promised year. Existing A-based corpus-admission evidence remains usable.

Distinguish two stronger readings:

- **“Execute no further Phase-1 weights, even for admission”:** stop U and use the already-admitted MaCro corpus, option (b). Historical admission evidence does not disappear.
- **“Reject all admission evidence obtained using Phase-1 weights”:** no final corpus is accepted under the surviving machinery. A new admission policy is needed. An 8k N pilot cannot simply inherit the A-based admission thresholds; calling all cold-start 8k evidence “uninformative” is nevertheless too absolute.

Also correct the FACTS: **A-conditioned admission is established; “the buckets were chosen by A’s errors” is not.** Council J explicitly rejected that blanket assertion.

**Q2. Pilot U — pick (a), FINISH.**

Spend the remaining **4,649 updates, approximately 5.44–6.43 GPU-hours**, to close the existing data decision. This is the last authorized A-initialized admission experiment; its checkpoint never becomes a production initialization.

At its normal 8k completion, apply the existing full admission test against O_ref, **m=4, quantile 0.99375**, including:

\[
S_U \le \min\left(S_{O_{\rm ref}}-0.30,\ \min_{\text{passing singles}}S+0.20\right).
\]

The displayed union ceiling is **20.280 pp**; authoritative **unrounded artifact values** govern.

- **U passes:** N uses **1,179,937 rows / 1,841.3 h / 875.4 unique new h**.
- **U fails its completed 8k admission:** N uses MaCro’s corpus: **1,084,696 rows / 1,692.4 h / 726.5 unique new h**.
- A registered **early pilot abort** remains terminal; it is not the ordinary failed-admission fallback.

Option (b) is a defensible explicit cancellation amendment, saving approximately six hours and dropping MADASR’s **149 h**. I would finish the decision already underway. **Reject (c):** the initialization directive does not waive union admission.

**Q3. Control completion — DROP it prospectively.**

Delete §8s.4’s requirement to complete O_A6000 and its consequence that failed qualification makes N wait for the A5000. Save approximately **4.6–5.5 GPU-hours**.

The reason is the removal of the planned initialization comparison. It is **not** evidence that the hardware qualification passed or that historical comparisons are unaffected:

- O_ref and the admission arms still have different hardware histories.
- The 8k agreement test remains **NOT COMPLETED**.
- Preserve the known 4k OpenSLR difference: **−0.546 pp**, exceeding the **0.30 pp** bound.
- Admission bootstrap intervals still omit training-run and hardware variability.

N may launch on the A6000 after its **own implementation, resume and resource qualification**. Keep the **12,000 MiB launch floor**, **13.0 GiB watchdog**, and co-tenant protections.

**Q4. N’s gates — retain 40k; report 60k; introduce no later terminal performance gate.**

| Update | Exact requirement or report | Consequence |
|---|---|---|
| **40,000** | S ≤ **22.0296**; D ≤ **21.348**; D empties ≤ **0.1%** | Failure: **terminal Phase-4 ending**. Pass: continue. |
| **60,000** | Report whether S ≤ **19.5296**, D ≤ **19.35**, every D source ≤ frozen P1 + **1.0 pp**, conversational ≤ **22.075%**, empties ≤ **0.1%** | **Diagnostic milestone only.** Failure does not stop training or exclude the procedure. |
| **150,000** | N40 passed; integrity requirements hold; at least one eligible checkpoint exists | **Normal ending**, followed by registered selection and evaluation. No eligible checkpoint: terminal, no headline. |

For D’s **5,385 recordings**, the empty-output limit permits **at most five empties**.

This explicitly removes the requirement to demonstrate the registered improvement by 60k. It is a relaxation, justified prospectively by choosing to fund the sole procedure through its full annealing schedule—not proof that a struggling run will recover.

**Phase 4 terminates** now means: durable terminal record, no A continuation, no alternative N run, no automatic production selection, no held evaluation or FLEURS headline. Screen failure establishes failure of the spending policy; full-budget performance remains unknown.

**Q5. Plateau — no. Keep N’s full 150k schedule.**

The packet’s later LR values are wrong. For the registered post-warmup cosine,

\[
\eta(u)=10^{-5}+\frac{0.00049}{2}
\left[1+\cos\left(\pi\frac{u-2000}{148000}\right)\right],
\]

the values are:

| Update | LR | Multiple of terminal LR |
|---|---:|---:|
| 60k | 3.36619×10⁻⁴ | 33.66× |
| 70k | 2.86119×10⁻⁴ | 28.61× |
| **100k** | **1.35515×10⁻⁴** | **13.55×** |
| **120k** | **5.80207×10⁻⁵** | **5.80×** |
| **140k** | **1.54990×10⁻⁵** | **1.55×** |
| 150k | 1.00000×10⁻⁵ | 1× |

Therefore **100k is not the proposed low-LR checkpoint**. Do not transplant A’s 50k-anchor/70k-earliest-stop rule. With one run and an affordable full schedule, I would preserve the annealing tail.

**Q6. Selection and evaluation — yes.**

Select only from N: minimum unrounded eligible S, then the **earliest checkpoint within 0.05 pp of that minimum**. Freeze it before any held decode.

Evaluate that one checkpoint on:

- Holdout: **6,767** recordings.
- Vaani validation/test: **10,105 / 12,695**.
- FLEURS test: **920**, one scoring event.

That is **29,567 held utterance-decodes**, plus **920 FLEURS**, excluding latency work.

Drop the two-finalist decoding, A–N paired intervals and generalisation-winner rules. Preserve trajectory/provenance recording and the **paired FLEURS comparison with already-frozen P1/P2 hypotheses**.

**Do not add nearest-S checkpoints to held evaluation.** Their results would provide local checkpoint sensitivity, not an independent initialization comparison. The useful protection is a predeclared complete report: aggregate and available source/dialect strata, uncertainty, empties, substitutions/insertions/deletions, and coverage limitations. A FLEURS win must not obscure poor holdout or Vaani results; those results cannot switch the checkpoint.

**Q7. Budget and calendar.**

The packet’s **265–300 GPU-hours** omits or compresses evaluation at the upper end:

\[
90 + [175.44,207.47] + 8
= \mathbf{273.44\text{–}305.47\ GPU\!-\!h}.
\]

Allow approximately **1 additional GPU-hour for N qualification**, giving **274–307 GPU-hours** before any validation overhead excluded from the quoted update rates.

**Register a fixed aggregate ceiling of 350 GPU-hours; delete the automatic 525 escalation.** This leaves approximately **43–76 GPU-hours** for remaining overhead and reserve. Measure whether the quoted throughput includes validation; do not count that time twice or omit it.

Conditional arithmetic at **723–855 updates/hour**, all times UTC:

| Q2 option | Assumed N launch | Aggregate plan¹ | N reaches 150k | With 8 h final work |
|---|---|---:|---|---|
| **(a) Finish U — recommended** | Sep 8, 13:15 | **274–307 GPU-h** | Sep 15, 20:41 → Sep 17, 04:43 | Sep 16, 04:41 → Sep 17, 12:43 |
| **(b) Stop; use MaCro** | Sep 8, 07:45 | **≈268–301 GPU-h** | Sep 15, 15:11 → Sep 16, 23:13 | Sep 15, 23:11 → Sep 17, 07:13 |
| **(c) Stop; use unadmitted U** | Hypothetically as (b) | Approximately as (b) | Same arithmetic as (b) | **Not authorized** |

¹ Includes the stated evaluation and qualification allowances. Launch readiness, validation overhead, interruptions and N40 failure change these forecasts. The **13:15 launch is conditional**, not a deadline that permits skipping qualification.

**Q8. Required record wording.**

> **Admission:** “The production corpus was admitted using Phase-1-initialized data pilots under the registered development score, regression guards and minimum-hours rules. No NVIDIA-initialized corpus ablation was performed. These decisions are conditional on the evaluated Phase-1-initialized systems and do not establish the best corpus or each bucket’s benefit for the NVIDIA-initialized procedure.”

Use the following weight statement **after the tensor-provenance audit passes**:

> **Weights:** “No Phase-1 model parameters or buffers were imported into the Phase-4 production model. Its weight lineage is NVIDIA English FastConformer-CTC → Phase 4, with the entire English CTC head replaced by a freshly initialized Bengali head, including blank parameters. Phase 4 retains the Phase-1 tokenizer, audited corpus components, historical evaluation evidence and Phase-1-based data-admission process.”

> **Deferred comparison:** “The owner deferred the Phase-1-versus-NVIDIA training-procedure comparison to a future, separately authorized registration. Phase 4 does not establish which initialization would perform better after completed training.”

> **Hardware:** “The registered 8,000-update cross-card agreement test was not completed. Its completion requirement was withdrawn when the owner selected a single production procedure. The known 4,000-update OpenSLR discrepancy is preserved. Admission comparisons remain conditional on their trained systems and hardware histories; their bootstrap intervals do not quantify training-run or hardware variability.”

**Q9. Exact proposed §8t.**

```markdown
## 8t. Owner-directed NVIDIA-only production — 2026-09-08

Owner, verbatim: "you know what... I have changed my mind, I will not use
phase 1 weight at all... we will do that experiment later, maybe next year"

Status: the owner's NVIDIA-only production directive is effective.
The policy provisions below become operative upon recorded owner adoption,
before any init-N training update, including GPU qualification updates.
They are prospective amendments, not implementation fixes.

This section supersedes the operative requirements in §8q/§8r/§8s for an
A production continuation, an A-versus-N comparison, A-relative gates,
A-dependent phase termination, completion of O_A6000, cross-procedure
selection, paired-finalist held decoding and the 350-to-525 budget escalation.
Historical experiments, findings and deviations remain recorded.

1. Production scope and corpus

Init N is the only production training procedure in Phase 4. No A
production continuation is authorized. The initialization comparison is
deferred to a future registration with its own authorization, budget,
corpus, evaluation design and stopping rules.

The existing A-initialized data-admission evidence remains in force.
The already-running pilot_U may complete its registered 8,000 updates
solely to resolve corpus admission. No further A-initialized training is
authorized by this section.

If U completes normally, apply its existing full admission test against
O_ref, m=4, bootstrap quantile 0.99375, including all point, statistical
and regression requirements, and:

S(U8) <= min(
    S(O_ref8) - 0.30,
    min over passing single-bucket arms of S(arm8) + 0.20
).

Use authoritative unrounded values; the displayed union ceiling is 20.280.

A passing U supplies N's frozen corpus: 1,179,937 rows, 1,841.3 hours,
875.4 unique new hours. If completed U fails admission, use the frozen
O+macro_filtered corpus: 1,084,696 rows, 1,692.4 hours, 726.5 unique new hours.

A registered early pilot abort remains terminal under the existing policy.
No unadmitted union, subset search, additional filtering, sampling change
or init-N 8k admission substitute is authorized.

A pilot checkpoint supplies no production model, optimizer or training
state. Preserve historical evidence under the existing retention policy;
there is no requirement to maintain an executable A continuation.

2. Initialization and recipe

N loads the archived nvidia/stt_en_fastconformer_ctc_large parent.
Record its full hash and revision and any unresolved identity against the
Phase-1 parent artifact.

Preserve the intended NVIDIA encoder parameters and BN buffers. Replace
the entire English CTC head with a fresh 1,025-output Bengali head,
including every blank parameter. Record its seed and initialized hash.
Import no Phase-1 model parameters or buffers.

Retain the frozen tokenizer, targets, preprocessing, architecture, corpus,
AdamW settings, effective batch 252, augmentation and shard_seed=20260905.
Verify all intended encoder parameters are trainable.

Production begins at update zero with fresh optimizer, scheduler, sampler
and RNG state. Warm up from 1e-5 to 5e-4 over 2,000 updates, then cosine
to 1e-5 at H=150,000. No second warmup, LR reset, head freeze or retuning.

Use a distinct N-only run and policy identifier. Bind checkpoints,
validation records and decisions to initialization, head, tokenizer,
corpus, configuration and code identities. Refuse mismatched resumes.

Validate D/M/R/K every 2,000 updates, with S weights .60/.20/.10/.10.
Report every validation. No comparison with an A training trajectory is
required.

3. Performance screen and reported milestone

All comparisons use unrounded values and the existing 1e-9 pp
floating-representation tolerance.

At literal update 40,000, require all:
- S <= 22.0296;
- D <= 21.348;
- D empty-output rate <= 0.1%.

Failure is a terminal Phase-4 ending. No fallback production procedure,
automatic selection, held evaluation or headline follows.

A PASS must be durable and checkpoint-bound. Resume beyond 40,000 requires
the corresponding valid PASS. Crashes between the checkpoint and decision
must replay the owed validation and decision. Missing, duplicate or
mismatched decision evidence cannot be accepted as a PASS.

At literal update 60,000, report whether all former qualification
conditions hold:
- S <= 19.5296;
- D <= 19.35;
- every D-source WER <= its frozen P1 baseline + 1.0 pp;
- conversational WER <= 22.075%;
- D empty-output rate <= 0.1%.

This is a diagnostic milestone. A numerical failure neither stops N nor
excludes the procedure from later selection. Record measured values and
the outcome durably; replay an owed milestone report after interruption.
Do not require or fabricate GATE60K PASS.

This conversion explicitly removes the former terminal N60 qualification.
No replacement terminal performance gate is introduced at 100k or later.

N never requires or fabricates GATE20K. A's warm-start progress gate and
the data-pilot D/empty-output abort rules do not apply to N.

4. Endings and selection

N has no plateau stop and continues to H=150,000 subject to N40 and the
applicable numerical, integrity, resource and budget policies.

Reaching H after N40 PASS is a normal ending if at least one eligible
checkpoint exists. If none exists, record a terminal ending with no
production selection, held decode or headline.

Checkpoint eligibility remains:
- every D source <= its frozen P1 baseline + 1.0 pp;
- conversational WER <= 22.075%;
- D empty-output rate <= 0.1%.

After a qualifying normal ending, find the minimum unrounded eligible S
among this N run's registered validation checkpoints. Select the earliest
checkpoint within 0.05 pp of that minimum. Do not chain tie bands.
Preserve all checkpoints required by the existing eligibility and
tie-band retention rules. No averaging or decoder search is authorized.

N40 failure terminates Phase 4 without a fallback. It establishes failure
of the registered spending screen, not full-budget inferiority.

Existing infrastructure terminal-stop and owner-override provisions
remain applicable. STOP/ABORT records are not automatically cleared or
restarted. Graceful co-tenant handovers remain resumable.

No absent or hypothetical A20 result can terminate or authorize this
N-only procedure.

5. Launch, qualification and dispatch

Withdraw §8s.4's O_A6000 completion and agreement prerequisite and its
A5000-only fallback. Record the 8k cross-card test as NOT COMPLETED,
preserving all partial measurements and historical hardware caveats.

N may run on an eligible A6000 after its own qualification passes.
Retain the 12,000 MiB free-memory launch floor, 13.0 GiB watchdog,
polling interval no longer than 60 seconds, current co-tenant-demand
yield rules and checkpoint/release obligations. Never disturb a co-tenant.

Before production:
- verify parent and full-head tensor provenance, trainability and blank
  index consistency between targets, loss and decoder;
- exercise a disposable 60-update N qualification trajectory on the
  decided corpus, saving at 40 and replaying updates 41-60 after restore;
  this uses 80 executed updates and stays within the 100-update allowance;
- verify exact resumed cut IDs and LRs, complete state restoration,
  finite losses and gradients, validation and checkpoint writes;
- profile the final corpus's longest-clip batch on the intended card,
  including total process memory;
- restore the verified update-zero initialization for production.

The smoke is an engineering qualification, not a quality-selection event.
Its state is not selected for production and its cost is charged once.

Disarm the superseded post-union dispatcher and queued A/control actions.
Verify the active dispatcher and unit definitions reference the new
versioned N-only policy. Replacing a script pathname does not by itself
neutralise an already-running process.

6. Evaluation and claims

Freeze the selected N checkpoint, tokenizer, greedy decoder and scorer
before any held evaluation.

Decode that checkpoint only on holdout, Vaani validation and Vaani test.
SPRING remains excluded. Do not decode alternative near-S checkpoints
on these sets under this registration.

Predeclare the available source/dialect breakdowns and uncertainty
calculation before decoding. Publish aggregate and stratified results,
empty outputs and error decomposition, with coverage limitations.
Held results cannot switch the selected checkpoint.

Remove §8r's A-N paired-finalist decoding and comparative winner tests.
Retain provenance and trajectory reporting. No initialization superiority
claim is authorized.

FLEURS remains one 920-recording scoring event under §6, including the
paired 349-sentence-cluster bootstrap against frozen P1/P2 hypotheses:
20,000 draws, seed 20260905. Success remains WER <16.54% with the 95%
interval for P4-P2 entirely below zero; strong remains <=16.04%.
FLEURS success establishes the registered benchmark result, not broad
generalisation. Report contrary held-domain evidence prominently.

State explicitly that corpus admission used Phase-1-initialized systems
and does not establish the corpus's optimality or bucket effects for N.
State the NVIDIA-to-Phase-4 weight lineage and the retained Phase-1
tokenizer, corpus and admission influences.

The 8k cross-card agreement test remains uncompleted. Bootstrap intervals
condition on the evaluated trained systems and omit training-run and
hardware variability.

Repository privacy, owner-controlled visibility, licence designation and
applicable source-licence disclosures remain unchanged. Replace obsolete
claims of continued training from the Bengali Phase-1 checkpoint.

7. Budget

Replace the conditional 350/525 GPU-hour ceilings with one fixed aggregate
ceiling of 350 GPU-hours. N40 PASS does not raise it.

Charge prior work, remaining U training, qualification, production,
validation, final evaluation and reserve exactly once.
Measure validation overhead and held-decode throughput before treating
the forecast as confirmed. Only the owner may raise the ceiling.

Budget truncation does not establish full-budget inferiority or authorize
an automatic headline. Do not shorten or reset the registered cosine
schedule to manufacture completion within the ceiling.

8. Required acceptance cases before production

- N40 decisions use only the absolute thresholds and need no A artifacts.
- A missing or failed A20 artifact cannot affect the N-only dispatcher.
- N at 20,500 resumes without GATE20K.
- N beyond 40k without its bound N40 PASS is refused.
- Crashes at N40 replay the decision; failure remains terminal across
  fresh supervisor invocations.
- A missed former N60 threshold is reported and N resumes at 60,500
  without GATE60K PASS.
- Later eligible checkpoints remain selectable after a missed N60 milestone.
- Normal H completion with an eligible checkpoint selects N.
- H completion without any eligible checkpoint authorizes no held decode.
- No A/control continuation or A checkpoint enters production selection.
- Ineligible lower-S checkpoints cannot evict the required eligible state.
- Early N empty-output diagnostics do not invoke A/pilot abort rules.
- Resume refuses changed initialization, corpus, recipe or policy identity.
```

**Q10. Adversarial — my pick is inherited control logic falsely declaring the cold start unsuccessful.**

The revealing number is **N60 LR = 3.366×10⁻⁴, 33.66 times terminal LR**. A terminal N60 failure would discard the procedure after **70–83 GPU-hours**, with **90,000 scheduled updates still unobserved**. That establishes neither NVIDIA inferiority nor defective labels.

The cheapest decisive check is CPU fault injection:

> N40 passes → N60 misses S=19.5296 → checkpoint/resume at 60,500 → training continues without GATE60K → a later eligible checkpoint remains selectable.

The fresh-head blank behaviour is the early version of this same risk. Using Phase 1’s reported **≈6.3 encoder frames/token**, a valid alignment with one nonblank spike per token can be **≈84.1% blank**. High blank occupancy alone is therefore not collapse. CTC’s tendency toward peaky alignments and its dependence on frame/target geometry are documented; the ratio must be remeasured for this corpus. [Zeyer, Schlüter and Ney](https://arxiv.org/abs/2105.14849)

Require the smoke and first **2k/4k validations** to expose:

- Actual encoder lengths, target lengths and adjacent-repeat counts; verify **L+R≤T** and registered feasibility. Inspect raw losses so infinity-to-zero handling cannot hide impossible alignments. [PyTorch CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
- Blank posterior, blank argmax fraction, decoded-token/reference-token ratio, empty rate and deletion rate.
- Nonzero finite gradients and parameter changes in the new head and intended encoder.
- Toy decoder checks that distinguish repeated tokens separated by blank from repeated frames of one token.

**Collapse is near-all-blank argmax, near-100% empty outputs, length ratio near zero and deletion-dominated WER—even while loss falls.** Healthy CTC can also have many blank frames.

At update 60 the LR is only **2.47×10⁻⁵**. Empty output there does **not** invalidate the recipe; the smoke cannot certify eventual learning. Persistent collapse at the first validations demands an immediate engineering diagnosis, not an automatic warm-start abort or an unregistered blank-bias/LR adjustment.
