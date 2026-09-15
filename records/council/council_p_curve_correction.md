**Verdict: (d) survives. Continue the existing H=120,000 run without restarting or changing hyperparameters. Make 40k diagnostic; apply terminal qualification to the already prescribed selected checkpoint after completing 120k. Reject the proposed 27% terminal floor.**

This is a recommendation for recorded owner adoption. The existing terminal N40 policy remains operative until amended.

**Q1. The correction changes my assessment of the run, but not my policy choice.**

I retract the statement that N was 6.4 pp behind at 2k, and every projection constructed from that deficit.

| Update | N internal D | Correct P1 internal D | N’s lead over P1 raw | N’s lead over P1 best observed by then |
|---:|---:|---:|---:|---:|
| 2,000 | 46.702 | 56.5503 | 9.8483 pp | 9.8483 pp |
| 4,000 | 34.460 | 39.8213 | 5.3613 pp | 5.3613 pp |
| 6,000 | 30.412 | 36.4926 | 6.0806 pp | 2.3995 pp |

These are favourable early observations. They remove the stated evidence of a slow start. They do **not** supply a calibrated probability of N40 failure or eventual success.

The false premise was load-bearing for my description of progress and the numerical illustrations. It was **not necessary to the rationale for (d)**: one production procedure, no fallback, an uncalibrated irreversible screen, and 80,000 scheduled updates remaining at that screen. That rationale survives independently.

The cost changes in probability, not magnitude:

- If N would pass N40 anyway, adopting (d) adds **zero training updates**.
- If N would fail N40, completing the trajectory adds **92.0–97.6 training GPU-h**.
- Wasted additional spending concerns the joint event **“N40 fails and completion produces no qualifying model.”**
- Avoided premature rejection concerns **“N40 fails but completion produces a qualifying model.”**

The corrected evidence makes me less concerned about an unsuccessful start. It does not identify either joint probability. I cannot honestly turn the early lead into a numerical reduction in expected wasted GPU-hours.

**Q2. Rule the restart out. No hyperparameter defect is visible.**

**Do not discard approximately eight GPU-hours and restart on this evidence. Change no hyperparameter.**

Retain:

- Warmup: **2,000 updates**, from **1e-5 to 5e-4**.
- Cosine endpoint: **1e-5 at 120,000**.
- Effective batch: **252**.
- Existing AdamW, clipping, SpecAugment and frozen tokenizer settings.

At 4k and 6k there are **zero empty outputs**. From 4k to 6k, frozen D improves **4.149 pp**, conversational WER improves **3.320 pp**, and S improves **4.3245 pp**. The supplied losses and diagnostics identify neither divergence nor blank collapse.

However, the orchestrator should replace **“proven at this scale”** with **“previously successful on the smaller Phase-1 corpus.”** The present corpus is **1.720× larger in hours and 1.962× larger in rows**, and the cosine horizon is three times longer. Those differences prevent a claim that this exact training procedure is already validated.

There is no measured basis for a different peak LR, warmup, batch, augmentation or horizon. An early lead does not establish optimality; an erroneous early deficit does not justify restarting.

**Q3. No single historical alignment answers both questions.**

| Question | Appropriate comparison |
|---|---|
| Is N learning effectively at its current expenditure? | Approximately matched-update **raw internal trajectories**, assessed over several validations and with matched validation cadence. |
| Has N surpassed P1’s best observed development performance at that expenditure? | Running minima with their **real timestamps**; disclose unequal validation opportunities. |
| How much of the scheduled decay remains? | Matched **cosine fraction**. |
| Will literal N40 pass S/D/empty thresholds? | **None of these is a validated predictor.** Only N40 decides under the existing policy. |

At 6k, P1’s raw value is **3.6811 pp above its preceding best**. Consequently, the apparent 6.0806-point lead contains a large historical rebound; the 2.3995-point comparison against P1’s best is less flattering.

Running minima introduce another issue: P1 was evaluated approximately every 1k, N every 2k. P1 has more opportunities to record a favourable minimum. Report that explicitly; do not treat the minimum as an unbiased latent learning curve.

Matching cosine fraction answers a schedule question, not an exposure question. N40 corresponds to P1 approximately **14,237** by cosine fraction, but those states have different update counts, corpus exposure and source mixtures.

**Noise ruling:** a **2 pp allowance is insufficient even to cover the supplied historical excursions**. P1 rebounded:

- **3.6811 pp** at 5k→6k;
- **3.2052 pp** at 7k→8k;
- **3.1145 pp** at 16k→17k.

Any allowance advertised as covering *all observed P1 excursions above its preceding minimum* must therefore be at least **3.6811 internal pp**. That is an observed maximum, **not** a predictive bound, a frozen-WER allowance or a confidence interval. These excursions also do not establish that epoch phase is their sole cause.

**There is no defensible universal numerical noise allowance for a single future terminal validation from this one trajectory.** That limitation argues against deriving a new irreversible threshold from it.

**Q4. Choose (d); reject the proposed derivation of (e).**

The screen is **unsuitable as an evidence-based test of whether this cold start can produce a useful completed model**. It remains a coherent, deliberately restrictive spending policy. Whether N happens to pass does not validate its false-rejection behaviour.

But the packet overstates its historical counterexample. **“Phase 1 itself would have failed this screen” is not established.** P1 has no literal 40k validation, and its schedule-matched 14k checkpoint is not an equivalent N40 state.

The 27% proposal has four specific problems.

1. **The frozen conversion is unverified.** Subtracting offsets observed on other checkpoints does not reconstruct historical frozen WER.

2. **The proposed range uses only the favourable neighbouring validation.** Applying the packet’s illustrative offsets of 0.631–1.400 pp gives:

   | P1 observation | Illustrative frozen conversion | After adding the proposed 2 pp |
   |---|---:|---:|
   | 14k: 25.4767 internal | 24.0767–24.8457 | 26.0767–26.8457 |
   | 15k: 26.7297 internal | 25.3297–26.0987 | **27.3297–28.0987** |

   Thus **27.0 does not even cover the neighbouring historical observation under the proposal’s own conversion assumptions**.

3. **Two points do not cover the measured excursions.** Replacing two with 3.6811 would cover a historical construction, but still would not calibrate N’s future frozen WER.

4. **One successful historical run cannot identify a boundary separating recoverable from unrecoverable runs.** There is also no historical S trajectory here from which to derive a composite floor.

The measured **4.6508 pp** improvement from P1’s 14k observation to its best checkpoint is real. Calling it the gain N should receive from annealing is not justified: it includes further training and selection from a fluctuating trajectory. The projected **16.70 frozen D** is therefore not a forecast.

Also, D improvement cannot be compared directly with the FLEURS target. The registered displayed FLEURS baselines differ by **19.67−16.54 = 3.13 pp**, not 3.03 pp.

My exact choice remains:

> **40k diagnostic; no replacement early performance floor; qualification after H=120,000, applied to the existing S-selected eligible checkpoint using S≤22.0296, D≤21.348 and all existing eligibility guards.**

Those are **permissive qualification limits**, not proof that Phase 4 improved on Phase 1. The stricter 60k targets remain diagnostic, and benchmark success remains separately defined.

| Error direction | Consequence |
|---|---|
| Continue a run that ultimately fails qualification | **137.9–146.3 total production training GPU-h**, including **92.0–97.6 h more** than stopping at N40. |
| Stop a run at N40 that would later qualify | **46.0–48.8 h already spent**, the sole production opportunity lost, and **92.0–97.6 h saved** by foregoing its remaining trajectory. |

These exclude validation overhead, prior experiments and final evaluation. The aggregate ceiling remains **350 GPU-h**.

**Q5. Use better descriptive comparisons; do not manufacture a passing-trajectory classifier.**

**No quantity available at 8k/12k/16k/24k can establish that eventual qualification is impossible.** The useful distinction is between an observable warning and a forecast.

Keep the direct qualification margin:

\[
m_u=\max(S_u-22.0296,\;D_u-21.348).
\]

At 6k:

\[
S\text{ margin}=7.4871,\qquad
D\text{ margin}=7.6640,\qquad
m_{6000}=7.6640\ \text{pp}.
\]

That is distance from the limits, not predicted remaining improvement.

For historical shape, use a **cadence-matched rolling median**. Let \(P_k\) and \(N_k\) denote the internal D readings associated with nominal update \(k\), retaining actual event steps in the underlying record:

\[
Q_X(u)=\operatorname{median}\{X_{u-4000},X_{u-2000},X_u\},
\qquad
r_u=Q_N(u)-Q_P(u).
\]

This compares three observations over the same update window, using P1’s approximately 2k-spaced observations.

Also report the historical best comparison:

\[
B_N(u)=\min_{v\le u}N_v,\qquad
B_P(u)=\min_{t\le u}P_t,\qquad
a_u=B_N(u)-B_P(u),
\]

where \(B_P\) uses the full corrected history and is explicitly labelled as more densely evaluated.

The requested numeric boundaries are:

| N update | \(Q_P(u)\): boundary where \(r_u=0\) | Full-history \(B_P(u)\): boundary where \(a_u=0\) |
|---:|---:|---:|
| 8,000 | **36.4926%** | **29.4937%** |
| 12,000 | **29.4156%** | **26.1886%** |
| 16,000 | **25.4767%** | **24.5142%** |
| 24,000 | **22.6201%** | **22.3241%** |

Interpretation:

- \(r_u>0\) at **two consecutive scheduled validations** flags a sustained lag against the smoothed matched-update history.
- \(a_u>0\) means N’s best observed score no longer leads the more densely evaluated P1 history.
- Neither condition establishes failure of N40 or eventual qualification. Neither is a stopping rule.

These replace O’s invalid **28.368/26.869** boundaries and every illustration using a persistent **+6.4 pp** deficit.

For domain monitoring, define four-thousand-update improvements:

\[
I_X(u)=X_{u-4000}-X_u.
\]

Flag **\(I_S>0\) while \(I_D\le0\) or \(I_{\rm conv}\le0\)** at two consecutive validations. Report the four weighted contributions:

\[
0.60I_D,\quad0.20I_M,\quad0.10I_R,\quad0.10I_K.
\]

This exposes improvement concentrated away from D or conversation, using existing decodes. It authorizes diagnosis, not automatic retuning or termination.

**Q6. Exact proposed amendment text.**

> **§8w. NVIDIA-only performance policy after correction of the Phase-1 trajectory**
>
> **Status:** proposed; effective upon recorded owner adoption before observation of N40. This section prospectively replaces the terminal N40 performance screen. Correcting the historical curve does not itself authorize changing the running policy.
>
> **1. Training state and scope**
>
> Continue `N_O+macro_filtered_H120k` from its existing state. Retain H=120,000, the registered LR schedule, optimizer, sampler, RNG, initialization, tokenizer, corpus and validation schedule. No restart, hyperparameter change, additional production procedure or fallback is authorized.
>
> Version the amended policy. Record adoption time, current update, all results available at adoption, and an explicit migration from the previous policy identity. Verify that migration changes no training state. Unrelated identity mismatches remain refused.
>
> **2. Update 40,000**
>
> At literal update 40,000, record exactly one authoritative checkpoint-bound validation and milestone report containing S, D, panel scores, D empty count/rate, source and conversational guards, and whether each former N40 condition holds:
>
> - S≤22.0296;
> - D≤21.348;
> - D empties≤0.1%, meaning at most 5 of 5,385 recordings.
>
> Record the conjunction as `former_N40_satisfied`. It is diagnostic. A numerical miss creates no STOP, no terminal ending and no procedure exclusion. Training continues subject to the unchanged numerical, integrity, resource and budget rules.
>
> Bind the report to the run, policy, initialization, corpus, configuration and checkpoint identities. Resume must replay an owed validation/report idempotently. Resume beyond 40k requires valid milestone-completion evidence, **not a successful performance outcome**. Recover missing evidence from the corresponding saved checkpoint or verified validation record before advancing. Unrecoverable, duplicate or mismatched evidence is an integrity failure, not a performance failure. Never fabricate or require GATE40K PASS under this policy.
>
> **3. Update 60,000**
>
> Retain the diagnostic milestone: S≤19.5296, D≤19.35, each D source≤its exact frozen P1 baseline+1.0 pp, conversational WER≤22.075%, and D empties≤0.1%.
>
> Record measured values and each condition durably. Replay owed reporting after interruption. A numerical miss neither stops training nor excludes later checkpoints. No GATE60K PASS is required.
>
> **4. Completed-horizon selection and qualification**
>
> Complete update 120,000 and its validation. No plateau stop applies.
>
> Eligibility remains: every D source≤its exact frozen P1 baseline+1.0 pp; conversational WER≤22.075%; D empties≤0.1%.
>
> Among this run’s eligible registered validation checkpoints, find the minimum unrounded S. Select the earliest checkpoint within 0.05 pp of that minimum, without chained ties. Preserve the existing required checkpoint retention.
>
> Apply qualification to **that selected checkpoint**, which need not be the literal 120k checkpoint:
>
> - S≤22.0296;
> - D≤21.348;
> - all existing eligibility guards.
>
> Retain the 1e-9 pp representation tolerance. Do not substitute another checkpoint to obtain a qualification pass.
>
> No eligible checkpoint, or failure of selected-checkpoint qualification, produces a durable **terminal ending**: no production promotion, held decoding or headline.
>
> Successful qualification produces a durable **normal ending** and authorizes the existing frozen-selection and one-shot evaluation procedure. Finalization is checkpoint-bound and idempotent; interruption must replay owed finalization without training beyond H or duplicating held evaluation.
>
> **5. Unchanged constraints**
>
> No N20 gate, replacement early performance gate, fallback, LR reset, schedule extension, averaging or decoder search is introduced. The aggregate ceiling remains 350 GPU-h. Existing infrastructure and terminal-stop provisions remain binding, including the owner’s production no-yield amendment. Held results cannot change the selected checkpoint. Qualification alone is not benchmark success or evidence of broad improvement.
>
> Before deploying this policy, verify acceptance cases covering: an N40 numerical miss continuing across resume; missing/duplicate/mismatched milestone evidence; N60 misses; crashes during horizon finalization; qualification of an earlier selected checkpoint despite an ineligible H checkpoint; and terminal qualification failure never authorizing held decoding.
>
> **6. Disclosure**
>
> “The original policy required terminal performance qualification at 40,000 updates. Reconsideration initially followed an erroneous claim that Phase 4 lagged Phase 1 at 2,000 updates. The historical curve had assigned EarlyStopping improvement events to a fabricated 1,000-update grid. Correct event-step extraction reversed that comparison. At this review, Phase 4 led Phase 1’s reported internal D WER by 9.8483, 5.3613 and 6.0806 points at approximately matched 2k, 4k and 6k updates.
>
> “Before observing N40, the owner adopted diagnostic reporting at 40k and qualification of the prescribed selected checkpoint after completing 120k. Adoption time, update and all results then available are recorded. This was a data-informed prospective amendment following an evidence correction, not an unchanged preregistration. The outcome under the former N40 numerical conditions and the additional spending are reported. Neither the early lead nor this amendment establishes eventual or held-domain superiority.”

**Q7. Correct the dependent claims; do not reopen unrelated decisions.**

The error record should state:

> “Nineteen EarlyStopping improvement-event values were assigned invented updates 1,000–19,000. Their actual updates span 1,998–35,005. The supplied nineteen values match the improvement-event sequence exactly. Non-improving validations were omitted. The corrected artifact contains 39 validations with their original event steps; no 40,000-update validation exists. Claims using the former time axis are withdrawn. Frozen final-model baselines, pilot measurements and their derived thresholds were not generated from this curve.”

Preserve the erroneous artifact and affected packets as superseded evidence. Record hashes and extraction provenance. Verify the corrected extractor preserves non-improvements and actual steps, including the first validation and the 5k→6k regression.

| Council | Consequence of correction |
|---|---|
| **I** | Revisit historical cold-start timing claims and any confidence attributed to that curve. Loader provenance, full-head replacement, recipe specification and comparison-design requirements do not depend on it. Its gates were explicitly policy screens, not calibrated historical predictions. |
| **J** | Retract any corrupted-curve comparisons. “No empirical A-versus-N test on this corpus,” the required completed-procedure comparison, and held-evaluation standards remain correct. Do not mechanically update its subjective 40% forecast into a new probability. |
| **K** | Owner-directed ordering, removal of unavailable A-relative comparators, checkpoint preservation and hardware requirements are unaffected; the two-procedure sequence is subsequently superseded anyway. |
| **L** | **Revisit retaining terminal N40.** NVIDIA-only scope, diagnostic N60, no plateau, corpus-admission caveats, hardware disclosure and single-model evaluation remain justified independently. |
| **M** | No demonstrated dependency: its described subject was GPU yielding, subsequently superseded by the owner. The complete verdict is not supplied here, so a line-by-line certification would be unwarranted. |
| **N** | **Keep H=120,000, no plateau and diagnostic N60. Revisit terminal N40.** Horizon/exposure arithmetic and throughput savings survive. |
| **O** | Withdraw the +6.4 deficit, corrupted historical timing, 1.62-point illustration, extrapolation tables and associated disclosure. Reaffirm (d) with the corrected reasoning above. |

Audit H’s historical-curve references too, because the stated propagation begins there. Its measured pilot results and gate/resume fixes are not invalidated by this error.

Council N needs one precision correction:

- P1’s **completed run**: 40,000 updates, **18.2301 epochs**.
- P1’s **selected checkpoint**: 35,005 updates, **15.9536 epochs**, LR approximately **3.064e-5**.
- Equivalent nominal exposure on today’s corpus: **78,469** and **68,670** updates, respectively.

Thus the 18.23-epoch anchor remains valid as a **completed-run budget**, not the exposure embodied by `final.nemo`. N’s 120k horizon still gives **27.8788 epochs**. There is no resulting evidence for another horizon change.

Likewise, O’s description of the selected P1 model as fully terminal-annealed was too strong. The corrected 13k-to-best internal improvement is **5.1211 pp**, not its previously cited 1.62 pp.

The correction changes **none** of these measured decisions: MaCro admission; Vaani’s inconclusive exclusion; U’s conversational-guard failure; the production corpus; frozen baseline thresholds; or the owner’s NVIDIA-only instruction.

**Q8. The likeliest mistake is over-correction: treating “ahead early” as “will generalise well.”**

The revealing current numbers are:

- Lead over P1’s historical best: **9.8483 → 5.3613 → 2.3995 pp**.
- At 6k, frozen D remains **7.664 pp above** the proposed final D qualification limit.
- Conversational WER is **31.457%**, or **9.382 pp above** its eligibility limit.

Lead compression is not itself failure: learning curves need not preserve additive gaps, and validation cadence affects minima. But it prevents an honest claim that the initial advantage can simply be carried forward.

The cheapest check is the existing validation report: **\(m_u\), conversational eligibility, source guards, rolling historical residuals and weighted panel contributions**. In particular, flag improving S accompanied by stalled D or conversation. This requires **zero additional GPU decoding**.

Preserve the original N40 outcome and compare it with final qualification:

- Original N40 misses, final selected checkpoint qualifies: the old policy would have rejected a trajectory that later met the amended requirements.
- Final qualification fails: the extra expenditure did not produce a qualifying model.
- Qualification passes but held domains disappoint: completing training did not establish the generalisation the owner wanted.

**Continue the current training. Correct the evidence. Amend the spending policy explicitly. Do not restart, invent a 27% safety guarantee, or promise that the early lead will survive.**
