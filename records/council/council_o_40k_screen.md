**Council O verdict: amend the policy now. Choose (d): make 40k diagnostic and place the sole performance qualification at the completed 120k horizon.** Apply qualification to the checkpoint chosen by the existing eligible-S selection rule.

This deliberately accepts **92.0–97.6 additional training GPU-hours** if a run that would fail N40 ultimately proves unsuccessful. I recommend accepting that risk for the sole production run, given the owner’s quality priority, available budget and absence of a calibrated early-failure predictor. **I cannot establish that the present screen will fail, or that completing training will produce a good model.**

This is a recommendation for owner adoption. The running policy has not been changed.

**Q1 — The comparison is unsuitable for establishing eventual inferiority; it remains a coherent spending cutoff.**

The relevant arithmetic is:

| At N40 | Value |
|---|---:|
| Completed presentations | 10.08 million |
| Nominal epochs | 9.293 |
| Fraction of planned updates | 33.33% |
| LR | 0.000384953 |
| LR relative to peak | 76.99% |
| LR relative to terminal | 38.50× |
| Updates remaining | 80,000 |

Phase 1 completed **18.230 epochs**. Matching that exposure here requires **78,469 updates**, although matching epochs would still not match the corpus or LR trajectory.

Two qualifications matter:

- The *entire* anneal is not ahead: **23.48% of the peak-to-terminal LR reduction has occurred**. Most remains.
- Phase 1 is a completed, fully annealed reference. That does not prove mathematical convergence.

**There is no defensible “typical cosine annealing gain” in Bengali WER percentage points that I can assign to this run.** Budgeted-training research supports completing an appropriately scheduled decay and shows that schedule truncation changes performance. Its vision experiments do not calibrate a FastConformer-CTC WER allowance at one-third horizon. [Li, Yumer and Ramanan](https://arxiv.org/pdf/1905.04753)

Your strongest local illustration is Phase 1 itself: its reported internal running minimum at **13k—32.5% of its horizon—was 22.6**, versus **20.98** for the final model: a **1.62 pp difference**. That is one historical observation, with the supplied monitor/comparability limitations; it is neither a typical gain nor a threshold calibration.

**Even a typical remaining gain above 1 pp would not establish a near-certain kill.** A hypothetical N40 at D=21.0 that finishes at D=18.0 gains 3 pp and passes the present screen. Failure probability also depends on N’s unknown endpoint.

Council L’s asymmetry is partly explainable: N60 demanded **2.5 pp better S and 1.998 pp better D** than N40. Keeping a loose early screen while deleting a stricter later one is not logically contradictory. But neither cutoff has a measured false-rejection rate.

**Ruling:** the screen is not numerically invalid as a spending policy. It is **uncalibrated for deciding whether this sole cold start deserves completion**. Council N’s retained cutoff valued saving approximately 95 hours over an unknown probability of rejecting a useful run. I would reverse that policy preference; I would not describe the screen as a proven near-certain kill.

**Q2 — No defensible N40 point forecast or calibrated prediction interval exists. Neither alignment is established.**

The supplied arithmetic confirms **46.7−40.3=6.4 pp on the reported internal metric**. It does not establish that this difference persists for another 38,000 updates.

The two projections make different unsupported assumptions:

- **Per-update plus 6.4:** assumes an additive early deficit persists through different learning dynamics and LR schedules.
- **Per-epoch:** assumes doubling the corpus primarily stretches the learning clock, despite changing its composition and training objective.

Both runs present **504,000 recordings by update 2,000**. Corpus size alone does not halve the number of learning examples consumed. Under nominal proportional sampling, however, only about **49.9%** of N’s presentations come from retained O; the other half comes from MaCro. Matching updates therefore does not match exposure to Phase 1’s source mixture.

The proposed epoch model also fails to reproduce the existing observation: N2k maps to P1 update **1,019.5**, where linear interpolation of the supplied history gives **56.28%**, versus N’s **46.7%**. That does not disprove future epoch-like behaviour, but shows this is not an already fitting model.

The frozen conversions are unsupported too:

- N2k internal minus frozen: **1.10 pp**.
- P1 final internal minus frozen: **0.63 pp**.

Neither is a universal correction. The historical series contains running minima, and its **20.8 at 19k is below the final 20.98**, reinforcing that it is not a literal-checkpoint trajectory. Its supplied dense history also ends before the **20,390-update** epoch-aligned target.

Consequently, **“19.7–26.3% frozen” is not a prediction interval**. I would report N40’s point estimate, interval and pass probability as **not estimable from the available evidence**. Bootstrapping existing hypotheses measures evaluation uncertainty conditional on those checkpoints; it cannot supply the missing training-trajectory uncertainty.

The deciding evidence would be comparable completed trajectories validating an extrapolation method. Upcoming validations can test whether these illustrative shapes remain plausible; they cannot establish either alignment as a law.

**Q3 — None of 8k, 12k, 16k or 24k guarantees a trustworthy passing-trajectory forecast. Use 12k for the first useful shape comparison, checked again at 16k.**

Here is the exact diagnostic available without another GPU decode. Let:

- \(F(t)\) be linear interpolation of the supplied P1 internal running-minimum series;
- \(U(u)=F(u)+6.4\);
- \(E(u)=F(0.509757573u)\).

For consistency with that historical monitor, compare these with N’s **running minimum of the reported internal D metric**, while continuing to report literal-checkpoint frozen values separately.

| N update | Epoch illustration \(E\) | Update-plus-gap illustration \(U\) | Equal-distance boundary |
|---:|---:|---:|---:|
| 8,000 | 32.542% | 32.300% | 32.421% |
| 12,000 | 27.436% | 29.300% | **28.368%** |
| 16,000 | 25.838% | 27.900% | **26.869%** |

At 8k the illustrations differ by only **0.242 pp**. That checkpoint cannot usefully distinguish them.

At 12k, below **28.368%** is closer to the epoch illustration; above it is closer to the update-plus-gap illustration. At 16k the corresponding boundary is **26.869%**.

**These are descriptive boundaries, not “pass” and “fail” thresholds, and have no calibrated confidence level.** Being far from both illustrations is also possible. Before using them, verify evaluated row IDs and historical monitor semantics from existing artifacts; the contract explicitly warns about internal evaluator filters.

For operational monitoring, the more useful exact quantity at every validation is:

\[
m_u=\max\{S_u-22.0296,\ D_u-21.348\}.
\]

Report it alongside empty count and each eligibility guard. At 2k:

- S margin: **+25.1266 pp**;
- D margin: **+24.252 pp**;
- empties: approximately **one recording**.

A negative margin means that checkpoint meets both numerical screen limits. It does not guarantee that the literal 40k checkpoint will, or establish held-domain generalisation.

**Q4 — Ranked options and my exact amendment.**

Times below are **production update-time estimates from zero at 820–870 updates/hour**, excluding validation overhead, prior work and final evaluation.

| Rank | Option | Exact proposed operation | Training expenditure before performance rejection |
|---:|---|---|---:|
| **1** | **(d), chosen** | 40k diagnostic; sole qualification at **120k**, using the selected eligible checkpoint and existing N40 numerical limits | **137.9–146.3 h** |
| 2 | (e) | Illustrative gross-underperformance floor at **40k: D≤40.000%, empties≤1%—at most 53 recordings**; no S threshold there. Final qualification at 120k as in (d) | **46.0–48.8 h** if floor fails; otherwise up to **137.9–146.3 h** |
| 3 | (c) | 40k diagnostic; move unchanged screen to literal **100k: S≤22.0296, D≤21.348, empties≤0.1%**. Failure terminal; pass continues to H | **114.9–122.0 h** |
| 4 | (a) | Literal **40k**, current thresholds; failure terminal, no fallback | **46.0–48.8 h** |
| 5 | (b) | Claimed scientifically re-derived 40k allowance | **Not presently specifiable** |

For (b), **I cannot honestly provide a derived replacement number**. Adding 1.62 pp from Phase 1, 6.4 pp from the first comparison, or a vision-paper accuracy difference would not calibrate N40.

The **40%** floor in (e) is an explicit, rejected policy illustration—not a measured boundary between learning and hopelessness. A model above it may have learned substantial Bengali. It offers protection against gross persistent underperformance, but introduces another arbitrary irreversible cutoff.

At 100k, option (c) saves only **23.0–24.4 hours** versus completion while still risking rejection before the remaining decay. At 80k the correct LR is **13.63× terminal**, not 12.6×.

**My choice is the endpoint version of (d). It removes early performance-based spending rejection; it does not disguise that cost.**

Exact proposed amendment:

> **NVIDIA-only performance-policy amendment.** At literal update 40,000, record S, D, D empty count and whether the former N40 limits are satisfied: S≤22.0296, D≤21.348 and empties≤0.1%. This is diagnostic. A miss neither stops training nor excludes checkpoints. Record the milestone durably and replay any owed report after interruption. No N40 PASS is required for subsequent resume.
>
> The 60,000-update milestone remains diagnostic with its existing thresholds. H remains 120,000, with the existing cosine schedule and no plateau stop.
>
> At completion of update 120,000 and its validation, apply the existing checkpoint-eligibility and selection rules: find the minimum unrounded eligible S and select the earliest checkpoint within 0.05 pp of that minimum, without chained ties. Apply final qualification to that selected checkpoint: S≤22.0296, D≤21.348 and D empties≤0.1%, together with all existing P1-based eligibility guards. Retain the existing 1e-9 pp representation tolerance.
>
> No eligible checkpoint, or failure of final qualification, produces a **terminal ending**, with no production promotion, held decoding or headline. Do not substitute another checkpoint to pass qualification.
>
> Successful qualification produces a **normal ending** followed by the existing frozen-selection and one-shot evaluation procedure. Numerical, integrity, resource and budget rules remain binding. The aggregate ceiling remains 350 GPU-hours. No fallback, LR change, restart, extension or additional production procedure is authorized.

This explicitly changes qualification from a **literal interim checkpoint** to the **already prescribed final selected checkpoint**. It preserves the selector rather than introducing threshold-based checkpoint shopping.

Before adoption is implemented, version the policy, authorize the existing H120k state’s migration to it without changing training state, and test milestone replay, resume and final-ending semantics.

The cost in the two requested cases is:

- **Run truly unsuccessful:** up to **137.9–146.3 production training hours** before final performance rejection—**92.0–97.6 hours more** than an N40 rejection.
- **Run ultimately useful but N40 would reject it:** the same **92.0–97.6 hours** buys the remaining registered trajectory and the opportunity to qualify. That opportunity is not a promised improvement.

**Q5 — A legitimate prospective amendment, informed by observed results. It is not an untouched preregistration.**

The change would operate before its governed 40k observation exists. However, the early result plausibly influenced the concern, and early and later performance are related. Therefore “prospective” does not mean “independent of outcomes.”

Exact disclosure:

> “The original policy required terminal qualification at 40,000 updates. After observing the replacement run’s first 2,000-update validation—D 45.6% on the frozen surface, 46.7% internally and S 47.1562—and its reported 6.4-point internal difference from Phase 1, the owner requested reconsideration. Before any 40,000-update result was observed, the owner adopted a change making N40 diagnostic and moving qualification to completed-horizon selection at 120,000. This was a data-informed prospective amendment, not part of the original preregistration. The original N40 outcome will be reported, and additional spending and subsequent results will be attributed to the amended policy.”

Record the actual adoption time and every additional result seen before adoption. Do not wait for N40 and then describe the amendment as prospective.

**Q6 — If the original rule remains and fails.**

Exact reporting wording:

> “At its literal 40,000-update checkpoint, Phase 4 failed the registered continuation screen: [insert measured S, D, empty count and failed conditions]. Training therefore terminated under that spending policy. This establishes failure of the registered interim requirements. It does not establish that the model could not qualify after completing its 120,000-update schedule, that NVIDIA initialization is inferior, or that the corpus is unsuitable. The remaining 80,000-update trajectory and its held-domain performance were not observed. No production selection, held evaluation or headline was authorized.”

The owner may accept termination without further registration. **Any further training or held decoding requires a new authorization and registration**, for example:

- Resume the preserved N40 state under a separately registered continuation study, preserving the original failure record.
- Register a different training procedure as a new study.
- Register a diagnostic evaluation of specified archived checkpoints, with its evidence role and protected-set consequences fixed before decoding.

None permits deleting the failure, fabricating a PASS, automatically activating Phase 1, or presenting later success as success under the original policy.

**Q7 — Paragraph for the owner.**

Yes, its reported internal WER is currently worse: **46.7% versus 40.3%, a 6.4-point difference**. That is worth watching, but doubling the corpus neither makes this deficit inevitable nor tells us it will persist. Both runs have consumed 504,000 presentations; this run allocates roughly half to MaCro and has a much longer schedule. Its one empty output, 0.985 hypothesis/reference length ratio and non-dominant deletions argue against blank collapse, not for eventual quality. I recommend continuing and making 40k diagnostic, because ending our only production run there risks rejecting it before most of its scheduled decay. This accepts roughly **95 extra training hours** if it ultimately fails. I would become more concerned if frozen D and conversational performance stopped improving across successive validations while gains concentrated only on M; a verified implementation defect would require immediate correction. The present result establishes neither eventual failure nor eventual superiority.

**Q8 — The main failure mode is mistaking intermediate progress for evidence about completed quality.**

With the present rule, that means killing a useful unfinished model. With my amendment, it means assuming annealing will rescue persistent domain mismatch and spending the full budget.

Preserve two exact numbers:

\[
m_{40}=\max(S_{40}-22.0296,\ D_{40}-21.348)
\]

and, after completion,

\[
m_*=\max(S_{\mathrm{selected}}-22.0296,\ D_{\mathrm{selected}}-21.348).
\]

Check empties and eligibility separately.

- **Original N40 fails, but final qualification passes:** the original policy would have rejected a run that subsequently produced a qualifying model. Held evaluation must still determine how broadly useful it is.
- **Final qualification fails or no eligible checkpoint exists:** the extra spending did not produce a model meeting the amended requirements.

**Cheapest warning check:** at each existing validation report D, conversational WER, all guard violations and the weighted contributions  
\(0.60\Delta D,\ 0.20\Delta M,\ 0.10\Delta R,\ 0.10\Delta K\). An improving S accompanied by flat or worsening D/conversation exposes the specific risk that more training is improving the wrong domains.

That costs **zero additional GPU decoding**. It supports diagnosis and honest reporting; it must not quietly become another unregistered stopping rule.
