**Council N: change H to 120,000; restart once from update zero; retain the terminal 40k screen and diagnostic 60k milestone; add no plateau stop.**

The annealing-tail argument was sound. It did **not** establish that the endpoint must remain 150k after the corpus and production scope changed. My recommendation is a training-design judgment, **not evidence that 120k will achieve lower WER**. I independently recomputed the arithmetic; I have not changed the running job.

**Q1 — Pick H = 120,000.**

First, three supplied LR values are wrong. Using the registered cosine,

\[
\eta(u)=10^{-5}+0.000245
\left[1+\cos\left(\pi\frac{u-2000}{H-2000}\right)\right],
\]

the comparison is:

| H | Nominal epochs | LR at 60k | Training GPU-h at 820–870 updates/h |
|---:|---:|---:|---:|
| 78,000 | 18.12 | **7.47476×10⁻⁵** | 89.7–95.1 |
| 100,000 | 23.23 | **1.85291×10⁻⁴** | 114.9–122.0 |
| **120,000** | **27.88** | **2.61522×10⁻⁴** | **137.9–146.3** |
| 150,000 | 34.85 | 3.36619×10⁻⁴ | 172.4–182.9 |

These are update-time estimates; add validation overhead if the measured rate excludes it. LR values use schedule coordinate \(u\); the recorded trainer applies `lr_at(update−1)`.

My reasons for 120k:

- **The strongest local empirical anchor is Phase 1’s completed cold start:** 18.23 epochs. Matching that exposure on this corpus gives **78,469 updates**. That establishes a useful starting point, not an optimum.
- **120k supplies 52.9% more presentations per retained row than that anchor.** It provides substantial allowance for the expanded mixture and its learning dynamics.
- **The original 27-epoch planning rationale maps to 116,217 updates.** Rounding to 120k is reasonable planning; neither 27 nor 27.88 has special scientific significance.
- **120k completes its own annealing tail.** It is not equivalent to stopping the existing 150k schedule at 120k. Research on budget-aware schedules supports this distinction, but does not identify the correct Bengali endpoint. [Li, Yumer and Ramanan](https://arxiv.org/abs/1905.04753)
- Moving from 120k to 150k buys another **6.97 epochs for 34.5–36.6 training GPU-hours**. There is no measured result here establishing the value of those additional epochs.

I would choose 120k before either run launched. The small sunk cost makes the change inexpensive; it is not the reason for choosing it.

**150k is not demonstrably overtraining.** Conversely, the packet’s claim that best-checkpoint selection means excessive training “cannot select an over-trained checkpoint” is too strong. Selection protects against later deterioration **measured by S and the guards**. It cannot guarantee held-domain generalisation or recover a better trajectory produced by a different LR schedule.

**Evidence that would change my pick:** a completed, comparable cold-start schedule study showing that 150k delivers a material improvement—say **≥0.20 pp eligible S**, with consistent D/conversational and held-domain evidence—would favour 150k. No such comparison exists. Phase 1’s internal curve cannot supply it.

**Q2 — Pick (a): no plateau stop.**

Shorten the planned schedule once, then finish its tail after N40 passes.

For a concrete rejected alternative under H=120k:

- Anchor at **100,000**, LR **4.39193×10⁻⁵**.
- Reset on **≥0.20 pp cumulative eligible improvement**.
- **Five** consecutive non-reset validations.
- Earliest stop **110,000**, LR **1.86319×10⁻⁵**.
- Maximum saving: **10,000 updates = 11.5–12.2 GPU-hours**.

That is defensible as a spending policy. I would **not** add it here. A cumulative gain of **0.19 pp** could fail to reset patience while substantially improving the selected model; it is almost four times the **0.05 pp** selection tie band. Low LR also does not prove that remaining training is useless.

The chosen design saves **34.5–36.6 hours through the horizon change**, while preserving every update of the new schedule’s tail. Adding plateau stopping would accept another quality risk for a comparatively small saving.

There remains early termination for **N40 failure and the existing numerical, integrity, resource and budget conditions**. “No plateau stop” does not mean “ignore failures.”

**Q3 — Keep the terminal screen at exactly 40,000. Keep its thresholds.**

Under H=120k:

| Requirement | Threshold |
|---|---:|
| S | **≤22.0296 pp** |
| D | **≤21.348%** |
| D empties | **≤0.1%: at most five of 5,385 recordings** |
| LR | **≈3.84953×10⁻⁴**, or 38.50× terminal |
| Exposure | **9.29 nominal epochs** |
| Training time to screen | **46.0–48.8 GPU-hours** |

This is **a near-baseline spending screen**, not an “is it learning Bengali at all?” test. A recogniser with 25% WER plainly learned Bengali and would nevertheless fail.

I retain 40k because it gives a substantial training interval before deciding whether to fund the remaining **80,000 updates, approximately 92.0–97.6 training GPU-hours**. Moving it to 30k increases premature-rejection risk; moving it to 50k spends another **11.5–12.2 hours** without evidence that the later threshold is better calibrated.

Do not numerically “re-derive” the thresholds from Phase 1’s incompatible internal WER curve. **These limits remain policy choices, not validated predictions of eventual failure.** A miss terminates Phase 4; its full-schedule potential remains unknown. Preserve the durable decision, owed-validation replay and resume invariant.

**Q4 — Keep 60k diagnostic only.**

At 60k, H=120k still has:

- LR **2.61522×10⁻⁴ = 26.15× terminal**.
- **13.94 nominal epochs** completed.
- Another **60,000 updates**, approximately **69.0–73.2 training GPU-hours**, remaining.

Report the existing thresholds unchanged:

**S≤19.5296; D≤19.35%; every D source≤P1+1.0 pp; conversational≤22.075%; empties≤0.1%.**

A miss neither stops the run nor disqualifies later checkpoints. The corrected LR arithmetic strengthens this ruling. **A shorter horizon does not turn these thresholds into a calibrated terminal qualification.**

The milestone remains useful: it shows where progress and regressions occur and supports engineering diagnosis. It does not authorize performance-driven LR, mixture or threshold changes.

**Q5 — Restart from zero; write off the actual approximately 2–3 GPU-hours.**

The required launch changes are:

| Setting | Required value |
|---|---|
| Run identity | New, distinct H120k production run |
| Policy/config identity | New version recording this amendment |
| Training target and cosine endpoint | **120,000** |
| Post-warmup cosine span | **118,000 updates** |
| Initialization | Verified archived NVIDIA-derived **update-zero** state |
| Optimizer, sampler and RNG | Fresh, restored to registered initial state |
| Resume from current trained checkpoint | **No** |
| Warmup | **2,000 updates, 1e-5→5e-4** |
| Terminal LR | **1e-5** |
| Plateau | **Disabled** |
| N40 / N60 | **Terminal / diagnostic**, respectively |

Use the original head seed and **verify exact initialized tensor hashes**. Restoring the verified update-zero artifact is stronger than assuming that the same seed alone guarantees identical initialization. Corpus, tokenizer, augmentation, batch, optimizer settings and selection remain fixed.

Archive the abandoned run and charge its cost once. Its checkpoints do not enter the replacement run’s selection pool. Replace the active launch/supervisor configuration; prevent the old supervisor from relaunching it. Normal completion and final validation must now occur at **120k**, with the existing eligibility requirements.

One technical correction: **at update 1,450, the run is still inside the common warmup.** Its applied LR, **0.000365005**, is identical for all four horizons. Restarting is therefore a protocol-integrity choice under the current resume restrictions—not a mathematical necessity because the prefix already differs. Given the low cost, I favour the clean restart over introducing a special migration path.

Net training-time saving after a 2–3-hour write-off: approximately **31.5–34.6 GPU-hours**. The aggregate ceiling stays **350 GPU-hours**.

**Q6 — Paragraph for the owner.**

> I recommend one change now: restart with a 120,000-update schedule. That gives 27.9 passes through the retained corpus, versus Phase 1’s proven 18.2, and completes the full learning-rate decline while saving roughly 35 training hours. The 40k point is a spending screen requiring performance near Phase 1; failing it ends the run, although it cannot prove that further training would fail. The 60k point reports progress and regressions without stopping, because learning is still underway at a relatively high learning rate. I would add no plateau stop: even a late one could discard useful final improvements for only about twelve hours saved. Protecting the annealing tail was the right decision; retaining exactly 150k after the corpus and comparison plan changed was insufficiently justified. This revision is a reasoned choice, not a promise of better WER.

**Q7 — Adversarial: 120k can anneal too soon.**

The principal failure mode is that the expanded mixture needs more substantial parameter updates before fine-tuning. At update **100k**:

- H120k LR: **4.39193×10⁻⁵**.
- H150k LR: **1.35515×10⁻⁴**, **3.09 times larger**.

The shorter schedule could settle into a worse solution. Best-S retention cannot recover a trajectory that was never trained.

**Cheapest warning check:** use the already-required validations, with zero additional GPU decoding. Report improvement in best eligible S over **110k→120k**, alongside D and conversational WER. A **≥0.20 pp eligible-S gain during those final 10k updates**, supported by D improvement without conversational deterioration, would undermine any confident claim that training had converged.

That warning would **not prove 150k would have won**, and it must not trigger an improvised extension or LR reset. No statistic from this single trajectory can establish the unobserved alternative’s performance. Report the limitation, finish the registered selection, and keep the held evaluations sealed until that selection is frozen.
