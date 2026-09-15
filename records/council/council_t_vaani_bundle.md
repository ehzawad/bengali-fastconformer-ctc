**I choose (ii): audit and pilot the actual proposed mixture, O+MaCro+SPRING+Vaani, once. Admit the bundle only if it demonstrates incremental benefit over O+MaCro. Do not reverse Vaani’s standalone verdict, and do not restart production merely because its labels appear clean.**

First, the orchestrator is right on both disputed corrections. The dispositions total **240,941**, including the two empty rows; my missing-two-rows objection was wrong. I also accept the supplied ingest reconciliation: **5,935 + 408 + 10 = 6,353 upstream exclusions** explain the distinction between published and ingested inventories. I withdraw the “unexplained 22.11 h” objection. Neither correction establishes admission.

**Vaani may enter a new bundle experiment; its previous experiment remains a failure to establish the required benefit.**

The distinction is:

- **New information:** recovering SPRING creates a different proposed mixture. Its interaction with Vaani is unmeasured. NVIDIA-only production also limits how far the earlier A-based result transfers. Neither fact is new evidence that Vaani helps.
- **An always-arguable rule:** an 8k composite-improvement requirement can reject useful spontaneous speech, and label quality differs from training utility. Those objections existed before Vaani’s result. They cannot retrospectively turn that result into a pass.
- **Wanting more data:** rebuilding reduces some practical inconvenience. It supplies no evidence of benefit.

“Not harmful” overstates the record. Vaani passed the point guards; its **95% ΔS interval [−0.253, +0.106]** and simultaneous upper **+0.154** did not establish non-harm or equivalence. **Conversational Δ=0.000 is an observed aggregate equality**, not proof of identical errors or future behaviour.

I recommend registering this explicit amendment:

> Vaani’s O+Vaani admission result remains NOT ADMITTED, INCONCLUSIVE. Following inspection of prior development results and the SPRING recovery proposal, the owner authorizes one additional fixed-mixture admission experiment: O+MaCro+SPRING-subset+Vaani. This is an exception to the previous single-union restriction and permits exactly one further A-initialized admission pilot; production remains NVIDIA-only. Both additions must satisfy their applicable data audits. Admission applies only to the tested bundle and does not establish either addition’s individual benefit. A failure or inconclusive result authorizes no alternative subset, mixture, seed or longer pilot. The preserved production run resumes unless an audit independently invalidates its premises.

This is legitimate **adaptive development**, with its history intact. Increasing the multiplicity correction does not make reused development panels independent confirmation. [Dwork et al., *Generalization in Adaptive Data Analysis and Holdout Reuse*](https://arxiv.org/abs/1506.02629)

The actionable sentence for the owner is: **Authorize one test of both additions together; their apparent cleanliness does not justify admitting them without that test.**

**The pilot must test the bundle against the corpus production already uses.**

Define:

- **B:** the frozen **O+MaCro, init-A, literal-8,000-update** arm, S≈20.531.
- **O:** the frozen **O_ref, literal-8,000-update** control, S≈21.0735.
- **T2:** exactly one audited, frozen **B+SPRING-subset+Vaani** manifest.

T2 starts from **init A at update zero**, with the original admission recipe: **8,000 updates, effective batch 252, 1,000-update warmup, peak 2e-4, original H=150,000 cosine prefix**. Its schedule must not use the proposed production horizon. Preserve the registered tokenizer, transformations, augmentation, seed and validation cadence.

Require **every** row below at literal update 8,000:

| Requirement | Admission criterion |
|---|---|
| Incremental composite benefit | **S(T2) ≤ S(B)−0.30 ≈20.231** |
| Original control requirement | S(T2) ≤ S(O)−0.30 ≈20.7735 |
| Previously specified union ceiling | S(T2) ≤ **20.280**; the B-relative requirement is tighter |
| Composite uncertainty | For **T2−B and T2−O**, both the paired 95% CI upper bound and the required simultaneous upper quantile are **<0** |
| D protection | D(T2) ≤ min[D(B)+0.30, D(O)+0.30] |
| D-source protection | For every registered source j: WERⱼ(T2) ≤ min[WERⱼ(B)+1.0, WERⱼ(O)+1.0, WERⱼ(P1)+1.0] |
| **Conversational benefit** | **Conv(T2) ≤ Conv(B)−0.50 ≈21.392%**, with paired 95% and simultaneous upper bounds for T2−B **<0** |
| Other conversational guards | Conv(T2) ≤ Conv(O)+0.50 and **≤22.075%** |
| D empty outputs | **At most 5 of 5,385** |

The conversational point requirement means **at least 34 fewer net edits over 6,628 words**. Its confidence requirement establishes evidence of improvement; it does **not** establish that the entire confidence interval exceeds a 0.50 pp benefit. This remains my prospective policy tightening, not a measured optimum.

Use integer edit/reference counts, unrounded decisions, the existing **10,000 paired sentence-cluster bootstrap draws, seed 20260905**, and coupled weights across panels—including the conversational subset of D. Keep **S=.60D+.20M+.10R+.10K**. Existing pilot abort rules apply.

Before training, complete SPRING’s audits and the combined leakage/deduplication/cap audit, including the acknowledged Vaani protected-split fingerprint gap. “Vaani passed every gate” is too absolute given that documented coverage deviation. Preserve **every B row and target exactly**; if new protection requirements force changes to B, its historical checkpoint is no longer a matched corpus comparator and this design needs revision.

No official held split is decoded for admission. The current cold-start trajectory is not an 8k pilot comparator. A pass supports **the bundle under this admission procedure**, with individual contributions unresolved.

**For multiplicity, I require m=6 and q=0.9958333333—the 99.583333rd percentile.**

My accounting follows this phase’s conservative announced-slot convention:

- Four historical slots remain, including the unused CV17 slot.
- Council S’s announced SPRING-only candidate retains slot five.
- The changed **SPRING+Vaani bundle** receives slot six.

Vaani already occupies a historical slot, but that does not prepay every subsequent attempt to admit it in a different mixture. T1 need not run; its announced slot remains reserved, consistently with the CV17 precedent. This is an explicit conservative accounting policy, not a claim that there are six distinct datasets.

The required statistical endpoints form **one conjunctive admission decision**: all must pass. They are not separate opportunities to choose a favourable endpoint.

Preserve historical m=4 records and their original verdicts. Recompute any historical statistical approval relied upon by the expanded amendment at **m=6**, publishing separately identified results; do not invent those quantiles from the available summary. Vaani cannot become a pass merely by increasing q, and U’s conversational failure remains. The original production authorization remains an m=4 historical decision.

**The rounded horizon is correct, but the exact arithmetic and mixture shares need correction.**

Using the supplied provisional rows:

\[
H_{\rm scaled}
=120{,}000\frac{1{,}293{,}803}{1{,}084{,}696}
=\mathbf{143{,}133.523},
\]

not 143,140. Rounding upward to a validation boundary gives **144,000**.

| Corpus and horizon | Nominal epochs |
|---|---:|
| Current B, 120k | **27.879** |
| Expanded T2, 120k | **23.373** |
| Expanded T2, 144k | **28.048** |

Retain the registered formula using the **final audited row count**, rather than fixing 144k regardless of exclusions. Keep 40k/60k diagnostic, completed-horizon qualification, and no plateau. For H=144k, move the final-10k report to **134k→144k**.

At **871 updates/hour**, and pausing the current run at approximately 9,300:

| Path from that pause | Additional training GPU-h |
|---|---:|
| Finish current production | **127.10** |
| T2 pilot fails; resume current production | **136.28** |
| T2 pilot passes; fresh NVIDIA production to 144k | **174.51** |

A passing expansion therefore costs **47.42 additional training GPU-hours** versus continuing: **9.18 pilot + 10.68 repeated prefix + 27.55 longer horizon**. These exclude audit inference, validation and final evaluation.

**The 350 GPU-hour ceiling remains binding; the supplied record does not certify that the revised whole-phase ledger fits.** Let P be all spending already incurred, including both abandoned prefixes, and E the remaining audit, qualification, validation, evaluation and reserve allowance. At the assumed throughput, the pass path requires:

\[
P+E+174.51\le350,
\qquad\text{hence}\qquad
P+E\le\mathbf{175.49\ GPU\!-\!h}.
\]

Re-profile T2 and close that ledger before restarting. Count the already-spent prefix once in P.

The packet’s mixture percentages are **audio-hour shares**, not example-sampling shares. Actual row shares are:

| Source | Current B | Proposed T2 |
|---|---:|---:|
| MaCro | **50.10%** | **42.00%** |
| SPRING | 0% | **9.52%** |
| Vaani | 0% | **6.64%** |

At fixed updates, each existing row receives **16.16% fewer expected presentations**. The claimed conversational-leaning share of 47.0% is neither a measured conversation-label fraction nor the training presentation fraction.

**To the owner:** You are right that useful human-labelled data need not be perfect, that the old corpus contains defects, and that Vaani’s result did not establish bad labels or broad harm. You are wrong to conclude that Vaani has no latent issues or that cleaner labels necessarily improve the resulting model. Sampling, domain coverage, transcription conventions and interactions matter; U already improved S while hurting conversation. More clean speech can improve coverage, but it also changes what the model practises. Your argument supports spending roughly **9.2 training hours testing the actual expanded mixture**. It does not establish the benefit that would justify the production restart.

**The strongest adversarial failure is a benefit that fails to transfer from the short warm-start pilot to completed cold-start production.**

There are two distinct possibilities:

1. **SPRING’s benefit conceals Vaani’s negative marginal effect.** For a hypothetical example, T1 could improve S by 0.70 pp and conversation by 1.00 pp, while adding Vaani gives back 0.20 and 0.30 pp respectively. T2 would still improve by **0.50/0.70 pp** and could pass. The revealing numbers would be **S(T2)−S(T1)=+0.20** and **Conv(T2)−Conv(T1)=+0.30**. Our chosen single pilot does not measure them. A second 8k arm could reveal that particular A-based interaction, at another **9.18 training GPU-hours**; it still would not establish NVIDIA’s full-budget response.

2. **The mixture’s ranking changes with initialization, annealing or domain.** At 8k, T2 gives B’s rows approximately **1.690 million presentations**, versus **2.016 million** in B’s pilot. The conversation difference and source breakdowns can expose immediate dilution damage. But warm-start Bengali knowledge can mask learning requirements that matter for NVIDIA initialization, and effects can change during the longer anneal. **No 8k number is guaranteed to reveal that later reversal**, even when every pilot criterion passes.

Finally, if the current trajectory is abandoned, its completed outcome remains **unobserved**. Final held reports can reveal poor transfer, but they cannot establish exactly how much better the abandoned run would have become. This pilot buys a disciplined admission decision; it cannot buy that counterfactual.
