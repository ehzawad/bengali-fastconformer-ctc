**I pick D: recover and fully audit the subset now, then run one explicitly amended admission experiment that tests its incremental value over O+MaCro. Restart NVIDIA production only if that experiment passes. Preserve the current run so a failure costs one pilot, rather than the production trajectory.**

The owner is right that whole-bucket exclusion does not establish that every row is unusable. The orchestrator is wrong to call these **298.5 hours “ADMISSIBLE,”** wrong to equate the 910 transcript matches with the audio-duplicate gate, and wrong to suggest that beating O would establish SPRING’s contribution.

**Q1. The decision and its cost**

At 871 updates/hour, these are **additional training GPU-hours from update 9,100**, excluding audit inference, validation overhead and final evaluation:

| Option | Additional training GPU-h | What it buys |
|---|---:|---|
| A: finish current run | **127.3** | Completion of the admitted corpus’s experiment; no SPRING evidence |
| B as written: 8k pilot, then restart at H=120k if it passes | **136.5 if rejected; 147.0 if admitted** | Evidence against O, which cannot distinguish SPRING’s contribution from MaCro’s |
| C: immediately restart with SPRING | **137.8 at H=120k; 153.8 at H=134k** | An expanded production run with no measured incremental admission evidence |
| **D: corrected pilot; restart at exposure-scaled H≈134k only on a pass** | **136.5 if rejected; 163.0 if admitted** | A reversible admission decision testing the actual proposed addition |

D costs **9.2 extra training hours if the pilot fails**, or **35.7 extra if it passes**, relative to A. The latter comprises approximately 9.2 hours of pilot, 10.4 hours to repeat the abandoned production prefix, and 16.1 hours for the longer horizon.

Those costs make this worth investigating. They do not make admission automatic. The 10.4 hours already spent establish neither that we should continue nor that we should restart.

My proposed sequence is:

1. Register the amendment and deliberately checkpoint and pause production; preserve its complete resumable state.
2. Finish the candidate audits below. Audit failure means resume the existing run.
3. Freeze exactly one expanded mixture, **T = O+MaCro+SPRING-subset**, retaining the existing O+MaCro rows and targets under the registered precedence rules.
4. Run T’s admission pilot. Failure or inconclusiveness means resume the existing NVIDIA run. A pass authorizes a new NVIDIA production run from verified update zero.

**The cheap pilot needs an explicit authorization exception.** To compare with the existing 8k controls, it must use **init A and their original 150k schedule prefix**, from update zero. Section 8t currently prohibits further A-initialized training. Section 8o also restricted the SPRING amendment to a subsequent registration. Neither restriction disappears because the owner expressed interest in recovering rows.

I recommend authorizing this one additional **data-admission experiment**, while retaining NVIDIA-only production. An 8k NVIDIA run compared against A-initialized controls would be an invalid substitute. The A-based result would remain conditional evidence about admission, not proof of SPRING’s benefit after completed NVIDIA training.

Let **B** denote the frozen O+MaCro 8k arm and **O** the frozen reference control. Require **all** of the following at T’s literal update 8,000:

| Requirement | Proposed decision rule |
|---|---|
| Incremental composite benefit | **S(T) ≤ S(B) − 0.30**, approximately **20.231** |
| Existing control requirement | S(T) ≤ S(O) − 0.30, approximately 20.774 |
| Existing union ceiling | S(T) ≤ 20.280; the incremental requirement above is tighter |
| Statistical evidence | Paired sentence-clustered 95% CI upper bound <0 for ΔS against **both B and O**, plus the simultaneous upper quantile <0 |
| D protection | D(T) ≤ D(B)+0.30 **and** D(T) ≤ D(O)+0.30 |
| Source protection | Every D source ≤ its corresponding B+1.0 and O+1.0 limits; also satisfy the existing P1 eligibility guards |
| **Conversational benefit** | **Conv(T) ≤ Conv(B)−0.50**, approximately **21.392%**, with its simultaneous paired upper bound below zero |
| Existing conversational protection | Also satisfy O+0.50 and P1+0.50 = **22.075%** |
| Empty outputs | ≤0.1% on D: **at most five recordings** |

Use authoritative unrounded values, the existing coupled sentence-cluster bootstrap, 10,000 draws and seed 20260905. Charge SPRING as the **fifth candidate slot**, giving **q=1−0.025/5=0.995** for the simultaneous bounds. Preserve historical m=4 verdicts and identify any recalculations for the enlarged family explicitly.

The **−0.50 pp conversational requirement is my proposed prospective tightening**, not an existing rule or a measured optimum. It requires approximately **34 fewer net edits over 6,628 reference words**, plus statistical support. We are reopening production specifically for conversational benefit; merely consuming the old allowance for conversational harm is insufficient justification.

The current NVIDIA conversational WER, **29.39%**, is not the pilot comparator. Comparing T’s warm-start 8k result against that cold-start result would manufacture an apparent benefit.

**Q2. The post-hoc problem**

**This can be a legitimate adaptive amendment. Disclosure alone is insufficient.**

The original gate answered whether the **whole incoming source** met a uniform-script policy. Its failure remains valid. A newly authorized, deterministic subset answers a different question. Whole-row exclusion preserves the admitted reference’s words; deleting Latin words inside a retained utterance would corrupt its target.

SPRING’s publishers describe manually transcribed speech, supporting its candidacy, but that does not certify this particular subset’s labels. [SPRING-INX paper](https://arxiv.org/abs/2310.14654)

I would require this wording:

> After observing SPRING-INX script rates and Phase-4 development results, the owner authorized one Bengali-script subset of the fixed R1/R2 training inventories. The original whole-source G0.9(a) failure remains recorded. For this subset only, the whole-source script-loss ceiling is waived. Apply the frozen surface and existing character predicate; reject each offending row in full, retaining permitted digits. Do not delete spoken words, transliterate, rank rows by recognition accuracy, or revise the predicate after pilot results. Report exclusions against the complete incoming inventories and report all other gate denominators explicitly. This is an orthographic content filter that disproportionately excludes speech transcribed with Latin-script words; it does not establish label correctness or fully monolingual speech. Admission evidence uses previously inspected development panels and is conditional, adaptive development evidence—not independent confirmation of generalisation.

That must accompany **actual authorization, completed audits, a frozen candidate, fixed admission rules and no subsequent subset search**. Increasing multiplicity correction does not undo development-data reuse.

Had the filter been specified before measurement, its construction would have stronger prospective status. It would still need all other audits and admission evidence. Writing it early would not make its labels clean or its transfer beneficial.

**Q3. The 910 overlapping rows**

**The packet applies the wrong gate.**

The stated method finds **equal normalized transcripts**. Those are not evidence of byte-identical audio. G0.9(b) concerns incoming audio duplicates and audio defects; the protected-text rules concern sentence overlap. Therefore:

- **910 / 240,941 = 0.377686% is a measured transcript-overlap rate.**
- It does **not** establish a G0.9(b) PASS.
- All 910 overlapping training rows must be excluded, together with any additional matches found by the registered text keys, fuzzy audit and audio-component rules.
- Retained training overlap must be **zero unresolved matches**. The 0.5% bucket ceiling is not permission to retain contamination.

Shared sentences alone do not condemn unrelated recordings or prove copied audio. Conversely, an actual incoming audio-duplicate/defect rate that fails G0.9(b) still excludes the bucket unless separately amended. Passing one gate never compensates for failing another; U’s conversational rejection illustrates that conjunction.

There are additional holes in the claimed audit:

- The listed “FULL protected set” omits **FLEURS test and Vaani validation/test**. SPRING’s official non-training splits also require protection if its training data are reconsidered. Their evaluation role must be specified separately.
- The four reported disposition counts total **240,939**, leaving **two rows unexplained**.
- The registered SPRING training inventory totals **647.71 h**, versus **625.6 h** here: reconcile **22.11 h**.
- “Dropped digit or other script” needs a reason breakdown: **ASCII and Bengali digits are permitted**.

Complete source-identity checks, exact PCM hashing, calibrated fingerprint checks, protected-text matching, cross-source deduplication, caps, audio/CTC/tokenizer checks and the amended machine-consensus audit before calling anything admitted. Fix the loader to fail on missing or unreadable required panels and verify expected panel counts and hashes.

**298.5 h is presently an upper bound on recoverable candidate hours.**

**Q4. The horizon**

**If SPRING passes, preserve the exposure rationale by scaling H with the final retained row count.**

For the provisional inventory:

\[
H_{\mathrm{new}}
=120{,}000\frac{1{,}207{,}856}{1{,}084{,}696}
=\mathbf{133{,}625.2}.
\]

Register now: **round that formula upward to the next 2,000-update validation boundary, using the final audited manifest**. At the current candidate count, that gives **H=134,000**.

This produces **27.957 nominal epochs**, close to today’s **27.879**. Keeping 120k would give **25.036 epochs**, reducing each existing row’s expected presentations by **10.20%**. That could be a reasonable alternative policy, but it would change the exposure argument used to choose 120k.

Do **not** scale by the **17.6% increase in audio hours**: with the registered example-based batch, the relevant row increase is **11.354%**.

A 134k horizon costs approximately **16.1 more training GPU-hours than 120k** at the current throughput. Re-profile the expanded corpus; its different duration distribution could reduce throughput. Keep 40k/60k diagnostic and apply final qualification after the new horizon. Move the final-10k improvement report to **124k→134k**.

This is exposure consistency, not proof that 134k is optimal. The new manifest and schedule require a new run identity and fresh NVIDIA start. The **350 GPU-hour ceiling remains binding**; verify the complete ledger and remaining overhead before authorizing launch.

**Q5. What should be registered now if recovery does not lead to a restart?**

If the audits or pilot fail, resume the current run and carry forward a concrete registration package:

- Fixed R1/R2 revisions, complete inventories, original references, licenses and the exact whole-row predicate.
- Explicit parent/subset denominators, exclusion ledger, final unique rows/hours, speaker/session/domain distribution and retained-versus-rejected duration distributions.
- All protected-panel identities, complete leakage coverage, deduplication precedence and transcript caps.
- Quality-audit sampling, recognizer provenance limitations and gate outcomes.
- The single candidate mixture, comparator identities, statistical rules, conversational requirement, horizon formula, spending allowance and reporting plan.
- A permanent record of this pilot’s result, including an inconclusive result; no repeated attempts until a favourable seed appears.

That settles **how the subset is constructed and tested**. It cannot pre-authorize admission regardless of the findings. A later NVIDIA-specific comparison would require its own prospective experiment, not reinterpretation of an A-based rejection.

**Q6. The likeliest failure and cheapest check**

The likeliest mistake is **assuming that “conversational source” means “improves conversation.”**

The filter removes much of the code-switched material and changes the source’s content distribution. The retained subset may favour particular speakers, monologues or transcription conventions. At fixed updates it also displaces **10.20% of the existing mixture’s presentations**. These mechanisms can worsen conversation despite adding 298.5 audio hours.

The number to watch is:

\[
\Delta_{\mathrm{conv}}=\mathrm{Conv}(T_8)-\mathrm{Conv}(B_8).
\]

**The cheapest useful check is its paired clustered evaluation within the proposed 8k pilot**, alongside incremental ΔS and source guards. Existing B predictions make the comparator evaluation essentially free. U already demonstrated the relevant failure mode: **ΔS=−1.202 pp while conversation worsened +0.829 pp**.

The opposite mistake is treating a failed short A-based pilot as proof that SPRING could never help completed NVIDIA training. It would establish failure of this admission policy, with longer-budget NVIDIA performance unknown.

I would spend the roughly **9.2 training hours to obtain that decision**, provided the candidate audits pass and the one-off pilot exception is explicitly adopted. I would not spend the full restart budget merely because the surviving transcripts use the permitted script.
