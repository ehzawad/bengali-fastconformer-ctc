#!/usr/bin/env python3
"""Council L: the owner dropped init A entirely - N is the ONLY production run; the A-vs-N comparison is deferred to next year."""
import json, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
ck = (R / 'experiments/p4/council/council_k_nfirst.md').read_text(encoding='utf-8')
def last(run): return json.loads(open(R / f'experiments/p4/runs/{run}/steps.jsonl').readlines()[-1])['update']
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_directive_verbatim": "\"you know what... I have changed my mind, I will not use phase 1 weight at all... we will do that experiment later, maybe next year\"",
 "orchestrator_reading": "Init A (Phase-1 weights, ehzawad/stt_bn_fastconformer) is NOT used as the initialisation of any production run in Phase 4. Init N (nvidia/stt_en_fastconformer_ctc_large + fresh Bengali head, Phase-1 recipe, H=150k) is THE ONLY full run. The A-vs-N comparison (SPEC 8q/8r/8s) is deferred indefinitely. The owner's earlier principle stands: generalisation over benchmark.",
 "what_is_mid_flight": {
   "pilot_U": f"update {last('pilot_U')}/8000 on the A6000, INIT A, resumed 06:49 after a co-tenant yield. Purpose: ADMIT the union corpus U = O u {{macro, madasr}} (a DATA decision: admission vs O_ref at m=4 plus S(U) <= 20.280) and be A's resume point. Under the new directive the resume point is moot; the corpus decision is not obviously moot - it is the only admission evidence for U. ~6 h remain (ETA ~12:50 UTC).",
   "corpus_admission_was_A_conditioned": "MaCro and madasr were admitted by 8k pilots from init A; the buckets were chosen by init A's errors. Councils J/K recorded this as a confound that would be DISCLOSED, not removed. With A gone, it is the only admission evidence that exists; an 8k N-init pilot cannot admit anything (a cold start at 8k is uninformative, council J).",
   "post_union_chain_armed": "run_post_union.sh (SPEC 8s): after pilot_U -> U admission + 0.20 rule -> pin A's 8k -> GPU smoke of init N (60 updates + resume round-trip) -> O_A6000 control completion 4,038->8,000 (~5 GPU-h, the 8s.4 cross-card qualification) -> agreement check -> launch N on the A6000. It has not acted yet; it can be replaced via rename before 12:50.",
   "control_completion_purpose": "council D/I/K: cross-card agreement so an A6000-trained arm can be compared with A5000 results. With ONE procedure on ONE card and no A run, there is nothing to compare across cards.",
   "cross_card_4k_warning": "at 4,000 the two controls agree on S (-0.070) and D (-0.020) but openslr53 differs by -0.546 vs the 0.30 per-source bound; the 8k check might fail; under 8s.4 that would idle N. Moot if the control completion is dropped.",
   "N_gates_as_registered_8q_8s": "40k screen S<=22.0296 D<=21.348 empties<=0.1% (spending screen); 60k qualification S<=19.5296 D<=19.35 + all P1 guards (TERMINAL for N); no plateau stop for N; run to 150k. These were designed for N as a COMPARISON arm with A as the fallback. With no A, an N failure at 60k = Phase 4 ends with NOTHING; N's LR at 60k is still 3.366e-4 (33.7x terminal) - a cold start may be 'not yet 1.5 pp better than Phase 1' at 60k while still improving fast.",
   "phase1_cold_start_reference": "on 984 h at 5e-4/2000/H=40k: internal dev WER 25.9% @8k, 22.9% @12k, 20.8% @19k, final 20.35 frozen (S 21.03) @40k with LR at terminal. N has 1,841 h and H=150k: at 40k its LR is 4.245e-4, not 1e-5.",
   "budget_status": "~90 GPU-h spent by pilot_U's end; N to 150k ~175-207; total ~265-300 without the control completion; the 350/525 ceilings (8r/8s) were sized for TWO procedures.",
   "cards": "A6000 hosts N (co-tenant 33 GB; we fit 10 GB). A5000 blocked (10,538 free < 12,000 floor; co-tenant 13.5 GB).",
 },
 "owner_principles": ["generalisation over benchmark (SPEC 0)", "council reconciliation before any decision question (SPEC 0)", "never touch a co-tenant process", "'take your time' / 'not six months'"],
}
q = """
# QUESTIONS (the owner has DIRECTED: no Phase-1 weights in production; answer with picks, numbers, exact SPEC wording; this round precedes any question back to him)

Q1 READING. Confirm: init N is the only production run; the A-vs-N comparison is deferred (not cancelled) to a future registration; the corpus decision machinery (which used init A as its instrument) stands. If the owner's words admit another reading ('do not even use Phase 1 to admit data'), state it and its consequence (there would be NO admitted corpus and no way to admit one at 8k from a cold start).

Q2 PILOT_U: FINISH, STOP, OR STOP-AND-USE-U-ANYWAY? It is init A, ~6 h from its 8k. (a) finish it - the registered U admission + 0.20 rule decide N's corpus (U 1,841 h if it passes, MaCro 1,692 h if not); N starts ~13:15; (b) stop it now, corpus = MaCro alone (admitted), N starts ~07:45, madasr's 149 h are lost; (c) stop it now, corpus = U WITHOUT its admission (unregistered; the 0.20 rule exists because adding data must not hurt - but that rule was measured on init A). Pick, with the exact SPEC consequence. Note the card would otherwise idle or run the (possibly moot) control completion.

Q3 DROP THE CONTROL COMPLETION? With one procedure on one card, is the 8s.4 cross-card qualification (O_A6000 4,038->8,000, ~5 GPU-h, plus the risk it FAILS on the per-source bound and idles N) still required? Rule, with the SPEC consequence for 8s.4 and for the disclosed 'cross-card agreement test not completed' caveat (which then stays forever).

Q4 N'S GATES AS THE SOLE RUN. Re-rule N's gates now that there is no fallback: keep the 40k spending screen? Keep the 60k qualification as TERMINAL (Phase 4 ends with nothing if a still-improving cold start is not yet 1.5 pp better than Phase 1 at 60k), or convert it to a REPORTED milestone and let (i) the P1 eligibility guards at checkpoint selection and (ii) the FLEURS success criterion be the judgment? Or set a LATER terminal gate (e.g. 100k, where N's LR is 5.8e-5)? Give exact update, thresholds, and ending class for each gate you keep. State what 'Phase 4 terminates' means with no fallback.

Q5 PLATEAU FOR N? Council I removed it for a comparison arm. As the sole production run, should N have a plateau/early-stop rule (saves GPU-h if it converges) or run to 150k regardless? If a rule: exact anchor update, patience, earliest stop, given N's LR curve (2.86e-4 at 70k, 5.8e-5 at 100k, 1.2e-5 at 140k).

Q6 SELECTION, HEADLINE, ONE-SHOTS. With one run: selection = lowest eligible S with the 0.05 tie band (4); one FLEURS decode on it (6); one-shots (holdout, Vaani official) on it. Confirm; and say what of 8r's paired-diagnostic machinery survives (nothing to pair) and what disclosure replaces it. Under the generalisation principle, is there any additional protection worth registering for a single-run selection (e.g. reporting the holdout/Vaani one-shots for the 2-3 nearest-S eligible checkpoints as diagnostics, never for selection)?

Q7 BUDGET AND CALENDAR. N alone: give the total (spent + N + eval) and a ceiling to register in place of 350/525; calendar to 150k and to the headline for each Q2 option.

Q8 WHAT THE RECORD MUST SAY. Exact wording for: the A-based admission of an N-trained corpus (confound, disclosed); 'Phase-1 weights were not used in the production model'; the deferred comparison; the cross-card caveat if the control is dropped.

Q9 EXACT SPEC AMENDMENT (8t) implementing the directive, superseding the relevant parts of 8q/8r/8s.

Q10 ADVERSARIAL. The single most likely way a cold-start-only Phase 4 produces a wrong conclusion or a wasted week; the number that reveals it; the cheapest check. Include the blank-collapse geometry Phase 1 documented (fresh head, ~6.3 frames/token at 8x) and how the 60-update smoke + first validations would show it.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR programme. The owner has just "
          "DIRECTED that Phase-1 weights are not used at all: the NVIDIA cold start is the only production run and the A-vs-N comparison is deferred "
          "to a future year. Councils A-K shaped the contract; council K's verdict (N-first, now superseded) is included verbatim for the parts that "
          "survive. Several things are mid-flight. Be concrete, numeric, adversarial, brief where the answer is 'yes'. Below: the CONTRACT, then "
          "COUNCIL K, then FACTS, then QUESTIONS.\n\n# CONTRACT (experiments/p4/SPEC.md)\n" + spec +
          "\n\n# COUNCIL K VERDICT (verbatim; N-first, now partly superseded)\n" + ck + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4l').mkdir(exist_ok=True); Path('/tmp/council_p4l/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
