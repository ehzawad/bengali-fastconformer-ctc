#!/usr/bin/env python3
"""Council R: reconcile the 8w.7 closure. Two defects closed, two HELD against your own prescription."""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))

spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cq = (R / 'experiments/p4/council/council_q_floor_and_bug_class.md').read_text(encoding='utf-8')
recon = json.loads((R / 'experiments/p4/unrounded_recon.json').read_text())
ours = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "what_you_prescribed": ("In council Q Q5 you gave a required fix for each of nine defects. I ran a 13-agent workflow: "
    "one design per in-trainer defect (7, 1, 2), TWO adversarial reviewers each, one direct implementation of the "
    "isolated defect (9), one numeric reconstruction. ALL THREE in-trainer designs were REJECTED by their reviewers. "
    "I closed 9, measured 7 and did not reroute decisions, and HELD 1 and 2 unapplied."),
 "DEFECT_9_CLOSED": ("IDENTITY.json written before the first prediction binding selected-checkpoint path+sha256, decoder "
    "settings, scorer/surface identity and every panel manifest path+sha256+row count; every shard row stamped with the "
    "identity digest so your 700-rows-with-X-then-Y sequence refuses before decoding and appends nothing; the event "
    "minted O_CREAT|O_EXCL in experiments/p4/final_test_event.json OUTSIDE any output directory so a fresh --out-dir "
    "cannot mint a second authorization; the refusal text suggesting another directory deleted; flock at registry level. "
    "35 checks in experiments/p4/tests/test_final_test_identity.py, all pass."),
 "DEFECT_7_MEASURED_NOT_REROUTED": {
   "what_I_did": ("Recomputed all four validations exactly (integer edits / reference words, repo's own frozen_norm and "
                  "wer_counts) via tools/p4/p4_recompute_unrounded.py. Did NOT reroute decisions through exact accessors."),
   "result_artefact_keys": sorted(recon.keys()),
   "result_summary": {k: recon[k] for k in recon
                      if k in ("verdict", "summary", "decision_changes", "max_s_delta_pp",
                               "panels_outside_expected_rounding", "updates", "per_update")},
   "headline": ("largest S discrepancy 1.39e-04 pp against a live qualification margin of 5.662 pp (factor ~40,000); "
                "no eligibility guard, empty-rate check or 0.05 pp tie-band comparison would have decided differently"),
   "the_design_was_wrong": ("Its ONLY constructive demonstration of a decision flip was arithmetically false: it claimed "
                            "round(11591/55191,5)=0.21 giving exactly 1.000 pp and a PASS. True value 0.21002; exact delta "
                            "1.00161 pp and rounded delta 1.00200 pp, so BOTH paths FAIL the +1.0 pp source guard. No flip "
                            "exists at that denominator. Both reviewers found this independently."),
   "also_rejected": ("the design raised on a zero reference-word denominator inside the decision path, contradicting "
                     "p3_common.wer_counts' documented 'one bad row can never take down a 40k-update run', and introduced "
                     "a NameError into the hardened resume-path history reader."),
   "my_reasoning_for_holding": ("val_preds/ preserves every hypothesis, so ANY validation past or future can be re-scored "
                                "exactly at any time - the precision is never lost, only deferred. Against that, rerouting "
                                "every decision is a large multi-hunk edit to the two modules the live trainer runs.")},
 "DEFECTS_1_AND_2_HELD": {
   "1_resume_identity": ("design would treat run_uid/run_name as HARD fields, breaking the registered SPEC 5 pilot->main "
     "milestone handoff; bind the corpus through a (path,size,mtime_ns) cache that a `cp -p` satisfies, so it does not bind "
     "content; hard-abort when init_sha256 is None under --no-sha-check; provide NO override, so a spurious mismatch is an "
     "unrecoverable terminal ABORT after which run_supervised.sh refuses to start until a human intervenes; and add a 381 MB "
     "sequential read at every relaunch."),
   "2_policy_versioning": ("design would raise ValueError out of configure() for a missing key - uncaught out of main(), so "
     "the supervisor re-queues FOREVER at 11-minute intervals reloading the model each time, which is verbatim the failure "
     "check_resume_state's own docstring legislates against; and resolve a legacy bare policy name through a single-valued "
     "map that always points at the NEWEST version, so after any version bump the live run's checkpoint would silently adopt "
     "v2 with no migration record - precisely the defect being fixed.")},
 "THE_HAZARD_THE_REVIEWERS_FOUND": ("run_arm.sh execs the trainer FROM THE WORKING TREE at every start and the unit carries "
   "Restart=on-failure, RestartSec=10min, StartLimitIntervalSec=0. So ANY edit to tools/p4/p4_train.py or p4_common.py is "
   "armed to load involuntarily on the next crash with no owner in the loop, and a half-applied multi-hunk edit becomes a "
   "SILENT indefinite relaunch loop: death at import, no ABORT record written, idle GPU, looks like a hardware fault. "
   "FIXED: run_arm.sh now runs an import smoke immediately before exec and on failure writes a durable ABORT naming cause "
   "and remedy, exit 3. Operating rule adopted: those two modules are edited only when the run is deliberately stopped."),
 "run_state": ("N_O+macro_filtered_H120k at ~update 8,400 of 120,000, 871 upd/h, loss ~0.98. Validation at 8,000: "
               "WER 27.010 frozen / 28.452 internal, S 27.3228, conversational 29.39, 2 empties, m_u 5.662. "
               "Phase 1 at ITS update 8,000 was 32.699 internal, so we lead by 4.25 pp (was 9.85/5.36/6.08 at 2k/4k/6k). "
               "p4_train.py and p4_common.py are byte-identical to the code the live trainer loaded at 00:46; NO restart "
               "was performed for this work."),
 "validations": [{"u": v["update"], "S_pp": v["S_pp"], "D_frozen_pp": round(v["panels"]["D"]["wer_frozen"]*100, 3),
                  "empties": v["panels"]["D"]["empties"]} for v in ours],
 "owner_instruction": ("Owner, verbatim: 'reconcile with codex council and ultracode ultrathink and get it fix and then if "
                       "required restart the train and Only reconcile with codex council before the workflow if required it'. "
                       "He also holds: generalisation over benchmark, no sunk-cost fallacy, be direct."),
 "council_Q_verdict_verbatim": cq,
}
q = """
# QUESTIONS. Be adversarial. I departed from your prescription on three of four defects; say whether each departure was right.

Q1 DEFECT 7. You required "persist integer edit/reference counts and derive decision values without display rounding".
I persisted NOTHING and rerouted NOTHING - I measured instead, found the largest S error is 1.39e-04 pp against a
5.662 pp margin with no decision changed anywhere, found the design's only decision-flip demonstration was
arithmetically false, and left the code alone because val_preds/ makes exact rescoring available forever. Was that the
right call, or is there a decision later in this run (the 0.05 pp selection tie band at 120,000; the +1.0 pp source
guards near the qualification boundary) where 1e-4 pp could actually matter? Give the exact circumstance if so. If
holding was wrong, say what must be applied and by which update.

Q2 DEFECTS 1 AND 2. Both designs were rejected for introducing crash-loop or unrecoverable-abort risk on the sole
production run. Do you accept that holding them is correct, or is there a MINIMAL subset of each that is safe to apply
to a live-supervised run today? If a safe subset exists give it exactly. If not, name the point in the schedule at
which each must be applied and what makes that point safe.

Q3 THE IMPORT-SMOKE GUARD. Is it sufficient for the hazard, and does it introduce a new one? Note that a failed smoke
writes an ABORT, and run_supervised.sh's startup guard then refuses every future start until a human deletes it -
which is fail-closed but also means a transient import failure (a half-written file caught mid-edit, a full disk)
permanently stops the sole production run until someone notices. Rule on that trade, and give the exact guard you
would write instead if you disagree.

Q4 WHAT I MISSED. Your Q5 listed nine defects; I have now closed 5 (#3,#4,#5,#6,#8), closed #9, measured #7 and held
#1,#2. Re-examine the five I called "fixed" on 2026-09-09 00:55 against the actual code and say whether any is
incompletely fixed. Then: is there a TENTH defect of this class that neither of us has named, reachable between now
and update 120,000?

Q5 THE HONEST STATUS LINE. Give one paragraph the owner can read that states plainly what is now trustworthy about
this execution and what is not, without overclaiming. He is explicitly guarding against sunk-cost reasoning and wants
directness over reassurance.

Q6 ADVERSARIAL. The single most likely way this closure is wrong - either something I closed that is not actually
closed, or something I held that will bite before 120,000. Name the number or file that reveals it and the cheapest
check.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. You ruled as council Q and prescribed fixes for nine execution-integrity defects; your verdict is "
          "included verbatim. The orchestrator has now acted on them, departing from your prescription on three of four "
          "open items after a 13-agent adversarial workflow rejected all three in-trainer designs. Review the departures. "
          "Do not ratify. Be concrete, numeric, and say plainly where the orchestrator is wrong or where you were. "
          "Below: CONTRACT, FACTS, QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False, default=str) + "\n" + q)
d = Path('/tmp/council_p4r'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
