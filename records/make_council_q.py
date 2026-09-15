#!/usr/bin/env python3
"""Council Q: attack the §8w.2a learning floor and the restart-bug class.

Council P chose "40k diagnostic, next performance check at 120k". I verified in
code that init N has NO §5 pilot abort, so that leaves 92-98 GPU-h with no
performance stop at all. I adopted P's amendment WITH a floor it did not see.
This round exists to attack that floor, not to defend it.
"""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
from tools.p4 import p4_common as P

spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cp = Path('/tmp/council_p4p/out.log').read_text(encoding='utf-8')
ours = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "what_changed_since_council_P": (
   "1) Your amendment was ADOPTED at update 7,212, before any 40,000 observation, so it is prospective. "
   "2) I added ONE terminal condition you did not consider, because you did not have this fact: "
   "check_registered_stops applies the SPEC 5 pilot aborts under `if d is not None and init == \"A\"`. "
   "Init N therefore has NO 5 abort. Your version leaves the sole production run with no performance-based "
   "termination whatsoever between update 6,000 and update 120,000 - 92-98 GPU-h unwatched. "
   "3) A separate and worse bug was found and fixed: run_arm.sh decided --resume at unit-WRITE time, the "
   "production unit was written when the run dir was empty, and p4_train.py had no guard - so restarting that "
   "unit would have silently retrained the sole production run from update 0 over its own checkpoints."),
 "THE_FLOOR_TO_ATTACK": {
   "rule": ("At update 40,000, TERMINAL if EITHER frozen panel-D WER is not strictly better (1e-9 pp tolerance) "
            "than this run's OWN best frozen panel-D WER recorded at or before update 20,000, OR D empty rate > 1%. "
            "No validation record at or before 20,000 is an integrity failure, also terminal."),
   "claimed_answers_to_your_four_objections": {
     "1_unverified_frozen_conversion": "none performed: both sides are this run's own wer_frozen on panel D",
     "2_cherry_picked_neighbour": "nothing is selected from Phase 1's curve; Phase 1 does not appear in the rule",
     "3_two_points_cannot_cover_excursions": "no historical excursion must be covered; comparison is within one trajectory on one panel",
     "4_one_run_cannot_separate_recoverable_from_unrecoverable": ("not claimed. The rule asks only whether 20,000 further "
        "updates and another 17% of the anneal produced ANY improvement. It contains no chosen constant.")},
   "measured_context": [{"update": v["update"], "D_frozen_pp": round(v["panels"]["D"]["wer_frozen"]*100, 3),
                         "S_pp": v["S_pp"], "empties": v["panels"]["D"]["empties"]} for v in ours],
   "implementation": ("_best_d_frozen_by(run_dir, 20000) reads this run's val.jsonl; five test cases cover "
                      "still-improving/stalled/empties/missing-reference/absent-under-terminal-policy. 139 checks pass."),
   "epoch_phase_note": ("Phase 1's raw curve oscillated +-2-3 pp with epoch phase. Our epoch is 4,304.35 updates, so "
                        "update 20,000 sits at within-epoch phase 0.646 and update 40,000 at 0.293. Validations are "
                        "every 2,000, so the 20,000 reference is the MINIMUM over ~10 validations, not one reading.")},
 "run_state": ("N_O+macro_filtered_H120k, NVIDIA cold start, H=120,000, corpus 1,084,696 rows / 1,692.4 h, batch 252. "
               "Gracefully stopped at update 7,212 to adopt the policy (the trainer holds cfg in memory) and relaunched "
               "from its checkpoint. 868 updates/h. Loss ~1.03."),
 "council_P_verdict_verbatim": cp,
 "owner_position": ("Owner, verbatim, this round: \"Dude, don't fall into sunk cost fallacy. Just do whatever you need "
                    "to. I don't understand why you are not being honest and frank and direct to me. Just do what you "
                    "need to do. You have everything.\" He also holds: generalisation over benchmark; the model must be "
                    "good, not merely benchmark-passing; and he has ordered that Phase 1/2/3 logs are dead context."),
}
q = """
# QUESTIONS. Be adversarial. I adopted a rule you did not review; your job is to break it, not to ratify it.

Q1 BREAK THE FLOOR. Give the concrete scenario in which "D at 40,000 must be strictly better than this run's own best D
at or before 20,000" kills a run that would have produced a good final model, or lets through a run that is genuinely
hopeless. Quantify it. If the false-kill scenario requires a trajectory that is implausible for this recipe, say so.

Q2 THE REFERENCE POINT. Is 20,000 right? Alternatives: the best at or before 30,000 (tighter, closer to 40k); the best
at or before 12,000 (looser); the value at exactly 20,000 rather than the minimum to 20,000; or a required MARGIN
(e.g. at least 0.5 pp better) rather than strictly-better. Our epoch is 4,304.35 updates and validations are every
2,000, so a "best at or before U" is a minimum over U/2000 readings and is therefore biased low - which makes the floor
STRICTER than a single reading would be. Does that bias argue for a different reference or for a margin in the other
direction? Pick one exact rule and justify the number.

Q3 IS THE FLOOR WORTH HAVING AT ALL? You explicitly rejected a floor. Knowing now that init N has no SPEC 5 abort, do
you still prefer no performance stop between 6,000 and 120,000? If yes, say what watches the run and what the operator
does at 40,000 with a run that has flatlined. If no, say whether my rule or a different one.

Q4 ADAPTIVE-ANALYSIS RISK. The floor was written after seeing three validations of this run. Does the rule leak any
information from those observations that makes it self-fulfilling, and is its disclosure wording adequate? Give exact
wording if not.

Q5 THE RESTART BUG CLASS. run_arm.sh baked a launch-time decision into a unit written before the state it tested
existed. What OTHER decisions in this contract are frozen at write time but evaluated later, and which of them could
destroy or corrupt the run? Name them specifically. This matters more than the floor.

Q6 ADVERSARIAL. Given the owner's explicit sunk-cost warning, state the strongest case that continuing this run at all
is the wrong call, and what single measurement at 8,000/12,000/16,000 would make you say stop now rather than at 40,000.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. You ruled as council P; your verdict is included verbatim and was ADOPTED - but with one addition "
          "you did not review, made because of a code fact you did not have. This round is adversarial: attack the "
          "addition and the bug class around it. Do not ratify. Be concrete and numeric, and say plainly where you were "
          "wrong or where I am. Below: CONTRACT, FACTS, QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
d = Path('/tmp/council_p4q'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
