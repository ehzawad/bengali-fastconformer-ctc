#!/usr/bin/env python3
"""Council W: the owner wants a card-following policy - never idle, prefer the A6000, move when blocked."""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
sup = (R / 'experiments/p4/run_supervised.sh').read_text(encoding='utf-8')
cu = Path('/tmp/council_p4u/out.log').read_text(encoding='utf-8')

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_requirements_verbatim": (
   "\"if a something is not written for a 5000 like auto start you should write it and also if you see a 6000 is free "
   "then you can go back to it because it has more GPU ... like 5GB headroom if you move to it so that's the idea and "
   "if that a 5000 get block then you should move to there but you should always occupy one GPU because training has "
   "to be going\" and \"every time someone occupy GPU you should let me know ... ultimately you should occupy at least "
   "one GPU but sometimes but very very rarely I may need to stop it entirely\"."),
 "what_just_happened": (
   "The A6000 filled to 128 MiB free (co-tenant A at 38 GB). The owner chose to move. I executed: graceful stop of "
   "the trainer at update 19,921 (checkpoint written), killed the supervisor with NO live worker present so "
   "adopt_active() could not misfire, then relaunched the supervisor with preference a5000 and P4_ONLY_PREF dropped so "
   "it may fall back. The run is now loading on the A5000. Minutes later the A6000 freed to 10,323 MiB - which is "
   "exactly the thrash scenario a naive 'move back when free' rule would trigger."),
 "cards": {"A5000": "24,564 MiB total, uuid GPU-3ce8e4c2-...", "A6000": "49,140 MiB total, uuid GPU-4754ca84-..."},
 "our_footprint": "peak reserved 9.59 GiB; watchdog stop at 13.0 GiB; launch floor P4_NEED_MIB 12,000 MiB",
 "measured_throughput": ("A6000: 815-912 updates/hour across this run and the historical controls. A5000: UNMEASURED - "
   "the historical run named O_A5000 in fact executed on an A6000 (run_meta gpu field), so there is no A5000 number for "
   "this workload. Migration costs a graceful stop plus ~5-6 minutes of model load and dataloader construction."),
 "current_state": "update 19,921 of 120,000, WER 22.765% at the 18,000 validation, ~4.9 days remaining at A6000 rates",
 "existing_machinery": ("run_supervised.sh already has pick_card (preferred card if no FOREIGN compute process and "
   "enough free memory, else the other card, else wait) and a yield rule (a foreign process appearing on our card -> "
   "systemctl stop -> back to card selection). P4_ONLY_PREF=1 pins it to one card; P4_NO_YIELD=1 disables yielding "
   "entirely. The owner set P4_NO_YIELD=1 on 2026-09-08: 'no don't yield to another it's just MB man; as full training "
   "has started you won't yield it to anyone.' That rule and the new 'move when blocked' requirement are in tension."),
 "council_U_constraint": ("You ruled earlier today that adopt_active() is a substring search that returns the first "
   "match, suppresses query errors, and DEFAULTS TO A5000 when reading the worker environment fails - so it is not an "
   "adoption handshake, and you refused a live supervisor swap. Any automatic migration feature exercises exactly this "
   "machinery, repeatedly and unattended, for ~4.9 days."),
 "council_U_verdict_verbatim": cu,
 "run_supervised_sh": sup,
 "notifier_already_built": ("experiments/p4/watch_cotenants.sh is armed: it polls every 60 s and emits one line when a "
   "foreign compute process appears on or leaves either card, with the free memory on both. It is read-only. The owner "
   "is notified on each event so he can intervene."),
}
q = """
# QUESTIONS. The owner wants training never to sit idle, and wants the better card when it is available. Design it, or
# tell him the automation is a worse idea than the manual arrangement. Be numeric.

Q1 SHOULD THIS BE AUTOMATED AT ALL? Given your own finding that adopt_active() is unsafe and that each migration costs
a stop plus ~5-6 minutes of reload, is an automatic card-following policy net positive over 4.9 days, or does it create
more expected loss (thrashing, mis-adoption, a failed relaunch leaving the run idle) than the idle time it prevents?
Give a clear recommendation. If your answer is "notify the owner and let him decide each time", say so plainly - the
notifier already exists.

Q2 IF AUTOMATED, THE EXACT POLICY. Specify: the free-memory threshold at which the current card counts as BLOCKED; the
threshold at which the A6000 counts as AVAILABLE enough to move back to (the owner said "5GB headroom"); a minimum
dwell time on a card before any migration is permitted, to prevent ping-pong; a maximum number of migrations per 24 h;
and what happens when BOTH cards are blocked. Give numbers, not principles. Note that within ten minutes today the
A6000 went 128 MiB free -> 10,323 MiB free, so any rule must survive that.

Q3 THE TENSION. The owner's standing P4_NO_YIELD=1 rule says never surrender a card to a co-tenant. His new rule says
move when blocked. Reconcile them into one policy statement he can approve, and say which of his two instructions must
give way in which circumstance.

Q4 SAFE IMPLEMENTATION. Given adopt_active()'s defects, what is the minimum change that makes repeated unattended
migration safe? Is "stop the trainer first, then swap, so there is no worker to adopt" - the sequence I used manually
today - sufficient as an automated primitive, or does automation reintroduce the race you refused?

Q5 THE A5000 UNKNOWN. We have no measured A5000 throughput. Should the policy prefer the A6000 on measured evidence, or
should a throughput measurement be taken first, and how (the run is on the A5000 right now, so its rate will be
observable within the hour)? What rate difference would justify migrating back and paying the reload cost?

Q6 ADVERSARIAL. The most likely way an automatic card-following policy ends up costing MORE training time than doing
nothing. Name the mechanism, the number that reveals it, and the cheapest guard against it.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. A production run at update 19,921 of 120,000 has just been migrated by hand from a full A6000 to "
          "an idle A5000. The owner now wants this automated: never idle, prefer the bigger card, move when blocked, and "
          "notify him on every co-tenant change. You refused a live supervisor swap earlier today; that verdict is "
          "included. Rule on whether to automate at all, and if so give the exact policy. Be adversarial about "
          "automation on a run with no fallback.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
d = Path('/tmp/council_p4w'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
