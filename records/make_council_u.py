#!/usr/bin/env python3
"""Council U: move the live supervisor into its own systemd unit, without disturbing the trainer."""
import json, datetime, subprocess, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
sup = (R / 'experiments/p4/run_supervised.sh').read_text(encoding='utf-8')
arm = (R / 'experiments/p4/run_arm.sh').read_text(encoding='utf-8')

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_verbatim": "\"I need precision and correctness. So reconcile with Codex Council and then do it.\"",
 "the_question": ("The owner asked whether closing his terminal kills the training. Verified answer: no. But the "
   "SUPERVISOR is a bare detached process sitting in a GNOME Terminal cgroup, and the proposal is to move it into its "
   "own systemd --user unit while the run is LIVE at update ~16,900 of 120,000. Rule on whether to do it, and give the "
   "exact sequence."),
 "measured_current_state": {
   "trainer": {"pid": 703527, "ppid": 1810, "own_session": True, "tty": None,
     "cgroup": "/user.slice/user-1004.slice/user@1004.service/app.slice/worker-n-o-macro-filtered-h120k-a6000.service",
     "unit": "worker-n-o-macro-filtered-h120k-a6000.service (transient, Restart=no, SuccessExitStatus='0 4')",
     "holds": "an exclusive flock on experiments/p4/runs/N_O+macro_filtered_H120k/.run.lock (SPEC 8w.10); "
              "pt_data_worker children share the fd"},
   "supervisor": {"pid": 213976, "ppid": 1810, "own_session": True, "tty": None, "elapsed": "22h22m",
     "cgroup": "/user.slice/user-1004.slice/user@1004.service/app.slice/app-org.gnome.Terminal.slice/"
               "vte-spawn-13707d5b-8c0f-4c5c-91fa-529141eea63d.scope",
     "argv": "/bin/bash experiments/p4/run_supervised.sh O+macro_filtered 120000 N_O+macro_filtered_H120k a6000",
     "cwd": "/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline",
     "env_that_must_be_preserved": {"P4_EXTRA_ARGS": "--config configs/p4_ctc_initN.yaml", "P4_ONLY_PREF": "1",
                                    "P4_AFTER": "", "P4_NO_YIELD": "1"},
     "holds_no_lock": True},
   "linger": "Linger=yes for this user, so the user systemd manager survives logout",
   "terminal_records": "no STOP, no ABORT, no WATCHDOG_STOP in the run directory",
   "run": "update ~16,900 of 120,000, 859 upd/h, WER 23.216% at the 16,000 validation, ~4.9 days remaining"},
 "why_this_matters": ("The trainer is safe: own service cgroup, PPID 1 systemd, Linger=yes. The SUPERVISOR is in the "
   "terminal's vte-spawn scope. A scope-level teardown (that terminal's scope killed, or a session cleanup that walks "
   "the cgroup) would take it, leaving the trainer running but unsupervised - so no relaunch after a crash, OOM, "
   "watchdog stop or card switch, for the remaining ~4.9 days. run_supervised.sh's OWN header says it is meant to run "
   "as a systemd user unit 'p4-sup-<run>'; running it bare is the deviation."),
 "the_proposed_operation": ("1) kill PID 213976; 2) systemd-run --user --unit=p4-sup-n-o-macro-filtered-h120k with the "
   "same argv, cwd and the four P4_* variables; 3) confirm it adopts the RUNNING worker rather than launching a second "
   "one. The trainer is never signalled."),
 "adoption_mechanism_verified": ("run_supervised.sh has adopt_active(): it lists active 'worker-*' units, matches the "
   "run key (RUN lowercased with '+' and '_' -> '-', i.e. 'n-o-macro-filtered-h120k'), and adopts. Verified that key IS "
   "a substring of 'worker-n-o-macro-filtered-h120k-a6000.service'. The D11 startup guard passes because there is no "
   "STOP/ABORT."),
 "risks_I_can_see": [
   "a gap between kill and restart in which the trainer could exit with nobody to relaunch it (seconds)",
   "two supervisors briefly coexisting if ordering is wrong -> two workers, two trainers on one run directory (the "
   "trainer's flock from SPEC 8w.10 would make the second abort, but that is a terminal ABORT poisoning the run dir)",
   "losing P4_EXTRA_ARGS='--config configs/p4_ctc_initN.yaml': a future relaunch would then use the DEFAULT "
   "configs/p4_ctc.yaml, which is init A with H=150,000. The SPEC 8w.10 identity binding should refuse that, but it "
   "would be a terminal ABORT rather than a clean continuation",
   "losing P4_NO_YIELD=1: the run would start yielding its card to co-tenants again, against the owner's standing rule",
   "the new unit inheriting a different PATH or cwd and failing at launch",
   "doing this at all when the trainer is already safe - i.e. accepting an avoidable operation on a live 5-day run"],
 "alternative_considered": ("Leave it. The trainer survives terminal close and logout regardless. The only exposure is "
   "'trainer crashes AND terminal scope is gone' - and a human could relaunch manually."),
 "run_supervised_sh": sup,
 "run_arm_sh": arm,
}
q = """
# QUESTIONS. The owner asked for precision. Be exact; this touches a live production run at 14% of a 5-day schedule.

Q1 DO IT OR NOT. Is moving the supervisor into its own unit worth the operational risk on a live run, or is the
residual exposure (trainer safe, supervisor in a terminal cgroup) small enough to leave alone for ~4.9 days? Quantify
the exposure as best you can: what actually has to happen for the current arrangement to cost anything, and what would
it cost? Give a clear yes/no.

Q2 IF YES, THE EXACT SEQUENCE. Give the precise ordered commands, including the systemd-run invocation with every
property and environment variable, in the order that makes a double-supervisor or an orphaned trainer impossible.
State what must be verified between each step and the exact check that proves adoption happened rather than a second
worker being launched. State what to do if adoption FAILS.

Q3 THE ENVIRONMENT. Four variables must survive: P4_EXTRA_ARGS='--config configs/p4_ctc_initN.yaml', P4_ONLY_PREF=1,
P4_AFTER=, P4_NO_YIELD=1. Is passing them via --setenv sufficient and correct, and does an empty P4_AFTER need special
handling? Is there anything else in the current process environment (PATH, cwd, CUDA_*) the supervisor or its children
actually depend on that a fresh unit would not inherit?

Q4 THE KILL. Is a plain SIGTERM to PID 213976 safe, given it is a bash while-loop that may be mid-`systemctl` call or
mid-sleep? Could killing it leave a half-created worker unit or a stale lock file? Should it be killed at a particular
point in its loop, and how would one know?

Q5 WHAT ELSE IS EXPOSED. The supervisor is not the only thing launched from that terminal scope. The memory watchdog
(watch_p4.sh) and the validation Monitor were started the same way. Should those move too, and does moving only the
supervisor create a misleading impression of safety?

Q6 ADVERSARIAL. The single most likely way this maintenance operation ends up harming a run that was otherwise fine.
Name the mechanism, the command that causes it, and the check that catches it before damage.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. A production training run is LIVE at update ~16,900 of 120,000 with ~4.9 days remaining. The "
          "owner asked whether closing his terminal would kill it; the answer is no, but the SUPERVISOR that relaunches "
          "the trainer after a crash is a bare process in a terminal cgroup. The proposal is to move it into its own "
          "systemd user unit, live. The owner said: 'I need precision and correctness.' Be exact, be adversarial about "
          "the operation, and say plainly if the right answer is to leave it alone.\n\n"
          "# CONTRACT\n" + spec + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False, default=str) + "\n" + q)
d = Path('/tmp/council_p4u'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
