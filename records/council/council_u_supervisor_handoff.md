**Verdict: NO to the proposed live kill-and-recreate operation. Leave the running processes alone now.** Moving supervision into systemd is the right eventual arrangement, but the supplied handoff is not sufficiently controlled to approve as written.

I reviewed the supplied scripts and their repository copies. I did not signal processes or modify files. The production PIDs are not visible in this review environment; live-state statements below rely on your measured packet.

**Q1 — What is actually at risk?**

Closing the terminal does not stop the trainer under the measured arrangement. Its separate worker service and lingering user manager provide that separation. One correction: its reported parent is PID **1810**, not PID 1; that is consistent with a user systemd manager.

For loss of the supervisor to delay training, both must occur:

1. The supervisor disappears or becomes ineffective.
2. The worker needs a **permitted recovery** before supervision is restored.

Those events need not be independent. A resource incident could affect both processes.

The consequences are:

| Event | Consequence |
|---|---|
| Supervisor disappears; trainer remains healthy | Training continues. |
| Trainer subsequently crashes | Training waits for human recovery. |
| Crash with healthy 500-update checkpoint rotation | Up to approximately **35 minutes** of training must be replayed at 859 updates/hour. That rollback also occurs with a functioning supervisor. |
| Nobody detects the stopped trainer | Additional downtime lasts until somebody intervenes—potentially days. |

The supervisor’s absence principally adds **recovery delay**, not loss of all accumulated training. No supplied failure-rate data support a numerical probability or expected-loss estimate.

Two corrections to the packet’s rationale matter:

- A **watchdog STOP must not automatically restart**. That requires the registered owner-authorized recovery.
- `P4_ONLY_PREF=1` and `P4_NO_YIELD=1` mean routine card switching and courtesy yielding are not active requirements for this run.

I would accept the present arrangement temporarily while preparing a controlled replacement. I would **not certify it for 4.9 unattended days**. That limitation does not justify killing the working supervisor before its replacement is demonstrably ready.

**Q2 — Why I cannot approve the requested command sequence**

There is no ordering of the supplied three commands that makes both overlapping supervision and a supervision gap impossible:

- Kill first: a gap exists, and replacement startup can fail.
- Start first: two supervisors can act concurrently.
- Check, then act: the checked state can change between those operations.

The trainer remains owned by systemd throughout; “unsupervised” is the precise concern, rather than Unix process orphaning.

More importantly, [`adopt_active()`](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:109) does not implement **“adopt this exact worker or refuse.”** It:

- Searches by a substring.
- Returns the first matching active unit.
- Suppresses query errors.
- Treats a failed query like absence and enters card selection.
- Defaults to A5000 if reading the worker’s environment fails.

The memory checks and existing unit name provide additional protection; a failed lookup does **not automatically produce a second trainer**. Nevertheless, they are not an adoption handshake. The trainer lock is also an unsuitable handoff mechanism because a rejected second trainer writes an ABORT.

Therefore, **I approve no production `kill`/`systemd-run` sequence against this script today**. A replacement needs an independently tested initial mode that refuses launch unless it has adopted the exact expected worker, plus a controlled transfer of exclusive supervisory authority. Prepare that separately; do not edit the executing Bash script in place.

The acceptance evidence for a later handoff must include all of these:

- Exactly one supervisor with authority to launch or stop workers.
- A new, timestamp-bounded adoption record naming the exact worker.
- Worker `MainPID`, process start time, `InvocationID`, and cgroup unchanged from the pre-handoff snapshot.
- Worker still `active/running`.
- No new worker invocation or trainer startup.
- No new STOP, ABORT, or WATCHDOG_STOP.
- Training updates continue.
- The replacement’s **actual process environment** matches the approved settings.

An “adopted” log line alone is insufficient. Likewise, `Type=exec` proves executable startup, not successful application-level adoption. [systemd service documentation](https://github.com/systemd/systemd/blob/v249/man/systemd.service.xml)

If adoption fails, the replacement must remain **unable to launch**. Stop that replacement, verify its processes and outstanding actions are gone, and restore the former supervisor only after rechecking the worker and terminal records. If the worker exited meanwhile, classify that event before recovery; do not force an adoption or clear records.

**Q3 — Environment**

Yes, these are correct `systemd-run` arguments:

```bash
--setenv='P4_EXTRA_ARGS=--config configs/p4_ctc_initN.yaml'
--setenv='P4_ONLY_PREF=1'
--setenv='P4_AFTER='
--setenv='P4_NO_YIELD=1'
```

`P4_AFTER=` explicitly supplies an empty value. No additional escaping is needed. Include the `=`; omission is not equivalent. The outer shell quotes preserve the space-containing assignment as one argument. [systemd-run documentation](https://github.com/systemd/systemd/blob/v249/man/systemd-run.xml)

**Those four assignments are necessary, but not a complete environment qualification.**

A user service inherits the **user manager’s environment**, not an arbitrary launching terminal’s current environment. [systemd execution documentation](https://github.com/systemd/systemd/blob/v249/man/systemd.exec.xml)

Check these dependencies before any handoff:

- **Working directory:** explicitly set `/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline`; invoke Bash and the supervisor by absolute paths.
- **PATH:** the supervisor resolves `python3`, `systemctl`, `nvidia-smi`, `flock`, and ordinary utilities through PATH. Verify their actual resolution inside a harmless service preflight.
- **User-manager access:** verify UID 1004, the correct runtime directory, and successful queries to the same user manager from that preflight.
- **Other controls:** inspect both environments for additional `P4_*` settings, particularly `P4_NEED_MIB`, `P4_DRY`, and override flags.
- **Shell/library injection:** inspect relevant `BASH_ENV`, exported shell functions, `PYTHONPATH`, and `LD_*` settings. Do not blindly copy the entire environment.

The worker is another service launched by the manager. It does **not** inherit arbitrary supervisor variables merely because the supervisor invokes `systemd-run`. The launcher explicitly sets its GPU UUID, A6000 authorization, allocation configuration, thread limits, cache directory and temporary directory. Compare the generated replacement worker specification with the deployed worker specification.

**Q4 — Is SIGTERM safe?**

A positive-PID `SIGTERM` to the verified supervisor does not inherently signal the trainer in its separate service. The supplied supervisor has no explicit TERM trap that stops the worker.

But **“the Bash process exited” does not prove “all actions it initiated have ceased.”**

- A child `run_arm.sh` or `systemd-run` can outlive its parent.
- A start/stop request already accepted by systemd can continue after the requesting process dies.
- A sleeping child can survive briefly.
- An inherited open file description can keep an `flock` held after the parent exits.

A leftover lock **pathname** is harmless; an open locked descriptor is what matters. Never delete a lock file to “clear” it.

The preferable termination point would be the active-worker monitoring sleep, with the expected worker still running, no launch/stop child or outstanding job, and no card lock held. Merely seeing `sleep 60` is insufficient: the card-selection loop also sleeps for 60 seconds.

Bash’s documented signal behavior also distinguishes interactive shells, noninteractive shells and trapped signals; the command’s return status is not a termination handshake. [Bash signal documentation](https://www.gnu.org/software/bash/manual/html_node/Signals.html)

Any eventual signalling must verify process identity using more than yesterday’s PID—preferably a pidfd—and observe termination. Do not use a process-group kill, `pkill -f`, or stop the terminal scope.

**Q5 — The watchdog and Monitor remain exposed**

Yes. Moving only the supervisor would protect **only that supervisor**. It would not establish that the watchdog, validation reporting or checkpoint-preservation helper survives terminal-scope teardown.

Their replacements need separate lifecycle reviews. In particular, the repository watchdog has concrete dependencies and limitations:

- Its default unit expression resolves this run name to **`worker-h120k`**, not the actual worker. Preserve and verify its explicit `P4_UNIT`.
- Preserve its actual interval, `WATCHDOG_GIB`, and `STALL_MIN`. The script’s default interval is **3,600 seconds**; do not assume the live watchdog polls every minute.
- It exits successfully when the worker is inactive or STOP/ABORT appears. Simply placing it in a service with `Restart=on-failure` would not restore monitoring after that successful exit. [Watchdog source](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/watch_p4.sh:109)
- Its memory-breach counter resets on restart.

The Monitor’s implementation and environment were not supplied, so I cannot certify its migration.

I also found **two surviving supervisor defects directly relevant to claims of robustness**:

1. [`run_supervised.sh:212`](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:212) still declares COMPLETE from the last logged update alone. It bypasses the stronger checkpoint-and-validation check at the top of the loop.
2. `WATCHDOG_STOP` is checked at startup, but not after worker exit or immediately before another launch. A running supervisor can therefore accept the trainer’s generic `graceful_stop` and requeue despite the external terminal cause. The inspected trainer’s resume refusal checks STOP and ABORT, not WATCHDOG_STOP.

These reopen the corresponding closure claims. Service placement fixes neither. They warrant a separately prepared, tested supervisor repair; they do not require resetting training.

**Q6 — The most credible maintenance-induced harm**

**Starting a replacement with the wrong effective environment, then declaring success because it adopted the currently healthy worker.**

For example, a `systemd-run` invocation missing the explicit `P4_EXTRA_ARGS` assignment can adopt perfectly today. On the next crash, its launcher uses the default init-A configuration, and the identity guard turns the attempted recovery into a terminal refusal. Missing `P4_NO_YIELD=1` instead re-enables the branch containing:

```bash
systemctl --user stop "$UNIT"
```

That can interrupt healthy training after co-tenant activity.

The preventive check is an **exact environment assertion inside a harmless service preflight before touching the old supervisor**, followed by inspection of the replacement’s actual environment before granting it authority. Also inspect the future worker command it would generate. Neither `systemd-run` returning zero nor one adoption message checks those future behaviors.

**The trainer can keep running throughout this preparation. Do not execute the proposed kill-first handoff.**
