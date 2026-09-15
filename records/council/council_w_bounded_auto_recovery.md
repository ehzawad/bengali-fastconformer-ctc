**Approve bounded automatic recovery, but reject deployment against the current supervisor.** Keep the A5000 run going while a replacement is prepared and tested. Until those checks pass, **notify the owner and handle migrations manually**. Council U’s refusal still stands.

I inspected the repository copies of the supervisor, launcher, watchdog and notifier. I changed nothing and did not inspect or signal production processes; current GPU state comes from your packet.

**Q1 — Should this be automated?**

Yes, with a controller that prioritises keeping an existing trainer productive. Automatically chasing whichever card currently shows more free memory is unacceptable.

The immediate arithmetic is particularly revealing:

- A6000 free memory increased by **10,323 − 128 = 10,195 MiB**.
- That lies within the previously observed **10,096–10,230 MiB whole-process footprint** of our trainer.
- Therefore, the increase is consistent with **our departure releasing our allocation**. It does not establish that the co-tenant freed anything.
- Returning with the same allocation would leave approximately **128 MiB again**.
- **10,323 MiB is already below the unchanged 12,000 MiB launch floor.**

There are **100,079 updates remaining**: approximately **109.7–122.8 training hours** at the supplied A6000 rates, before additional overhead. Neither card’s name establishes which completes those updates sooner.

Expected net benefit cannot be calculated from this packet: it contains neither intervention frequency nor controller-failure probabilities. The known costs are substantial enough to constrain the policy:

- One six-minute reload costs approximately **86 potential updates** at 859 updates/hour.
- Switching every 30 minutes would consume **20% of wall time in reloads alone**—approximately **23.5 hours over 4.9 days**.
- Four migrations daily over five days cost approximately **two hours of reloads**, excluding checkpointing, failures and replay.

Avoiding an unattended multi-hour outage can justify reliable recovery automation. Those figures do **not** justify introducing an untested migration controller into the sole production run.

**Q2 — Exact policy I would approve**

These are proposed operational thresholds, not measured failure boundaries. All memory figures below are **MiB reported by UUID**, with valid, fresh telemetry.

| Decision | Rule |
|---|---|
| Polling | Inspect both cards, the exact worker and controller state every **60 seconds**. Recheck immediately before an action. |
| Current card “blocked” for migration | During established training, free memory **<1,024 MiB on two consecutive polls**, or **<256 MiB on one poll**. This requests a resource handover; it does not itself declare a terminal failure. |
| Destination for necessary recovery | At least **12,000 MiB free before our allocation**, observed on two consecutive polls and rechecked before launch. Foreign-process presence alone does not disqualify it. |
| A6000 available for an elective return | At least **18,500 MiB free before our allocation throughout 15 minutes**, with no missing samples, plus the throughput condition in Q5. |
| Minimum residence | **Six hours after verified training resumes** before an elective migration. A verified resource emergency may bypass residence time. |
| Migration ceiling | **Four attempted card changes in a rolling 24 hours**, including failures, rollbacks and manual changes. At most **one elective return**; initiate it only when no more than one earlier change is in that window, leaving two attempts for recovery. Persist counters across supervisor restarts. |
| Both cards unavailable; worker still progressing | **Keep training where it is.** Report memory pressure. Do not stop a functioning worker merely to enter a queue. |
| Both unavailable; worker already stopped | Retain the checkpoint and poll every **60 seconds**. Start on the first qualifying card; prefer A6000 when both qualify. Never lower the launch floor to manufacture availability. |
| Migration ceiling reached | Keep a healthy worker running. If stopped, enter an operational hold and notify the owner; no fifth automatic migration. |
| Missing or failed telemetry | State is **UNKNOWN**, never “empty” or “free.” Make no new migration decision; preserve a healthy worker and alert after two consecutive failed polls. |

The **18,500 MiB** elective threshold makes “5 GB headroom” explicit: even against a **13,312 MiB whole-process planning envelope**, it leaves **5,188 MiB, approximately 5.44 decimal GB**. Verify whole-process usage; **9.59 GiB torch-reserved memory is not total process memory**. Free-memory observations cannot reserve capacity against subsequent co-tenant allocations.

The **12,000 MiB launch floor must never be applied to raw free memory after our trainer has loaded**. Doing that recreates the earlier self-induced stop loop.

When both cards are unavailable, literal “never idle” is impossible without violating another constraint. Migration limits, terminal outcomes and the owner’s pause must also take precedence. That limitation must be stated in the amendment.

**Q3 — Reconcile no-yield with move-when-blocked**

I would record this policy:

> Maintain one productive Phase-4 trainer whenever resources and the registered rules permit. A co-tenant arriving, leaving or changing memory use triggers notification, not courtesy yielding. A resource handover may interrupt training only under the registered memory-pressure conditions and with a qualifying destination. An elective return to A6000 additionally requires sustained headroom, residence time, migration allowance and measured benefit. Owner pause, terminal outcomes, watchdog causes, integrity failures and the budget ceiling take precedence over automatic continuation.

Thus:

- The old **absolute prohibition on moving** gives way to a qualifying resource handover.
- Its prohibition on **courtesy yielding merely because another process appeared** remains.
- The preference for A6000 gives way to continuity when the elective conditions fail.

**Do not implement this by setting `P4_NO_YIELD=0`.** That reactivates the existing newcomer/memory-growth branch, which implements a different policy.

The owner’s rare complete stop also needs a **durable PAUSED record written before signalling**. A generic `graceful_stop` must not cause the controller to undo his pause. Only explicit owner resumption clears it.

**Q4 — Minimum safe implementation**

“Stop the trainer, then swap supervisors” is **not sufficient as an automated primitive**. While the old supervisor remains authoritative, it can relaunch; outstanding launch requests can outlive it; failed queries can falsely establish absence. Your observed manual sequence does not prove those races are excluded.

The replacement should be **one persistent supervisor that changes worker placement**, not a supervisor replacement on every migration. Minimum requirements:

1. **Exclusive authority and exact worker identity.**  
   Hold a process-lifetime supervisory lock, separate from the trainer lock. Every authorised launch path must respect it. Replace substring adoption with an exact expected unit, run identity, GPU UUID, `InvocationID`, PID/start time and cgroup check. A query error means refusal to act—not absence and never default A5000. The trainer lock remains a backstop; a rejected duplicate launch must not poison the real run with an ABORT.

2. **A durable migration transaction.**  
   Record the reason, source invocation, destination and attempt count before stopping. Obtain and pin a completed checkpoint, its SHA-256 and full resume identity. Verify the source worker and descendants have exited and no start/stop request remains pending. Recheck terminal records and destination memory, then launch exactly that checkpoint. Verify restored update, sampler/LR handshake, GPU and subsequent progress before declaring success. Keep the checkpoint through any failed attempt.

3. **Selection and launch must share the same lock interval.**  
   Currently, [`pick_card` runs in command substitution and its lock is reacquired later](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:140), without repeating the eligibility checks under the reacquired lock. Fix that gap. The lock coordinates our launchers; it cannot reserve memory against co-tenants.

4. **Bound startup failure and distinguish waiting from training.**  
   The [launcher can wait indefinitely for memory inside an active service](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_arm.sh:77), while the supervisor treats that service as running. Require explicit WAITING, LOADING, VALIDATING and TRAINING states. After **10 minutes without verified startup progress**, alert and classify the failure. A confirmed pre-training memory waiter may be stopped and rescheduled; an ambiguously identified live trainer may not. Allow **one recovery attempt after a failed destination launch**, then hold and notify. Never start another worker while the previous invocation remains unresolved. `Type=exec` establishes executable startup, not training readiness. [systemd service documentation](https://github.com/systemd/systemd/blob/v249/man/systemd.service.xml)

5. **Repair the surviving terminal and completion paths before deployment.**  
   The [post-exit branch still declares COMPLETE from the logged update alone](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:212). `WATCHDOG_STOP` is checked at startup but omitted from the later exit/launch checks. Use one authoritative classifier at startup, after exit, before signalling and immediately before launch. Watchdog and terminal causes override a simultaneous graceful-stop acknowledgement. Qualified completion must require the bound finalisation evidence. Never automatically set an override or erase a terminal cause based on a matching error substring.

A successful stop command is not a checkpoint acknowledgement. The existing launcher permits forced termination after its stop timeout; `KillMode=mixed` does not guarantee checkpoint completion. If checkpoint confirmation is absent after **five minutes**, alert and inhibit destination launch rather than escalating merely to finish a migration. [systemd termination documentation](https://github.com/systemd/systemd/blob/v249/man/systemd.kill.xml)

Prepare this separately from executing scripts. Before one controlled installation, test controller death at each transaction boundary, destination loss after selection, query failures, duplicate supervisors, partial checkpoints, simultaneous watchdog/manual stops, the active memory-wait case and a crash after logging update 120,000.

Inspect the **actual replacement environment and future worker command**: init-N configuration, H=120,000, exact run identity, restart ownership and disabled override flags. Watchdog, reporting and checkpoint-preservation helpers must follow the exact new invocation and survive terminal loss. Passing tests against a file that the deployed unit does not execute is insufficient.

No restart from update zero is required.

**Q5 — Measure A5000 before paying to leave it**

Measure it now through ordinary training. After loading and initial warm-up settle, collect **three consecutive 20-minute training windows**, recording:

- Completed updates and elapsed time.
- Actual GPU UUID and worker invocation.
- Whole-process memory and co-tenant conditions.
- Validation/checkpoint overhead separately for the wall-clock forecast.

Use comparable UUID-bound A6000 production windows. A run name—or an overwritten final `run_meta`—does not establish the hardware used during an earlier interval. Do not launch a separate benchmark or change the sample sequence.

My elective-return requirement is:

> The conservative observed A6000 rate must exceed the upper observed A5000 rate by **at least 10%**, alongside Q2’s memory and residence conditions.

With only the supplied A6000 range, using **815 updates/hour** conservatively means the A5000 must measure at most approximately **741 updates/hour** to clear that rule. This is an operational margin, not a statistical confidence bound. If the evidence is insufficient, the elective return stays disabled.

For a fixed six-hour wall-clock window, the break-even speed ratio is:

\[
r_{6000}/r_{5000} > T/(T-C)
\]

where \(C\) includes all migration downtime.

| Downtime within six hours | Minimum speed advantage |
|---|---:|
| One six-minute migration | **1.7%** |
| One ten-minute migration | **2.9%** |
| Two ten-minute migrations | **5.9%** |

Use **ten minutes as the initial planning allowance**, replacing it with a larger measured checkpoint-to-progress cost where necessary. Require the complete transition cost to be observed before enabling elective migration.

The 10% requirement provides margin above nominal break-even. Fifteen quiet minutes do not guarantee six hours of future residence; co-tenant history and realised migration costs must remain visible. At an already necessary launch, A6000 may retain the owner’s capacity preference without claiming a demonstrated speed advantage.

**Q6 — Most likely automation-induced loss**

**Mistaking the memory we just released for newly available co-tenant capacity.**

The revealing number is **10,195 MiB**. A naive “A6000 now has more than 5 GB free” rule sends us back, recreates approximately 128 MiB headroom, and triggers another departure. Alternatively, the existing 12,000 MiB launcher floor leaves the new worker **active but waiting**, while the previously usable card sits idle.

The cheapest guard is **no elective return based on gross free memory**: account for our arriving allocation, enforce the higher sustained threshold, and keep a healthy worker running.

The notifier also needs small corrections before its output is trusted. Its [snapshot includes memory in the process identity](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/watch_cotenants.sh:9), so a memory change currently becomes both ARRIVED and LEFT. Its messages query free memory by indices `0/1` while labelling them A5000/A6000.

Use UUIDs throughout; identify processes by UUID, UID, PID and start time; report memory changes separately; preserve the prior snapshot on query failure. Deliver every observed change without the migration debounce, and verify the owner-facing delivery path. A 60-second poll cannot promise detection of jobs that begin and end between polls.

**The immediate disposition is: continue on A5000, measure its rate, keep notifications, and prepare the bounded controller. Do not turn the current supervisor into an unattended card follower.**
