**Council M: recommend (a), with a fixed baseline. Reject the proposed upward rebasing.** This requires owner adoption; until then, rule (c) remains binding.

**Q1 — No polling threshold guarantees co-tenant safety.**

| Situation | Memory available for another allocation |
|---|---:|
| A6000 before newcomer: 49,140 − 33,064 − 10,200 | ≈5,876 MiB |
| A6000 after 356 MiB newcomer | ≈5,520 MiB |
| A5000, last recorded 10,538 MiB free before our launch | ≈338 MiB after our 10,200; **launch prohibited** |
| Either card launched at the permitted 12,000 MiB minimum | ≈1,770–1,800 MiB after our measured process footprint |

On the A6000, a **6,144 MiB allocation can fail immediately**, although it would fit after our release, when approximately **15,720 MiB** would be available. A failed allocation need not produce observable memory growth.

A successful growth event can wait **60 seconds for detection plus up to 300 seconds for release**. Even a **0 MiB growth threshold** cannot prevent an allocation between polls. NVIDIA explicitly documents races between reported free memory and other processes’ allocations. [CUDA documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)

Thus **2,000 MiB is a courtesy trigger, not reserved headroom**. No replacement number guarantees protection. Coordinated release before allocation—or **zero GPU occupancy by us**—removes our contribution to memory contention. Never signalling a co-tenant satisfies the process-control prohibition; it does not guarantee no interference.

**Q2 — Pick (a), corrected.**

Compare **total foreign memory against one fixed, pre-launch baseline**. Never increase that baseline during the worker’s lifetime.

Otherwise six successive **356 MiB** arrivals consume **2,136 MiB**, yet individually rebasing each one prevents the registered growth trigger from firing.

Option (b) is not equivalent: a **2,200 MiB newcomer replacing 2,200 MiB of departing foreign allocation** triggers its size condition but produces zero aggregate growth.

**Q3 — For the 356 MiB arrival, no stop under the amended memory rule.**

One unnecessary yield per hour costs **72–120 minutes/day**, or **5–8.3% of wall time**.

When release is required, retain **our worker’s graceful checkpoint-and-exit as the sole authorised release mechanism**. Never signal a co-tenant. Sleeping or suspending our process retains allocations; `empty_cache()` releases only unused cache, not live tensors. [PyTorch documentation](https://docs.pytorch.org/docs/main/notes/cuda.html)

**Q4 — Proposed exact amendment:**

> **8u. Memory-based co-tenant yielding — effective upon owner adoption.**
>
> Replace the persistent-newcomer trigger with aggregate foreign-memory growth strictly greater than 2,000 MiB. Capture a valid foreign-memory baseline immediately before each worker launch. Keep that numeric baseline fixed until the worker exits; recording new process identities must not increase it.
>
> Poll at intervals ≤60 seconds. On the first valid sample exceeding the threshold, gracefully checkpoint and release our worker within five minutes of detection, without a second-poll delay. A newcomer alone does not trigger release. Invalid telemetry must not be interpreted as zero usage: inhibit launch, or gracefully release our running worker until telemetry recovers.
>
> Retain the 12,000 MiB launch floor, 13.0 GiB watchdog, resumable handovers and terminal-stop protections. Never signal or modify a co-tenant process.
>
> This is best-effort memory sharing. It does not guarantee protection against sudden allocations or compute contention.

Exact predicate, after validated integer sampling:

```bash
# sample_ok=1 only after successful telemetry and ownership accounting.
# base_foreign_mib is fixed for this worker lifetime.
if (( sample_ok != 1 ||
      foreign_mib > base_foreign_mib + 2000 )); then
    # Execute the existing own-worker graceful checkpoint/stop block.
fi
```

**Do not retain any assignment that absorbs newcomers into `base_foreign_mib`.** Predicate checks passed for +356, +2,000, +2,001, six ×356, departures and telemetry failure. The supplied script ends before its monitoring loop; integration is **not verified**. Deploy through a versioned supervisor handover, not an in-place edit of running Bash.

**Q5 — Adversarial checks before adoption:**

| Failure | Revealing number | Cheapest check |
|---|---|---|
| Allocation outruns polling | 6,144 requested >≈5,520 free | CPU event-sequence simulation; never force a co-tenant OOM |
| Baseline hides cumulative demand | 6×356 = 2,136 MiB | Replay arrivals; sixth must trigger |
| Launch cushion is smaller than growth threshold | 12,000−10,230 = 1,770 MiB | Log actual free and whole-process memory |
| Small allocation masks heavy compute | 356 MiB does not bound SM utilisation | Read-only per-process utilisation sampling |
| Replacement supervisor still runs old policy | Wrong policy version/poll timestamps | Inspect active unit, process and logs |

**Owner question:** “Adopt §8u’s fixed-baseline memory rule, accepting that it reduces unnecessary restarts but cannot guarantee against sudden-allocation OOM or compute slowdown?”
