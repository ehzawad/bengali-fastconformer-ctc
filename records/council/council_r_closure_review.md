**The rejected patches should not ship. But rejecting those implementations does not close the defects or justify leaving them indefinitely. Your closure report overstates what is fixed.**

I inspected the current source and deployed transient-unit file, and exercised actual functions with synthetic, memory-only inputs. I changed no files, touched no GPU job and opened no held evaluation data.

**Q1. Measuring #7 was useful; treating measurement as the remedy was wrong.**

The four reconstructed validations show no changed decision. All four are ineligible, however, so the current tie-selection comparison has **zero eligible candidates**. The 5.662 pp qualification margin says nothing about future differences between checkpoints.

Here is a **verified counterexample using the actual `selection_score`, `select_best` and `retain` functions**, with your frozen reference-word denominators:

| Hypothetical update | Edits: D / M / R / K | Exact S | Recorded S |
|---|---|---:|---:|
| 100,000 | 9,986 / 3,009 / 4,000 / 3,300 | 18.6129335071 | 18.6131 |
| 102,000 | 9,950 / 3,000 / 4,000 / 3,300 | 18.5629982031 | 18.5627 |

Denominators are respectively **55,191 / 16,669 / 14,434 / 23,995**. These aggregate counts can coexist with passing source, conversational and empty-output guards.

- Exact difference: **0.049935304 pp**. The registered rule selects **100k**.
- Rounded difference: **0.050400000 pp**. The implementation selects **102k**.
- Add eligible 104k and 106k checkpoints with the same scores as 102k. The actual retention function **deletes every copy of 100k**.

**Preserved hypotheses cannot reconstruct deleted model weights.** That defeats your strongest reason for indefinite deferral.

Qualification can also flip. With edits **D=10,000, M=3,010, R=8,908, K=3,300**, the actual score function produces:

```text
Exact S:     22.029658047728063 → FAIL
Recorded S:  22.0296            → PASS
Threshold:   22.0296
```

The excess is **0.0000580477 pp**, comfortably greater than the registered `1e-9` tolerance.

Your reviewers were right about `11591/55191`: it rounds to **0.21002**. That example was false. The two examples above are executable replacements.

I also checked the narrower source-guard question. **With the current fixed denominators and registered baseline constants, five-decimal rounding does not flip any of the five source guards, the conversational guard or the absolute D qualification boundary.** Their nearest attainable edit counts fall on matching sides. For example, Kathbath’s 755 edits give **13.484550813%**, rounded to **13.485%**; both exceed **13.484%**. I should not imply that every conceivable boundary defect is realizable on these particular panels.

**Required action:** persist integer counts for panels, sources and conversation; derive S, eligibility, retention and selection without display rounding; use the same values in replay and finalization. Keep the registered baseline constants unchanged.

My operational deadline is **before the 10,000 validation**, deployed during a controlled checkpoint pause. An acceptable temporary bridge is preserving every subsequent validation checkpoint independently of rounded retention, then completing exact decision handling before any selection or pruning. Sixty validation checkpoints at your stated 1.39 GB each cost approximately **83.4 GB**.

Keep the existing empty-reference **row** handling. A registered panel or source having **zero total reference words** is different: it must block authoritative evaluation, not silently become a zero-WER pass.

**Q2. Rejecting the identity and policy designs was correct. Leaving unrestricted restarts available is not.**

Neither repair needs to await a special scientific milestone. Their safe deployment point is **a deliberately checkpointed maintenance pause, before the next authoritative restart**. From the packet’s update 8,400, update **8,500** is about **6.9 training minutes** away at 871 updates/hour. No restart from zero is warranted.

The smallest defensible implementation is:

- **#1 — one immutable experiment identity.** Bind the exact training-manifest content, tokenizer, initialization/head provenance, recipe, required panels/baseline and authorized policy. Store that identity in new checkpoints and compare it before loading resume state. For existing checkpoints, permit a specifically audited migration bound to the **exact checkpoint SHA-256**; do not bless arbitrary legacy checkpoints on first use.
- **Separate identity from location.** A run-directory name is not experiment identity. An authorized handoff is an explicit parent-checkpoint→successor relationship. My earlier wording was underspecified about this distinction. However, the pilot→A-production handoff is no longer authorized for the current NVIDIA-only procedure, so preserving that obsolete path is not a sufficient reason to defer protection here.
- **Record each startup separately.** Preserve checkpoint identity, executable/configuration identities, effective arguments and migration authorization. Do not overwrite the only execution record.
- **#2 — pin one explicit policy generation.** For this run, it must identify diagnostic N40, the withdrawn floor, diagnostic N60 and completed-horizon qualification. A missing key must not restore `terminal`. A bare legacy `diagnostic` must resolve through an audited checkpoint/code/configuration generation, never a “latest version” map.
- **Handle rejection before GPU/model loading.** Establish the intended run directory and error-recording mechanism before configuration validation. Acquire the run lock and enforce terminal-status rules before inspecting or mutating state.

These checks can be prepared in an external launch preflight while training continues. The replacement worker must actually invoke that preflight; editing a launcher pathname is insufficient.

Two proposed requirements should be rejected:

**“Soft treatment of recomputed digests” is wrong for a genuine content mismatch.** Distinguish an unavailable read from different content; neither authorizes silently accepting different data.

**An unrestricted `P4_IDENTITY_OVERRIDE=1` is not an audited migration.** Any exception must identify the specific mismatch, checkpoint and authorized transition. Preserve the evidence.

The additional 381 MB read is not a serious argument for leaving identity unchecked. At an illustrative 100 MB/s, it takes **3.81 seconds**; measure the actual cost.

There is already a related startup hole: configuration runs at [p4_train.py:1474](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:1474), before `_RUN_DIR` is assigned at line 1515. The outer exception handler therefore can report an ABORT **without writing a run-local ABORT**. That can requeue indefinitely. It fails before model loading, so “reloads the model every time” is inaccurate for this path.

**Q3. The import smoke is insufficient, and its deployment claim is false.**

The stored production worker unit contains:

```text
Restart=no
ExecStart=... p4_train.py ... --steps 120000 ... --config configs/p4_ctc_initN.yaml
```

**Its embedded command contains no import smoke.** The supervisor explicitly launches workers with `P4_RESTART=no`; the launcher’s default `on-failure` is not this worker’s deployed policy.

A newly created worker would receive the edited launcher body. A direct restart of the existing unit reuses its existing body.

Even when installed, the smoke establishes only that imports succeeded once:

- An unauthorized semantic change can import successfully.
- Configuration and resume checks are not exercised.
- Import and execution occur in separate Python processes against mutable files.
- A full disk can prevent the proposed ABORT file from being written.
- The instruction to “delete ABORT” discards evidence and bypasses the registered recovery process.

Failing closed is appropriate. **Turning every pre-execution import failure into an irrevocable experimental outcome is unnecessary.**

I would implement this launch protocol:

```text
Acquire the process-lifetime run lock.
Enforce existing terminal records and any specifically authorized resolution.
Verify the pinned executable release, config, data and resume identities.
Run an import preflight from that release, CPU-only, with a timeout.
On preflight failure:
    record LAUNCH_BLOCKED and stderr; execute zero training updates;
    retry transient failures under an explicit bounded retry policy;
    escalate persistent failures without deleting experimental records.
On success:
    recheck the authorized identities and exec that same release.
```

A concrete retry policy could allow **three attempts, 60 seconds apart**, then require intervention. That is an operational choice, not a scientific stopping rule. Existing terminal STOP/ABORT records remain binding.

Use a versioned executable release whose files are not edited in place, including the local modules it imports. Keep **one restarter**, with production workers using `Restart=no`. Verify the replacement unit’s actual command before declaring deployment complete.

**Q4. Several of the five earlier “fixed” findings remain incomplete.**

| Finding | What is fixed | What remains in the actual code |
|---|---|---|
| **#3 — evidence binding** | The obviously wrong milestone update is rejected. | A record with `update=40000`, `result="missed"` and **wrong run/checkpoint/policy identities still passes**. With no file, a bare `sel["gate_n40"]="missed"` passes. `existing_val_record()` still reuses an unrelated checkpoint’s record. |
| **#4 — completion** | `--steps > horizon` is refused. | The supervisor still declares completion from the last logged update. **Log 120k → crash before checkpoint/validation → false COMPLETE** remains. Production also does not require target=H, and no durable selected-checkpoint qualification authorizes completion. |
| **#5 — fresh starts/concurrency** | More existing evidence blocks fresh initialization. | `best/` alone is still missed. Supplying `--resume` bypasses this guard immediately, including terminal-record checks. There is still no process-lifetime run-directory lock. |
| **#6 — frozen panels** | A configured panel whose file disappears now fails; losing R no longer changes the weights. | Panel identities and full coverage remain unbound. Removing R from configuration yields undefined S rather than immediate refusal. A present but truncated/replaced panel is accepted. Source guards iterate only the sources present. |
| **#8 — dispatchers** | The named `run_post_union_N.sh` destructive production-directory removal is gone, and its normal entry is disabled. | The older **`run_post_union.sh` remains executable without that guard** and still launches the withdrawn control and a 150k production supervisor. I have not established that this older dispatcher is currently armed. |

Concrete checks I executed:

- Wrong-identity N40 record at the correct update: **accepted**.
- Wrong-checkpoint validation at update 12,000: **reused**.
- Two update-17 history records with identical D but conflicting S: **accepted by `_d_frozen_history`**.
- Fresh start with only `best/step_8000.pt`: **allowed**.
- Current metrics omitting four D sources: **guards passed**, provided the remaining source, conversational and empty checks passed.

Relevant locations: [milestone reader](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:880), [supervisor completion](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:114), [fresh-start guard](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:1425), [source guards](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_common.py:561), [older dispatcher](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_post_union.sh:66).

**Your additional “#9 CLOSED” claim also needs reopening.** The X→Y mismatch protection and cross-directory registry are substantive improvements. But:

- `--no-resume` bypasses shard reuse and decodes existing items again. I executed the actual function against an in-memory existing shard: **one existing row became two**, and the new prediction was returned.
- `open_event()` accepts an already **completed** event when identity and directory match. Combined with `--force --no-resume`, this permits another decode without requiring the second-event override.
- The entry point still does not require qualified normal completion and the prescribed selected checkpoint.
- Its identity covers FLEURS and baseline files, **not holdout and Vaani**. The separate panel evaluator exposes holdout without this authorization machinery.

See [decode resume handling](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_final_test.py:601), [event re-entry](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_final_test.py:488), and [final entry point](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_final_test.py:777).

Before held decoding, require qualified finalization at **every held entry point**, reject duplicate item identities, forbid `--no-resume` over existing final predictions, and refuse a completed event regardless of the output-directory marker unless a specific authorized exception exists.

**A tenth defect: an external watchdog stop becomes a resumable handover.**

The external memory watchdog calls `systemctl stop` after three breaches of **13.0 GiB**, but writes no durable terminal cause. The trainer receives SIGTERM and writes `reason="graceful_stop"`. The supervisor consequently treats it as resumable and clears that STOP before relaunch.

That contradicts the registered requirement that a watchdog stop needs an owner-authorized recovery.

The chain is visible at [watch_p4.sh:126](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/watch_p4.sh:126), [trainer signal handling](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:1711), and [supervisor classification](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:171).

Persist the external terminal cause **before signalling**, and make it take precedence over the trainer’s generic graceful-stop acknowledgement. Test that a fresh supervisor refuses restart without the authorized resolution. This is a reachable code path, not a claim that the external watchdog has already fired.

**Q5. The owner-readable status paragraph**

> The run has documented early learning, and the four reconstructed validations show no decisions changed by rounding. The trainer, common-module and configuration hashes on disk match the recorded startup hashes. We have not found evidence that the current trajectory is corrupted. However, execution is not yet certified: resume identity, policy authorization, checkpoint-bound evidence, exclusive launching, exact checkpoint retention and qualified finalization still have gaps. The new import guard is absent from the deployed worker command, and the final evaluator still permits repeat-decoding paths. Preserve the current training state, repair these controls during a checkpointed maintenance pause, and keep held evaluation blocked. Approximately 128.1 training hours remain from the supplied update 8,400 at 871 updates/hour, excluding validation overhead; money already spent is not a reason to commit them.

**Q6. The single most revealing closure error is the undeployed import guard.**

Read the [deployed worker unit](/run/user/1004/systemd/transient/worker-n-o-macro-filtered-h120k-a6000.service:19). The cheapest check is:

```bash
rg -n '^(Restart|ExecStart)|IMPORT SMOKE|import tools\.p4' \
  /run/user/1004/systemd/transient/worker-n-o-macro-filtered-h120k-a6000.service
```

Its stored command lacks the smoke and specifies **`Restart=no`**. I read this file directly because systemd bus access was denied.

This is the same execution-integrity mistake as before: **the proposed behavior was reviewed, the source was edited, and deployment was inferred.** The cheapest decisive check is inspecting what the next process will actually execute.

One further numerical correction: §8w.2a reverses my probability illustration. With ten previous stationary exchangeable continuous readings, **9.09% pass and 90.91% reject**—not 9.09% reject.
