**Reject §8w.2a, including its automatic 1% empty-output stop. Continue the existing training state, subject to the registered integrity, numerical, resource and budget rules. Fix the restart and finalization defects below.**

I inspected the local implementation and exercised six isolated counterexamples using its actual functions with synthetic, in-memory inputs. I changed no files, touched no GPU job and opened no held evaluation data.

The absence of §5 aborts for N is consistent with §8t.3, which explicitly excludes them. It does not overturn my recommendation. **You are wrong to equate “no early performance termination” with “unwatched.”** I should, however, have made clearer in P that requirements such as “checkpoint-bound,” “unrounded” and “idempotent” were requirements—not verified implementation properties.

**Q1. The floor rejects demonstrated intervening improvement.**

Consider this **hypothetical continuation**, compatible with the supplied observations:

| Update | Frozen D WER |
|---:|---:|
| 6,000 | 29.012% — observed |
| 16,000 | 22.000% |
| 18,000 | **21.000%** |
| 20,000 | 21.600% |
| 30,000 | 20.300% |
| 38,000 | **19.400%** |
| 40,000 | **21.050%** |
| 120,000 | 18.400% |

Your floor terminates at 40k because **21.050 ≥ 21.000**. Yet:

- The run improved **1.600 pp beyond its reference minimum** by 38k.
- The terminating observation is only **0.050 pp above** that reference.
- If S40 = 20.800, empties = 0 and the guards pass, **40k satisfies the original absolute screen**.
- A qualifying eligible checkpoint can already exist at 38k.

Thus the rule can kill a run that already produced a qualifying model, before considering any hypothetical later recovery or held-domain improvement.

The example requires a **1.650 pp rebound from 38k to 40k**. That is not demonstrated for this N trajectory, and I cannot assign its probability. It is not an implausible construction that requires a new architecture or LR change. At 40k, the registered LR is still approximately **3.8495e-4, or 38.5× terminal**; another 80,000 updates and substantial annealing remain. No supplied evidence establishes that such a rebound cannot occur.

An even simpler failure needs no rebound: a model reaches D = 18.000%, remains there at 40k, and satisfies every other criterion. **Exact equality terminates an already useful model.** Stopped improvement and inadequate quality are different conditions.

The opposite failure is equally concrete. I counted **55,191 reference words** in frozen D. One fewer edit improves corpus WER by:

\[
100/55{,}191 = 0.00181189\text{ pp}.
\]

A trajectory whose best remains approximately 29.012% through 20k can pass at approximately **29.010%**, with zero empties, after gaining just **one edit**. It could then remain poor through 120k. Passing does not establish useful learning; failure does not establish its absence.

The empty-output clause also needs honest characterization:

- **53/5,385 = 0.9842%:** passes.
- **54/5,385 = 1.0028%:** terminates.

One additional empty utterance decides the entire trajectory. That is a chosen spending threshold, not a diagnosis of decoder failure.

Your actual [termination message](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:949) says:

> “so more of the same will not rescue it”

**That contradicts your stated disclaimer and is unsupported.** The remaining LR trajectory is also not “more of the same.”

**Q2. Changing the reference does not repair the inference.**

For a fixed history:

\[
\min D_{\le30k}\le\min D_{\le20k}\le\min D_{\le12k}.
\]

Therefore 30k makes the floor tighter; 12k makes it looser. Neither supplies calibration.

A simple mathematical illustration exposes the minimum’s effect. Assume stationary, exchangeable, continuous readings—**an illustration, not a model fitted to this run**. A new reading beats the minimum of \(m\) previous readings with probability \(1/(m+1)\):

| Reference | Earlier readings | Probability of passing |
|---|---:|---:|
| Best through 12k | 6 | 14.29% |
| Best through 20k | 10 | 9.09% |
| Best through 30k | 15 | 6.25% |
| Literal 20k | 1 | 50.00% |

Those rejection rates apply equally to a stationary model at **18%** or **29%** WER. The floor contains no notion of sufficient quality.

Requiring **another 0.5 pp improvement** tightens the criterion further; it does not compensate for selecting a favourable minimum. Compensation would go in the opposite direction—allowing some positive deterioration above the minimum. **There is no defensible calibrated allowance in the supplied evidence.**

My exact rule is **no terminal early-performance comparison**. For the requested 40k warning, use:

\[
\Delta_{40}=
\operatorname{median}(D_{36k},D_{38k},D_{40k})
-\operatorname{median}(D_{16k},D_{18k},D_{20k}).
\]

Flag \(\Delta_{40}\ge-10^{-9}\) pp for engineering review; it creates no STOP.

Three observations give equal comparison opportunities and span **4,000 updates, approximately 0.929 nominal epoch**, while reducing dependence on one extreme observation. This is a descriptive warning, not a calibrated futility test. It does not perfectly match epoch phase.

Also withdraw **“it contains no chosen constant.”** You chose 20k, 40k, D, a minimum, zero improvement tolerance and 1% empties. Those choices need justification even though the WER reference is data-dependent.

**Q3. Yes: retain no automatic early performance stop.**

The run remains watched by:

- Existing validations every 2,000 updates: **40 further validations after 40k through 120k**.
- Source and conversational guards, eligibility, empty outputs, error decomposition and domain-drift reports.
- Per-update finite-loss/gradient checks and resource monitoring.
- Checkpoint, sampler, policy and data-integrity checks—once their actual gaps below are fixed.
- The aggregate **350 GPU-hour ceiling**.

At 40k, an apparent flatline triggers investigation of actual optimizer updates, gradients, train/eval mode, target/tokenizer identity, sampler exposure, decoder behavior and evaluation integrity.

**If those checks pass and the issue is simply flat development WER, continue the registered annealing trajectory.** If the owner elects to stop spending, record that as a discretionary spending amendment—not proof that the model cannot recover, and not an automatic “learning floor.”

At the supplied **868 updates/hour**:

- 40k→120k costs **92.17 training GPU-hours**.
- 6k→120k costs **131.34 hours**, not 92–98.
- From update 7,212, approximately **129.94 training hours** remain, excluding validation overhead.

That cost is real. “I am unwilling to spend it” is an honest budget position. “This comparison proves learning stopped” is not.

**Q4. The amendment is adaptive, but not self-fulfilling or held-test leakage.**

The three early validations influenced the rule’s design. The future reference period is unfinished, so they do not guarantee passage.

In particular, **“the run clears it today, at 6k against 2k” is invalid**: that substitutes a different reference for the registered minimum through 20k.

This is data-informed development policy, not contamination of sealed evaluation. The problem is misrepresenting its evidential status and accepting further policy changes selectively after seeing their consequences.

The current disclosure is incomplete because it describes diagnostic N40 without adequately describing the added terminal condition. Use:

> “After observing this run’s 2k, 4k and 6k development evaluations, the owner adopted diagnostic reporting of the former N40 screen and qualification after the completed horizon. During implementation, the orchestrator additionally introduced a terminal N40 rule comparing literal D40 with the minimum frozen D through 20k, together with an empty-output threshold above 1%. Council P had not endorsed that addition. These are data-informed prospective policy changes, not an unchanged preregistration or a validated test of recoverability. Council Q subsequently recommended removing the added performance stop. Any withdrawal is recorded with its effective update, policy identity and deployment evidence. Held evaluation data were not consulted.”

Your addition’s deployment and its eventual removal must both remain in the record.

**Q5. The restart bug class is broader—and I found concrete instances.**

The governing distinction is: **freeze experiment identity and authorized policy; recheck mutable execution state immediately before acting.** Currently the implementation sometimes does the reverse.

1. **Resume identity is substantially weaker than the contract claims.**  
   [check_resume_state](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:652) checks only horizon, init label and effective batch. I reproduced acceptance despite differing saved/current corpus and tokenizer configuration. There are separate tokenizer checks during initial loading, so this isolated result is not proof that every changed tokenizer passes the complete startup path. It does establish that the resume validator does not bind the checkpoint to those identities.

   A changed target with unchanged cut IDs can also pass the sample-ID handshake. A different N checkpoint with compatible shapes and the same three checked fields is not sufficient evidence of this run’s state.

   **Fix:** bind checkpoints to immutable run, corpus/content, tokenizer, initialization/head, recipe and policy identities. Record each startup separately; overwriting `run_meta.json` is not an immutable execution history.

2. **“Config wins” permits unrestricted policy migration.**  
   [Resume currently accepts the current config’s policy](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:1483), printing a migration message. Printing is not authorization. Removing the config key restores the default terminal policy; changing code underneath `diagnostic` changes whether it includes your floor.

   **Fix:** explicit versioned policies and an authorized old→new migration record. Permit the specific adopted change; refuse unrelated differences. A single word, `diagnostic`, cannot identify both P’s policy and P-plus-floor.

3. **Milestone and validation evidence is not checkpoint-bound.**  
   [_n_gate_recorded](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:811) accepts a result string. My synthetic record had `result="missed"`, **update 17**, an unrelated run, the wrong checkpoint hash and `floor_pass=false`: **accepted**.

   [existing_val_record](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:753) accepts the unique record matching the update number, without checking model identity. I reproduced reuse of an explicitly unrelated checkpoint’s record.

   The new floor’s history reader additionally skips malformed JSON and accepts missing/negative update identifiers; a synthetic **update −1, D=0** poisoned its minimum to zero.

   **Fix:** validated identities, exact panel coverage, finite values, expected scheduled updates, duplicate rejection and atomic evidence commits. Persist the complete milestone outcome **before** terminal exit; the floor currently exits before writing its milestone record.

4. **The supervisor confuses a logged update with completed work.**  
   [Completion checks](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_supervised.sh:114) use the last `steps.jsonl` update. The trainer writes that record before checkpointing and validating.

   Concrete sequence: log update **120,000** → process dies before the checkpoint/validation → supervisor reports COMPLETE because the log reached its target. Owed finalization is never recovered automatically.

   Also, `--steps` is independent of the configured horizon. A stale unit requesting **150,000** with H=120,000 can execute **30,000 extra terminal-LR updates**, approximately **34.56 hours** at 868/hour.

   **Fix:** explicit production target=H and a durable completion state requiring the checkpoint, final validation, prescribed selection and qualification. “Update reached H” is not normal-ending authorization.

5. **The fresh-start guard does not close every entry point or race.**  
   [fresh_start_refusal](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:1345) checks `ckpt/step_*` and nonempty `steps.jsonl`. I reproduced acceptance with validation/milestone evidence present but those two absent. It does not itself enforce terminal STOP/ABORT status.

   Per-card supervisor locks also do not provide an exclusive lock on the **run directory**. Two launchers can both pass a fresh-directory check, or resume the same run on different cards.

   **Fix:** a process-lifetime run lock acquired before inspection or mutation; terminal-status checks in the trainer itself; fresh-run refusal on any prior authoritative run evidence. The supervisor must not be the only protection against a direct unit restart.

6. **The supposedly frozen score can change when a file disappears.**  
   [validate](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_train.py:713) silently omits missing panel files and chooses the score weights using `"R" in res`.

   With D=20, M=18, R=33, K=15:

   - Registered weights produce **S=20.4**.
   - Missing R switches the weights and produces **S=18.9**.

   **That is a fictitious 1.5 pp improvement with no model change.** R’s training bucket being absent from production does not authorize removing the registered selection panel.

   **Fix:** bind the weight vector and all panel identities prospectively; a missing required panel is an integrity failure.

7. **“Unrounded” decisions actually consume rounded metrics.**  
   [ctc_panel_eval](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_common.py:486) rounds WER fractions to five decimal places before selection and guards. That is a **0.001 pp grid**, with up to **0.0005 pp rounding error per panel**. S is subsequently rounded too.

   A **1e-9 pp comparison tolerance cannot restore discarded precision**.

   **Fix:** persist integer edit/reference counts and derive decision values without display rounding. Reconstruct existing development results from preserved predictions where possible, retaining the original records and reporting any changed decisions. Do not silently change thresholds.

8. **A surviving dispatcher still contains destructive launch logic.**  
   [run_post_union_N.sh](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/experiments/p4/run_post_union_N.sh:62) still executes:

   ```bash
   rm -rf experiments/p4/runs/N_$CORPUS
   ```

   and launches a **150,000-update** supervisor. This targets its legacy run name, not automatically the current H120k directory; nevertheless, reactivation can delete historical production state and launch an unauthorized procedure.

   **Fix:** permanently disable superseded dispatch actions and replace destructive initialization with exclusive creation/refusal. I could not inspect live unit state because access to the user systemd bus was denied, so I cannot certify whether this dispatcher is armed.

9. **The one-shot evaluator can mix checkpoints across a restart.**  
   [decode_items](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_final_test.py:264) resumes predictions by filename alone. Checkpoint identity is recorded later in the final report.

   Concrete sequence: decode **700 recordings with checkpoint X**, crash, restart the same output directory with checkpoint Y. The tool can retain X’s 700 predictions, decode Y’s remaining **220**, and label the report with Y’s checkpoint hash.

   Its completion marker is also output-directory-local, and [the refusal text explicitly suggests another directory](/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/tools/p4/p4_final_test.py:375). That does not enforce one Phase-4 event. The final entry point does not require the new qualified normal-ending artifact.

   **Fix before any held decode:** one phase-level event record, exclusive evaluation lock, immutable selected-checkpoint/decoder/scorer/panel identities recorded before the first prediction, and identical identities required on resume. A new output directory must not create another authorization.

These findings do **not** establish that the current trajectory has been corrupted. They establish that the stated guarantees are not implemented comprehensively. **139 passing checks cannot certify behaviors those checks never challenge.**

**Q6. The strongest case against continuing is about future value, not money already spent.**

At update 7,212, the decision commits approximately **130 more training hours**, plus evaluation overhead, to a single procedure whose:

- Corpus admission was conditioned on another initialization.
- Added MaCro supplies approximately **50.10% of presentations** and was selected upstream for ASR agreement.
- Development gains in its admission pilot were predominantly in-domain.
- Final qualification thresholds are permissive and do not establish broad improvement.
- Execution machinery still has consequential integrity gaps.

The strongest economic argument is that those remaining resources might produce more value elsewhere, while this run cannot answer the deferred initialization comparison. **Neither the early WER decrease nor already-spent GPU-hours establishes that continuation has the highest expected value.**

My recommendation to continue is therefore conditional on making this a trustworthy execution of the authorized experiment. It is not a claim that success is likely enough to quantify, or that the current design is optimal.

For your requested **single measurement that would make me stop execution at 8k/12k/16k**, I choose:

> **Zero actual parameter displacement across every intended encoder and head parameter over 100 consecutive completed optimizer updates, despite nonzero gradients.**

Those updates claim **25,200 training presentations**. Exact zero applied change would establish an execution defect, rather than merely disappointing WER. I would checkpoint and stop immediately for diagnosis and repair; I would not wait for 40k or restart from zero.

If you require a **single early WER value that irreversibly abandons an otherwise correctly executing run**, I have no defensible number. Supplying one would repeat the mistake under review: presenting an uncalibrated spending preference as evidence of hopelessness.
