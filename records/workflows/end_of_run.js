// Phase-4 end-of-run workflow — REVIEW COPY.
//
// Owner, 2026-09-13: "to do these things properly, you should create a dynamic
// workflow, ultra think, ultra code. And before that dynamic workflow, just
// before that, reconcile with Codex Council." Council AA (2026-09-13) ran and
// gave the delegation table this script encodes. This file is the reviewable
// record; at run time the same text is passed inline to the Workflow tool.
//
// HARD RULES (council AA Q6, owner standing rules):
//   * Every agent() is opus at medium effort. Never the session model.
//   * Subagents COMPUTE and REVIEW. They never write a canonical record, never
//     touch the sealed set, never receive an override env var, never publish,
//     never push, never kill a process. Those stay in the MAIN LOOP under the
//     human, and this script only *prepares* and *checks* them.
//   * The script refuses to start unless the main loop has already confirmed a
//     NORMAL ending and minted experiments/p4/qualified_selection.json with
//     tools/p4/p4_qualify.py. That mint is the gate, and it is not delegated.
//   * Nothing in here decodes a protected panel or the sealed FLEURS set.
//
// What the main loop does BEFORE this script:
//   1. supervisor.log shows COMPLETE (final checkpoint + final validation).
//   2. tools/p4/p4_qualify.py  -> qualified_selection.json (select, qualify,
//      export .nemo with exact dev-WER reproduction, atomic mint).
//   3. The owner's pending questions are answered (licence, MaCro record,
//      repo reuse-vs-separate, card scope).
// What the main loop does AFTER this script, using its outputs:
//   4. p4_final_test.py with P4_FINAL_TEST_SECOND_EVENT=1   (event two)
//   5. p4_panel_eval.py on the one-shot panels                (gated on 2 + 4)
//   6. card -> HF -> seven-system benchmark on the A5000 with a reviewed
//      launcher (never run_all_benchmarks.sh unchanged: pkill -f lines)
//   7. final pushes.

export const meta = {
  name: 'p4-end-of-run',
  description: 'Independently recompute, review and draft every Phase-4 end-of-run step; execution of sealed/protected/publishing steps stays in the main loop',
  phases: [
    { title: 'Gate',     detail: 'refuse unless a qualified normal ending is already minted' },
    { title: 'Verify',   detail: 'independent recomputation of selection, qualification, export parity, 8v window' },
    { title: 'Review',   detail: 'adversarial review of the exact event-two and panel invocations before the human runs them' },
    { title: 'Draft',    detail: 'model card, disclosure text, benchmark launcher, phase report skeleton' },
  ],
}

const OPUS = { model: 'opus', effort: 'medium' }
const REPO = '/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'
const RUN  = `${REPO}/experiments/p4/runs/N_O+macro_filtered_H120k`

const VERDICT = {
  type: 'object',
  properties: {
    ok:        { type: 'boolean' },
    summary:   { type: 'string' },
    evidence:  { type: 'array', items: { type: 'string' } },
    problems:  { type: 'array', items: { type: 'string' } },
  },
  required: ['ok', 'summary', 'evidence', 'problems'],
}

const READ_ONLY = `You are READ-ONLY. Do not modify, create, delete or rename any file under ${REPO}. Do not run any GPU decode. Do not set or export any environment variable beginning with P4_. Do not signal any process. If a check needs something you are not allowed to do, report that as a problem instead of doing it.`

// ---------------------------------------------------------------------------
phase('Gate')
// The gate is a plain read, done by a subagent so the main loop's own belief is
// checked by a second pair of eyes; but the DECISION to proceed is the script's,
// and the script cannot mint anything.
const gate = await agent(`${READ_ONLY}
Confirm ALL of the following from the files, quoting the exact lines/values:
1. ${RUN}/supervisor.log contains a line "COMPLETE at update 120000" (final checkpoint and validation both present).
2. ${RUN}/val.jsonl has a record with "update": 120000.
3. ${REPO}/experiments/p4/qualified_selection.json exists, with "ending": "normal", "qualified": true, a "selected_checkpoint" ending in .nemo, and "selected_sha256". Report selected_update and selected_S_pp.
4. The .nemo named in (3) exists and its sha256 (compute it with sha256sum) equals selected_sha256.
5. ${REPO}/experiments/p4/final_test_event.json exists and its "ckpt_sha256" is NOT equal to selected_sha256 (it must be the EARLY event, step_88000, not a second one). If it IS equal, event two has already run: say so loudly.
Return ok=true only if 1-5 all hold.`, { ...OPUS, label: 'gate', phase: 'Gate', schema: VERDICT })
if (!gate.ok) {
  log(`GATE FAILED: ${gate.summary}`)
  return { gate, aborted: 'gate' }
}
log(`gate passed: ${gate.summary}`)

// ---------------------------------------------------------------------------
phase('Verify')
const VERIFIERS = [
  { key: 'selection', prompt: `${READ_ONLY}
Independently recompute the Phase-4 selection from ${RUN}/val.jsonl using tools/p4/p4_common.py: build (S_exact-or-S_pp, update, is_eligible) for every record, call select_best, and compare to selected_update / selected_S_pp in ${REPO}/experiments/p4/qualified_selection.json. Also list every checkpoint inside the 0.05 pp tie band of the best eligible S. ok=true only if your recomputation matches the minted record exactly. Do not re-implement select_best; import and call it.` },
  { key: 'qualification', prompt: `${READ_ONLY}
Recompute SPEC 8w.2 item 4 qualification for the selected update in ${REPO}/experiments/p4/qualified_selection.json from the matching ${RUN}/val.jsonl record: S <= 22.0296, panel-D wer_frozen*100 <= 21.348, guards.pass true, is_eligible true, with 1e-9 pp tolerance. Quote the actual S and D. ok=true only if all four hold AND the minted "qualified" flag agrees.` },
  { key: 'export-parity', prompt: `${READ_ONLY}
Check the "export_verified" block in ${REPO}/experiments/p4/qualified_selection.json: recorded_wer_frozen must equal exported_wer_frozen to 1e-6, match must be true, n must be 5385. Then confirm source_pt_sha256 matches sha256sum of source_pt on disk, and selected_sha256 matches sha256sum of selected_checkpoint. ok=true only if all hold. Do NOT run any decode yourself.` },
  { key: 'final-window-8v', prompt: `${READ_ONLY}
Compute the SPEC 8v final-window report from ${RUN}/val.jsonl: (a) best eligible S through update 110000, (b) best eligible S through 120000, (c) gain = a - b; (d) the same for the TIE-SELECTED checkpoint (select_best truncated at 110000 vs at 120000). Threshold: a gain >= 0.20 pp undermines convergence. Also report D and conversational WER at the checkpoints involved. Return the numbers in evidence and state plainly whether the threshold tripped. ok=true means "computed", not "tripped".` },
  { key: 'guards-history', prompt: `${READ_ONLY}
From ${RUN}/val.jsonl list every validation where guards.pass was false and which check failed, and confirm the selected update is not among them. Also confirm empties at the selected update and its empty_rate <= 0.001. ok=true if the selected checkpoint has clean guards.` },
]
const verified = await parallel(VERIFIERS.map(v => () =>
  agent(v.prompt, { ...OPUS, label: `verify:${v.key}`, phase: 'Verify', schema: VERDICT })
    .then(r => ({ key: v.key, ...r }))))
const vfail = verified.filter(Boolean).filter(r => !r.ok)
if (vfail.length) {
  log(`VERIFY FAILED: ${vfail.map(f => f.key + ': ' + f.summary).join(' | ')}`)
  return { gate, verified, aborted: 'verify' }
}
log('all five independent verifications passed')

// ---------------------------------------------------------------------------
phase('Review')
// The human will run these two commands. The reviewers see the EXACT
// invocation and attack it. They do not run it.
const EVENT_TWO_CMD = `cd ${REPO} && CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 P4_FINAL_TEST_SECOND_EVENT=1 nice -n 19 .venv/bin/python tools/p4/p4_final_test.py --ckpt <selected_checkpoint from qualified_selection.json> --out-dir experiments/p4/final_test --device cuda`
const PANELS_CMD    = `cd ${REPO} && CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 nice -n 19 .venv/bin/python tools/p4/p4_panel_eval.py --ckpt <selected_checkpoint> --panels holdout vaani_validation vaani_test spring_validation spring_eval --save-hyps`
const reviews = await parallel([
  () => agent(`${READ_ONLY}
Adversarially review this command the human is about to run for the SECOND sealed FLEURS decode (SPEC 8w.14 makes it event two):
${EVENT_TWO_CMD}
Read tools/p4/p4_final_test.py. Confirm: (1) require_qualified_selection will accept ONLY the .nemo named in qualified_selection.json and its sha; (2) P4_FINAL_TEST_SECOND_EVENT=1 is the documented and ONLY way past the existing event registry, and what it does to experiments/p4/final_test_event.json (does it PRESERVE the early record? where?); (3) P4_ALLOW_UNQUALIFIED_FINAL is NOT set; (4) the baseline gates (P2 16.54, P1 19.672) will pass; (5) --out-dir does not collide with the EARLY bundle experiments/p4/final_test_EARLY_step88000. List anything that would make this decode unattributable, duplicated, or silently overwrite the early event. ok=true only if you would let it run as written.`,
    { ...OPUS, label: 'review:event-two', phase: 'Review', schema: VERDICT }),
  () => agent(`${READ_ONLY}
Adversarially review this command the human is about to run for the one-shot protected panels:
${PANELS_CMD}
Read tools/p4/p4_panel_eval.py, especially authorise_protected_panels and the --panels handling. Confirm: (1) the panel names/manifests actually resolve to holdout_corpus, Vaani validation+test and SPRING validation+eval; (2) the gate will REQUIRE qualified_selection.json and bind the event's ckpt_sha256 to selected_sha256 — which means it must run AFTER event two, not before; (3) P4_ALLOW_PROTECTED_PANEL is NOT set; (4) each panel is decoded exactly once by this command; (5) experiments/p4/protected_panel_decodes.jsonl is EMPTY before it runs. If the --panels names are not the ones the tool accepts, say exactly what they should be. ok=true only if you would let it run as written, after event two.`,
    { ...OPUS, label: 'review:panels', phase: 'Review', schema: VERDICT }),
])
const rfail = reviews.filter(Boolean).filter(r => !r.ok)
if (rfail.length) log(`REVIEW raised problems: ${rfail.map(f => f.summary).join(' | ')} — main loop must resolve before running`)

// ---------------------------------------------------------------------------
phase('Draft')
const drafts = await parallel([
  () => agent(`${READ_ONLY}
Draft the Hugging Face model card for the Phase-4 model as MARKDOWN TEXT in your reply (do not write files). Owner's instruction: minimal — "just dataset, benchmark and output", no hours, no GPU, no method. Council AA Q4 obligations that must still appear: CC-BY-SA-4.0 licence line (NOT NC — the owner reversed it); attribution for every source with its licence (MaCro CC-BY-4.0 IF the record in experiments/p4/licences/MACRO_BengaliAI_licence_OPEN_ITEM.md has been closed, otherwise leave a clearly marked placeholder); OpenSLR-53 CC-BY-SA lineage; BOTH FLEURS events (early step_88000 17.5441 WER / 5.2704 CER, unqualified interim; and the final one with <placeholders> for the numbers); the registered success verdict vs Phase 2 16.54 with <placeholder>; a link line for the seven-panel results. Read experiments/p4/SPEC.md sections 6, 8w.14 and 8w.15 for the facts. Keep it under 40 lines.`,
    { ...OPUS, label: 'draft:card', phase: 'Draft' }),
  () => agent(`${READ_ONLY}
Read /mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/run_all_benchmarks.sh and bench_server.py. Council AA Q6 forbids running it unchanged on this shared box because of the 'pkill -f "$PY bench_server.py"' lines (36-40, 74-78) and because it decodes eval_fleurs_bn again (a THIRD FLEURS exposure). Draft, as TEXT in your reply, a replacement launcher run_all_benchmarks_a5000.sh that: (1) pins CUDA_VISIBLE_DEVICES to the A5000 UUID GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567; (2) starts each bench_server as a child it owns, records the PID, and kills ONLY that PID after checking ps shows it is bench_server.py owned by the current user — never pkill -f; (3) checks the port is free before binding and refuses if another user holds it; (4) adds Phase 4 as one MODELS line pointing at <selected .nemo>; (5) runs the Qwen3 adapter through its own backend as before; (6) writes a prospective note that this run is a post-release reproducibility measurement, not selection. Preserve the benchmark protocol otherwise. Do not write the file.`,
    { ...OPUS, label: 'draft:launcher', phase: 'Draft' }),
  () => agent(`${READ_ONLY}
Draft the Phase-4 completion report skeleton as MARKDOWN in your reply: the 8v final-window numbers (take them from experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl exactly as the 'final-window-8v' verifier would), the selection and qualification facts from experiments/p4/qualified_selection.json, the two-event disclosure from SPEC 8w.14, placeholders for event-two and panel numbers, the seven-source corpus table from experiments/p4/corpus/build_report.json, and the open items list: MaCro licence record, licence choice, repo reuse-vs-separate. Under 80 lines.`,
    { ...OPUS, label: 'draft:report', phase: 'Draft' }),
])

return {
  gate, verified, reviews, drafts: { card: drafts[0], launcher: drafts[1], report: drafts[2] },
  next_for_main_loop: [
    'resolve any Review problems',
    'run EVENT TWO exactly as reviewed, in the main loop',
    'run the one-shot panels exactly as reviewed, AFTER event two',
    'fill the card and report placeholders from the real numbers',
    'owner decisions: licence, MaCro record, repo, then HF publish (main loop)',
    'benchmark x7 with the reviewed launcher (main loop owns processes)',
    'final pushes',
  ],
}
