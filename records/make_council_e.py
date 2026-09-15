#!/usr/bin/env python3
"""Compose the council-E (pre-launch) brief from the frozen Gate-0 artefacts and write /tmp/council_p4e/prompt.md."""
import json, subprocess
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
def J(p):
    try: return json.load(open(R / p, encoding='utf-8'))
    except Exception as e: return {"_missing": str(p), "_err": str(e)}
b = J('experiments/p4/corpus/build_report.json'); ma = J('experiments/p4/audit/machine_audit.json')
fp = J('data4/audit/fp_match_report.json'); lk = J('data4/audit/leakage_report.json')
cc = J('experiments/p4/cross_card_check.json'); rp = J('experiments/p4/replay_1000_check.json'); bl = J('experiments/p4/baseline_p1.json')
freeze = (R / 'experiments/p4/corpus/FREEZE.sha256').read_text() if (R / 'experiments/p4/corpus/FREEZE.sha256').exists() else 'MISSING'
spec = (R / 'experiments/p4/SPEC.md').read_text()
def slim(d, keys): return {k: d.get(k) for k in keys if k in d}
facts = {
 "arms": {k: slim(v, ["rows", "hours", "by_source"]) for k, v in b.get("arms", {}).items()},
 "g09a": b.get("g09a"), "g09b": b.get("g09b"), "panel_M": b.get("panels", {}).get("M"), "admission": b.get("admission"),
 "removed": b.get("removed"), "fingerprint_in_build": b.get("fingerprint"), "respin_cap": b.get("respin_cap"),
 "fingerprint_report": slim(fp, ["generated", "thresholds", "panel_clips", "train_clips_queried", "confirmed", "unresolved", "wall_s"]),
 "leakage_report": lk if isinstance(lk, dict) and "_missing" not in lk else lk,
 "machine_audit": {"baseline": ma.get("old_corpus_baseline"), "summary": ma.get("summary"), "verdicts": ma.get("verdicts")},
 "cross_card": {k: v for k, v in cc.items() if k != "blocks"}, "replay_1000": {k: v for k, v in rp.items() if k != "blocks"},
 "p1_baselines": {k: (v if k in ("S_pp",) else slim(v, ["n", "wer_frozen", "cer_frozen", "wer_nemo", "empty_rate", "per_source", "conversational"]) if isinstance(v, dict) else v) for k, v in bl.items() if k in ("S_pp", "D", "M", "K", "R")},
 "throughput_steps_per_h": {"A5000_from_cache": 891, "A6000_from_cache": 887, "pretokenize_false": 863},
 "freeze_sha256_head": freeze[:1200],
}
q = """
# QUESTIONS (final pre-launch reconciliation; answer with picks, numbers, exact SPEC wording where needed)
Q1 Gate-0 closure: given the frozen artefacts above, list anything that still blocks launching (a) the two O controls, (b) the three bucket arms, (c) N_final. Be specific; say CLOSED if nothing does.
Q2 The G0.6 human audit ended as "references OK" after the owner realised the board's Phase-1 hypothesis hint had been judged instead of the reference (12/12 flagged MaCro clips re-transcribe to the reference with hishab). Is that an acceptable closure of G0.6 together with the machine consensus audit (rule registered before results), or must a hypothesis-free re-audit run first? Pick; if re-audit, size it.
Q3 Machine audit caveats: hishab scores ~4.5 % WER on MaCro/RESPIN (has trained on them) and Phase 1 trained on the old corpus (baseline biased low). Does either invalidate the pass verdicts? What exact card wording?
Q4 RESPIN retained only 149 h under the sentence×dialect cap (each dialect has its own prompt inventory). Keep as registered (no re-raise) — confirm, and state whether O+madasr at 149 h is still a meaningful arm or should be folded into N_final only.
Q5 Cross-card: ids/LR identical, 20-update loss blocks differ ≤1.58 % between cards while same-card runs already drift ≤1.23 % after update 285 (CUDA nondeterminism). Confirm the ruling (8k control agreement is the binding test) or amend the numbers.
Q6 Vaani references are verbatim-ish spontaneous speech (some omit false starts that both models hear). Any registered treatment beyond "its arm decides"? Expected sign on D/FLEURS unchanged?
Q7 Old OpenSLR-53 shows 7.4 % consensus disagreement. Confirm "O stays frozen; disclose" for this phase, or propose a registered, non-ASR-ranked remedy.
Q8 Launch sequence and concurrency on two cards with these exact runs: O_A5000, O_A6000, O+macro_filtered, O+madasr, O+vaani, N_final; which arm goes to which card in which wave; the exact admission test list at 8k with m=3; and the STOP conditions during pilots. Output a table.
"""
prompt = ("You are the final pre-launch reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A–D shaped the contract; "
          "Gate 0 has now been executed. Below: the CONTRACT, then the FROZEN FACTS (JSON, measured), then QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FROZEN FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4e').mkdir(exist_ok=True); Path('/tmp/council_p4e/prompt.md').write_text(prompt)
print(len(prompt.split()), 'words')
