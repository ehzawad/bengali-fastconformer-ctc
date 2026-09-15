Read both files, plus `run_all.sh`, `bench_run.py`, `backends.py`, and `run.sh` for context.

## Draft: `run_all_benchmarks_a5000.sh` (text only — not written to disk)

```bash
#!/usr/bin/env bash
# Post-release reproducibility measurement of the Bengali ASR checkpoints,
# pinned to the A5000 and safe to run on a shared box.
#
# Differences from run_all_benchmarks.sh, and why:
#   * CUDA_VISIBLE_DEVICES is pinned to the A5000 by UUID, so a co-tenant on the
#     A6000 cannot perturb timings and this run cannot land on their card.
#   * Each bench_server is started as a CHILD of this script. We remember its
#     PID and kill exactly that PID. There is no pkill anywhere: a pattern kill
#     on a shared box can reap another user's process (or this script's own
#     command line, which contains the pattern).
#   * The port is checked for a foreign owner BEFORE binding. If someone else
#     holds it we refuse rather than fight for it.
#   * Phase 4 is added as one MODELS line.
#   * Results go to a fresh output root so the released outputs/ and
#     outputs_a5000/ trees are never overwritten.
set -euo pipefail
cd "$(dirname "$0")"

PY="$PWD/venv/bin/python"
QWEN_PY="/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/.venv-qwen/bin/python"
HARNESS="${HARNESS:?set HARNESS=/path/to/scripts/benchmark.py}"
DATA="$PWD/eval_fleurs_bn"
PORT="${BENCH_PORT:-8018}"
CONC=10
OUT_ROOT="outputs_p4_repro"
ME="$(id -un)"

# A5000, by UUID. Never a numeric index: CUDA's index order is not nvidia-smi's.
A5000_UUID="GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567"
export CUDA_VISIBLE_DEVICES="$A5000_UUID"

# ---------------------------------------------------------------- prospective note
# Written BEFORE any decoding, so it is prospective rather than a post-hoc gloss.
mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/RUN_NOTE.md" <<'NOTE'
# Prospective note — read before interpreting these numbers

This run is a POST-RELEASE REPRODUCIBILITY MEASUREMENT, not a selection run.

* The Phase 4 checkpoint benchmarked here was already selected and frozen on
  its own held-out development data before this script was ever executed. No
  checkpoint, decoding setting, threshold or hyper-parameter is chosen, tuned
  or re-ranked on the basis of any number this run produces.
* This is the THIRD time this Phase 4 lineage touches FLEURS-bn
  (eval_fleurs_bn). FLEURS is therefore NOT a held-out test set for Phase 4 in
  this run. Numbers here may be optimistic and must be reported as a
  reproducibility check under a known-exposed evaluation set, never as a clean
  generalisation estimate or as evidence for choosing Phase 4 over anything
  else.
* The only admissible use of this output is: "re-running the published
  protocol on the published models reproduces the published numbers (or does
  not)." Any other claim from these numbers is unsupported.
* Baseline rows (the non-Phase-4 models) are re-decoded solely so the
  comparison is made under one GPU, one loader and one scorer on this date;
  their published values remain the numbers of record.
NOTE
echo "==> prospective note written to $OUT_ROOT/RUN_NOTE.md"

# ---------------------------------------------------------------- port safety
port_owner_pids() {   # PIDs *we* can see holding the port; empty if none/foreign
	ss -ltnpH "sport = :$PORT" 2>/dev/null \
		| grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
}

require_port_free() {
	ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . || return 0
	local pids; pids="$(port_owner_pids)"
	if [[ -z "$pids" ]]; then
		# Listener exists but ss shows no pid => it belongs to another user.
		echo "REFUSING: port $PORT is held by a process this user does not own." >&2
		echo "  Pick another port with BENCH_PORT=<n> $0, or wait for it to clear." >&2
		exit 1
	fi
	local p owner
	for p in $pids; do
		owner="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
		echo "REFUSING: port $PORT already held by pid $p (user ${owner:-unknown}):" >&2
		ps -o pid=,user=,args= -p "$p" >&2 || true
	done
	echo "  Not killing it. Stop it yourself, or set BENCH_PORT." >&2
	exit 1
}

# ---------------------------------------------------------------- owned-child kill
SERVER_PID=""

stop_server() {   # kill ONLY the pid we started, and only after proving what it is
	local pid="${SERVER_PID:-}"
	SERVER_PID=""
	[[ -n "$pid" ]] || return 0
	kill -0 "$pid" 2>/dev/null || return 0

	# Prove identity before signalling: right user, and really bench_server.py.
	# A pid can be recycled between start and stop; this is the check that makes
	# a targeted kill safe where a pattern kill never is.
	local line owner
	line="$(ps -o user=,args= -p "$pid" 2>/dev/null || true)"
	owner="$(awk '{print $1}' <<<"$line")"
	if [[ "$owner" != "$ME" || "$line" != *"bench_server.py"* ]]; then
		echo "WARNING: pid $pid is not my bench_server.py; leaving it alone:" >&2
		echo "  ${line:-<gone>}" >&2
		return 0
	fi

	kill -TERM "$pid" 2>/dev/null || true
	for _ in $(seq 1 30); do
		kill -0 "$pid" 2>/dev/null || break
		sleep 1
	done
	if kill -0 "$pid" 2>/dev/null; then
		# Re-prove identity before escalating; never SIGKILL on a stale assumption.
		line="$(ps -o user=,args= -p "$pid" 2>/dev/null || true)"
		if [[ "$line" == *"bench_server.py"* && "$(awk '{print $1}' <<<"$line")" == "$ME" ]]; then
			kill -KILL "$pid" 2>/dev/null || true
		fi
	fi
	# The port must actually clear, or the next model's readiness probe would
	# pass against the previous checkpoint's listener.
	for _ in $(seq 1 30); do
		ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . || return 0
		sleep 1
	done
	echo "WARNING: port $PORT still occupied after stopping pid $pid" >&2
}

trap 'stop_server' EXIT INT TERM

# ---------------------------------------------------------------- models
# Phase 4 first, then the released baselines, in the order of the original script.
# NOTE: fill in the frozen, already-selected Phase 4 checkpoint path. It must be
# the checkpoint chosen on Phase 4 dev data, not one picked from this run.
MODELS=(
	"p4_fastconformer:<selected .nemo>"
	"ehzawad_fastconformer:model/stt_bn_fastconformer.nemo"
	"hishab_fastconformer:model/titu_stt_bn_fastconformer.nemo"
	"hishab_conformer_large:model/titu_stt_bn_conformer_large.nemo"
	"whisper_medium:model/bengali_whisper_medium"
)

# The wav2vec2 behind the pre-existing ASR service. Path recorded in .w2v_path
# rather than hardcoded, since it lives in the shared HF cache.
if [[ -f .w2v_path ]]; then
	MODELS+=("wav2vec2_local:$(cat .w2v_path)")
fi

for entry in "${MODELS[@]}"; do
	name="${entry%%:*}"
	path="${entry#*:}"
	echo "=============================================================="
	echo "==> $name  ($path)"

	case "$path" in
		"<selected .nemo>")
			echo "ERROR: Phase 4 checkpoint path not filled in; skipping $name" >&2
			continue ;;
	esac
	if [[ ! -e "$path" ]]; then
		echo "ERROR: $name: no such model path: $path" >&2
		continue
	fi

	require_port_free

	STT_MODEL="$path" "$PY" bench_server.py --port "$PORT" \
		> "bench_${name}.log" 2>&1 < /dev/null &
	SERVER_PID=$!
	echo "    server pid $SERVER_PID (owned by this script)"

	ready=0
	for _ in $(seq 1 90); do
		if ! kill -0 "$SERVER_PID" 2>/dev/null; then
			echo "ERROR: $name server exited during startup; see bench_${name}.log" >&2
			tail -5 "bench_${name}.log" >&2
			break
		fi
		if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
			ready=1; break
		fi
		sleep 3
	done
	if [[ $ready -ne 1 ]]; then
		echo "ERROR: $name did not come up; see bench_${name}.log" >&2
		tail -5 "bench_${name}.log" >&2
		stop_server
		continue
	fi

	# Confirm the port is serving the checkpoint we intended, and that OUR pid
	# is the one bound to it -- not a leftover listener from anyone.
	if ! port_owner_pids | grep -qx "$SERVER_PID"; then
		echo "ERROR: :$PORT is not held by my pid $SERVER_PID; refusing to benchmark" >&2
		stop_server
		continue
	fi
	serving=$(curl -s "http://127.0.0.1:$PORT/health")
	echo "    serving: $serving"
	case "$serving" in
		*"$path"*) ;;
		*) echo "ERROR: wrong model on :$PORT — $serving" >&2; stop_server; continue ;;
	esac

	"$PY" "$HARNESS" \
		--api-url "http://127.0.0.1:$PORT/asr" \
		--data-dir "$DATA" \
		--concurrency "$CONC" \
		--output-dir "$OUT_ROOT/$name" 2>&1 | tail -3

	stop_server
done

# ---------------------------------------------------------------- qwen3 adapter
# The Qwen3 adapter is NOT served through bench_server.py: it needs its own
# interpreter (.venv-qwen) and its own backend, exactly as in run_all.sh. Run it
# in-process there, same eval manifest, same asr_core audio path, no port at all.
echo "=============================================================="
echo "==> qwen3_adapter (own backend, .venv-qwen)"
P=/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline
if [[ -x "$QWEN_PY" ]]; then
	"$QWEN_PY" bench_run.py --kind qwen \
		--model "Qwen/Qwen3-ASR-1.7B-hf" \
		--revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
		--adapter "$P/experiments/qwen_final/candidates/step_17280" \
		--label qwen3_adapter > "logs_qwen3_adapter.txt" 2>&1 \
		&& echo "    qwen3_adapter ok" \
		|| { echo "ERROR: qwen3_adapter failed; see logs_qwen3_adapter.txt" >&2
		     tail -5 logs_qwen3_adapter.txt >&2; }
else
	echo "ERROR: $QWEN_PY missing; skipping qwen3_adapter" >&2
fi

echo "=============================================================="
echo "done — results under $OUT_ROOT/ ; read $OUT_ROOT/RUN_NOTE.md first"
```

## Problems to flag (things I could not resolve read-only)

1. **`<selected .nemo>` is a literal placeholder.** No Phase 4 checkpoint has been named to me, so the script refuses on that line rather than guessing. Someone must substitute the frozen, dev-selected path before this runs.
2. **The Qwen3 branch's output path is not under `$OUT_ROOT`.** `bench_run.py` hardcodes `out_dir = ASSETS / "outputs" / a.label` (line 181) with no `--output-dir` flag, so running it would **overwrite `/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/outputs/qwen3_adapter`**-style paths in the released tree. I am read-only and did not add a flag. Either add `--output-dir` to `bench_run.py`, or copy the repo's `outputs/` aside first. As drafted, this is a real collision, not a hypothetical.
3. **`bench_run.py` also ignores `CUDA_VISIBLE_DEVICES` pinning only insofar as it hardcodes `.to("cuda")`** — which is correct given the export, but it means the A5000 pin is enforced purely by the exported env var; there is no in-script assertion that the visible device is actually the A5000. A `nvidia-smi -L`/`torch.cuda.get_device_name` assertion would be stronger but needs a code change I am not permitted to make.
4. **Third FLEURS exposure is mitigated by disclosure, not avoided.** The instruction was to preserve the protocol, so `eval_fleurs_bn` is still decoded; `RUN_NOTE.md` states prospectively that FLEURS is not held out for Phase 4 and that nothing may be selected on these numbers. If Council AA wants the exposure actually avoided rather than disclosed, this script does not do that — a different eval set would be required.
5. **`setsid nohup … & disown` was deliberately dropped.** The original detaches the server so the script cannot own it; ownership is the whole point of the targeted kill, so the server is now a plain child with a `trap` that stops it on exit/interrupt. Consequence: if this script is killed with `SIGKILL`, the server survives and must be cleaned up by hand (`ss -ltnp` then `kill <pid>`).
6. **`ss -ltnpH "sport = :$PORT"`** requires a reasonably modern iproute2; if `-H` is unsupported on this box the filter still works but a header line appears, which `grep -o 'pid='` tolerates. Not verified here — I did not run it.

Relevant paths: `/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/run_all_benchmarks.sh`, `/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/bench_server.py`, `/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/bench_run.py`, `/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/run_all.sh`.