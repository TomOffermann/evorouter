#!/usr/bin/env bash
# Runs INSIDE the container (started by diag.sbatch): one shard of a diagnostic script per GPU,
# then the merge step. Configuration via environment variables:
#   SCRIPT      e.g. scripts/scan_experts.py or scripts/probe_objective.py   (required)
#   NAME        output group under $EVOROUTER_ROOT/runs/diag                 (required)
#   SHARDS      number of GPUs / shards                                      (default 4)
#   EXTRA_ARGS  passed through to every shard
set -euo pipefail

ROOT="${EVOROUTER_ROOT:?EVOROUTER_ROOT not set (is the evorouter EDF active?)}"
SCRIPT="${SCRIPT:?set SCRIPT}"
NAME="${NAME:?set NAME}"
SHARDS="${SHARDS:-4}"
OUT="$ROOT/runs/diag/$NAME"
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"

pids=()
for ((i = 0; i < SHARDS; i++)); do
  # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
  CUDA_VISIBLE_DEVICES="$i" python "$SCRIPT" --out-dir "$OUT" --shard "$i" --num-shards "$SHARDS" \
    ${EXTRA_ARGS:-} > "$OUT/shard$i.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "a shard failed; last lines of each log:"
  for ((i = 0; i < SHARDS; i++)); do echo "== shard $i"; tail -n 20 "$OUT/shard$i.log"; done
  exit 1
fi
python "$SCRIPT" --out-dir "$OUT" --merge
