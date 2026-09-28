#!/usr/bin/env bash
# Runs INSIDE the container (started by run_search.sbatch): one search process per GPU.
# Configuration via environment variables:
#   OPTIMIZER   cma | sep-cma | random            (default cma)
#   NAME        run group under $EVOROUTER_ROOT/runs (default <optimizer>-v1)
#   SEEDS       space-separated, at most 4          (default "0 1 2 3")
#   BUDGET_MIN  search time budget per job, minutes (default 45)
#   EXTRA_ARGS  passed through to scripts/run_search.py (only used when a run is created)
set -euo pipefail

ROOT="${EVOROUTER_ROOT:?EVOROUTER_ROOT not set (is the evorouter EDF active?)}"
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$PWD/src"

OPTIMIZER="${OPTIMIZER:-cma}"
NAME="${NAME:-${OPTIMIZER}-v1}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3}"
BUDGET_MIN="${BUDGET_MIN:-45}"

pids=()
for i in "${!SEED_LIST[@]}"; do
  seed="${SEED_LIST[$i]}"
  dir="$ROOT/runs/$NAME/seed$seed"
  mkdir -p "$dir"
  # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
  CUDA_VISIBLE_DEVICES="$i" python scripts/run_search.py \
    --run-dir "$dir" --seed "$seed" --optimizer "$OPTIMIZER" --time-budget-min "$BUDGET_MIN" \
    ${EXTRA_ARGS:-} >> "$dir/log.txt" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for seed in "${SEED_LIST[@]}"; do
  echo "== $NAME seed $seed"
  tail -n 2 "$ROOT/runs/$NAME/seed$seed/log.txt"
done
exit "$status"
