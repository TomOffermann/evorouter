#!/bin/bash
# Milestone 1: base-model ARC-Challenge evaluation + router statistics (1 GPU is enough).
# Submit from the repo root:  sbatch -A <account> -p debug cluster/clariden/eval_base.sbatch
#SBATCH --job-name=evorouter-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=evorouter-base-%j.out
set -euo pipefail

ROOT="${EVOROUTER_ROOT:-$SCRATCH/evorouter}"
REPO="$PWD"
OUT="$ROOT/runs/base/base-${SLURM_JOB_ID}.json"

srun --environment=evorouter bash -c "
  set -euo pipefail
  source '$ROOT/venv/bin/activate'
  export PYTHONPATH='$REPO/src'
  python scripts/eval_base.py --splits test validation train --out '$OUT'
"
