#!/bin/bash
# Smoke test: environment, 4x GH200, OLMoE load + forward + router hook, ARC-C download.
# Submit from the repo root:  sbatch -A <account> cluster/clariden/smoke.sbatch
#SBATCH --job-name=evorouter-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=evorouter-smoke-%j.out
set -euo pipefail

ROOT="${EVOROUTER_ROOT:-$SCRATCH/evorouter}"
mkdir -p "$ROOT/smoke"

srun --environment=evorouter bash -c "
  set -euo pipefail
  source '$ROOT/venv/bin/activate'
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  python scripts/smoke_test.py --out '$ROOT/smoke/smoke-${SLURM_JOB_ID}.json'
"
