#!/bin/bash
# Independent cross-check: lm-eval-harness arc_challenge on the base model.
# Our eval_base.py acc / acc_norm on the test split should match these numbers closely.
# Submit from the repo root:  sbatch -A <account> -p debug cluster/clariden/lmeval_arc.sbatch
#SBATCH --job-name=evorouter-lmeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --output=evorouter-lmeval-%j.out
set -euo pipefail

ROOT="${EVOROUTER_ROOT:-$SCRATCH/evorouter}"
OUT="$ROOT/runs/lmeval/lmeval-${SLURM_JOB_ID}"

srun --environment=evorouter bash -c "
  set -euo pipefail
  source '$ROOT/venv/bin/activate'
  export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
  python -c 'import lm_eval' 2>/dev/null || pip install -q 'lm_eval>=0.4.5'
  lm_eval --model hf \
    --model_args pretrained=allenai/OLMoE-1B-7B-0924,dtype=bfloat16 \
    --tasks arc_challenge --batch_size 32 --output_path '$OUT'
"
