#!/bin/bash
# v0 search: up to 4 seeds in parallel (one per GPU), <= 1 h per job, resumable.
# Resubmitting the same command continues every unfinished seed from its saved state.
#
#   sbatch -A <account> cluster/clariden/run_search.sbatch                                  # CMA-ES, seeds 0-3
#   sbatch -A <account> --export=ALL,OPTIMIZER=random cluster/clariden/run_search.sbatch    # random search
#   sbatch -A <account> -p debug --export=ALL,NAME=pilot,SEEDS=0,EXTRA_ARGS="--generations 10" \
#          cluster/clariden/run_search.sbatch                                               # quick pilot
#
# Outputs: $SCRATCH/evorouter/runs/<NAME>/seed<k>/{config.json,metrics.jsonl,val.jsonl,log.txt,final.json}
#SBATCH --job-name=evorouter-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH --output=evorouter-search-%j.out
set -euo pipefail

srun --environment=evorouter bash cluster/clariden/run_seeds.sh
