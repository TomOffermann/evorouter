#!/bin/bash
# Diagnostics sharded over the 4 GPUs of one node, followed by a merge step (<= 30 min).
#
#   sbatch -A <account> -p debug --export=ALL,SCRIPT=scripts/scan_experts.py,NAME=scan-scored \
#          cluster/clariden/diag.sbatch
#   sbatch -A <account> -p debug --export=ALL,SCRIPT=scripts/probe_objective.py,NAME=probe \
#          cluster/clariden/diag.sbatch
#
# Outputs: $SCRATCH/evorouter/runs/diag/<NAME>/{shard*.json,shard*.log,summary.json}
#SBATCH --job-name=evorouter-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=evorouter-diag-%j.out
set -euo pipefail

srun --environment=evorouter bash cluster/clariden/run_shards.sh
