# Clariden (CSCS Alps) setup

One node = 4x GH200 (aarch64 Grace CPU + H100 96 GB), 288 CPU cores. We use the CSCS
container engine with the NVIDIA NGC PyTorch image (aarch64 + CUDA), plus a small venv on
`$SCRATCH` for transformers/datasets/cma. Large files (image, HF cache, runs) live under
`$SCRATCH/evorouter`.

## 1. One-time setup (login node, from the repo root)

```bash
git clone <repo-url> evorouter && cd evorouter
ACCOUNT=<your_project> bash cluster/clariden/setup.sh
```

This (a) imports `nvcr.io/nvidia/pytorch:25.01-py3` to `$SCRATCH/evorouter/images/` (~10 min),
(b) writes `~/.edf/evorouter.toml`, (c) creates `$SCRATCH/evorouter/venv` inside the container
and installs `cluster/clariden/requirements.txt`. Override defaults with `PARTITION=...`,
`NGC_TAG=...`, `EVOROUTER_ROOT=...`.

## 2. Smoke test

```bash
sbatch -A <your_project> cluster/clariden/smoke.sbatch
tail -f evorouter-smoke-<jobid>.out
```

Checks: env (aarch64, 4 GPUs, versions) · bf16 matmul per GPU · OLMoE-1B-7B load + config ·
option log-likelihoods on one ARC-style question · router forward hook · batch throughput ·
ARC-Challenge download. The JSON report goes to `$SCRATCH/evorouter/smoke/smoke-<jobid>.json`.
**Send back the `.out` file** (or the JSON): the throughput and router-margin numbers feed the
runtime plan and the CMA-ES step size.

Expected on success: all checks PASS; `score` should pick `wind` under acc_norm; ~14 GB GPU memory
after model load.

## 3. Milestone 1: base model on ARC-Challenge

```bash
git pull
sbatch -A <your_project> -p debug cluster/clariden/eval_base.sbatch     # ours, ~5-10 min
sbatch -A <your_project> -p debug cluster/clariden/lmeval_arc.sbatch    # lm-eval cross-check, ~10-20 min
```

`eval_base` writes `$SCRATCH/evorouter/runs/base/base-<jobid>.json` with acc / acc_norm / fitness
per split, evaluator throughput, a zero-genome sanity check (must be ~0), and per-layer router
top-k margins and loads. The lm-eval job installs `lm_eval` into the venv on first use. Our test
acc and acc_norm should match lm-eval's `arc_challenge` numbers; C3PO reports acc_norm 51.3.

## 4. Milestone 2: CMA-ES search (v0)

```bash
git pull
# pilot: 1 seed, 10 generations, checks speed and logging (~10 min)
sbatch -A <your_project> -p debug --export=ALL,NAME=pilot,SEEDS=0,EXTRA_ARGS="--generations 10" \
       cluster/clariden/run_search.sbatch
# full: CMA-ES on seeds 0-3, one per GPU; resubmit the same line until every seed has final.json
sbatch -A <your_project> cluster/clariden/run_search.sbatch
```

Each seed writes `$SCRATCH/evorouter/runs/<NAME>/seed<k>/`: `config.json` (settings, margins),
`metrics.jsonl` (per generation), `val.jsonl` (every 10 generations), `log.txt`, and `final.json`
(test acc_norm of the selected genome vs base) once 200 generations are done. Other arms:
`--export=ALL,OPTIMIZER=random` or `OPTIMIZER=sep-cma`.

## 5. Diagnostics (sharded over 4 GPUs, then merged)

```bash
sbatch -A <your_project> -p debug --export=ALL,SCRIPT=scripts/scan_experts.py,NAME=scan-scored \
       cluster/clariden/diag.sbatch
sbatch -A <your_project> -p debug --export=ALL,SCRIPT=scripts/probe_objective.py,NAME=probe \
       cluster/clariden/diag.sbatch
```

Results: `$SCRATCH/evorouter/runs/diag/<NAME>/summary.json` (also printed at the end of the job log).
Add `EXTRA_ARGS="--scope all"` to run the scan with the bias on all positions.

## Job length

Every job must finish within **1 hour** (course compute limit). All sbatch files request at most
45 minutes; long searches will be split into resumable, chained jobs.

## Interactive shell (debugging)

```bash
srun -A <your_project> -t 01:00:00 --environment=evorouter --pty bash
source $EVOROUTER_ROOT/venv/bin/activate
```

## Troubleshooting

- **`enroot import` fails with an auth error for nvcr.io:** add NGC credentials to
  `~/.config/enroot/.credentials` (`machine nvcr.io login $oauthtoken password <NGC_API_KEY>`),
  or pick another tag with `NGC_TAG=...`.
- **pip or the Hub is unreachable from compute nodes:** tell me the error; we will pre-download
  the model/dataset into `$SCRATCH/evorouter/hf` and run with `HF_HUB_OFFLINE=1`.
- **`--environment=evorouter` not found:** check that `~/.edf/evorouter.toml` exists.
- **`gpus` shows fewer than 4 devices:** add `#SBATCH --gpus-per-node=4` to the sbatch file.
