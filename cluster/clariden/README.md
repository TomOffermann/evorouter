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
