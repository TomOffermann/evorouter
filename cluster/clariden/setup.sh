#!/usr/bin/env bash
# One-time Clariden setup. Run from the repo root on a login node:
#   ACCOUNT=<project> bash cluster/clariden/setup.sh
# Creates: container image (squashfs) + Python venv on $SCRATCH, and ~/.edf/evorouter.toml.
set -euo pipefail

: "${ACCOUNT:?set ACCOUNT=<your Slurm project account>}"
: "${SCRATCH:?SCRATCH is not set (are you on Clariden?)}"
PARTITION="${PARTITION:-normal}"
NGC_TAG="${NGC_TAG:-25.01-py3}"                      # NVIDIA NGC PyTorch image tag (aarch64 + CUDA)
ROOT="${EVOROUTER_ROOT:-$SCRATCH/evorouter}"         # everything large lives here
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMG="$ROOT/images/ngc-pytorch-${NGC_TAG}.sqsh"
EDF="$HOME/.edf/evorouter.toml"

mkdir -p "$ROOT"/{images,hf,runs,smoke} "$HOME/.edf"
echo "repo:  $REPO"
echo "root:  $ROOT"

# 1) Import the container image once (enroot must run on a compute node).
if [[ ! -f "$IMG" ]]; then
  echo "== importing nvcr.io/nvidia/pytorch:${NGC_TAG} -> $IMG (takes a few minutes)"
  srun -A "$ACCOUNT" -p "$PARTITION" -N1 -n1 -t 00:45:00 \
    enroot import -x mount -o "$IMG" "docker://nvcr.io#nvidia/pytorch:${NGC_TAG}"
else
  echo "== image exists: $IMG"
fi

# 2) Environment definition file (EDF) for the CSCS container engine.
cat > "$EDF" <<TOML
image = "$IMG"
mounts = ["$SCRATCH:$SCRATCH", "$REPO:$REPO"]
workdir = "$REPO"

[env]
HF_HOME = "$ROOT/hf"
EVOROUTER_ROOT = "$ROOT"
PYTHONUNBUFFERED = "1"
TOKENIZERS_PARALLELISM = "false"
TOML
echo "== wrote $EDF"

# 3) Python venv inside the container, reusing the image's torch (system site packages).
echo "== creating venv $ROOT/venv and installing requirements"
srun -A "$ACCOUNT" -p "$PARTITION" -N1 -n1 -t 00:30:00 --environment=evorouter bash -c "
  set -euo pipefail
  # The NGC image's /etc/pip.conf adds pypi.ngc.nvidia.com as extra index, which is unreachable
  # from Alps and makes pip retry every request. Ignore that config and use PyPI only.
  export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_DEFAULT_TIMEOUT=60
  [[ -d '$ROOT/venv' ]] || python -m venv --system-site-packages '$ROOT/venv'
  source '$ROOT/venv/bin/activate'
  pip install --upgrade pip -q
  pip install -q -r '$REPO/cluster/clariden/requirements.txt'
  python -c 'import torch, transformers, datasets, cma; print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__)'
"
echo "== setup done. Next: sbatch -A $ACCOUNT cluster/clariden/smoke.sbatch"
