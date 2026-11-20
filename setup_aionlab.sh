#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ASTRAMIS_PYTHON:-/home/alexsandrdev/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
SHARED_SITE="${ASTRAMIS_SHARED_SITE:-/home/alexsandrdev/.venvs/aion-modular-recovery-20260907/lib/python3.12/site-packages}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12 not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$SHARED_SITE/torch" ]]; then
  echo "The selected shared AION environment has no PyTorch package: $SHARED_SITE" >&2
  exit 2
fi

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m venv .venv
printf '%s\n' "$SHARED_SITE" > .venv/lib/python3.12/site-packages/astramis_aion_torch.pth
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-aionlab.txt
.venv/bin/python -c 'import torch, torchvision, ultralytics; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "torchvision", torchvision.__version__, "ultralytics", ultralytics.__version__)'

