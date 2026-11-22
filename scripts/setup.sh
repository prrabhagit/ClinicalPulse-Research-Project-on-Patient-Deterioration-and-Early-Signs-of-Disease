#!/usr/bin/env bash
# One-time setup: virtualenv + dependencies. For an NVIDIA GPU, first install the CUDA build of PyTorch from
# https://pytorch.org/get-started/locally/ (inside the venv), then run this script again.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[imaging,ocr,dev]" fastapi uvicorn python-multipart httpx
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Setup done. OCR for scanned reports also needs the tesseract binary (apt install tesseract-ocr / brew install tesseract)."
