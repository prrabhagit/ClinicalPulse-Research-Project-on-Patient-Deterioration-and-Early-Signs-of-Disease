@echo off
REM One-time setup. For an NVIDIA GPU install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/).
cd /d "%~dp0.."
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[imaging,ocr,dev]" fastapi uvicorn python-multipart httpx
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
