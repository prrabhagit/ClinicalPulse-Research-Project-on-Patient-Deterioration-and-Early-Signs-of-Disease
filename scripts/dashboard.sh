#!/usr/bin/env bash
# Start the dashboard at http://localhost:8000 using the trained checkpoint (override with CKPT=...).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export CLINICALPULSE_CKPT="${CKPT:-runs/clinicalpulse-cxr/best.pt}"
[ -f "$CLINICALPULSE_CKPT" ] || echo "Note: $CLINICALPULSE_CKPT not found; the X-ray panel will say no model is configured."
python -m uvicorn app.server:app --port 8000
