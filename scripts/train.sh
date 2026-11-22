#!/usr/bin/env bash
# Train the ClinicalPulse X-ray model on every ChestXray14_*.tar shard in data/shards/.
#   scripts/train.sh            GPU/online mode: DenseNet121, torchvision ImageNet weights (downloads once)
#   scripts/train.sh offline    ResNet18 with the bundled weights/resnet18_a1_0-d63eafa0.pth (no download)
# Overrides: EPOCHS, BATCH, WORKERS, OUT.   Example: EPOCHS=30 scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
MODE="${1:-online}"
if [ "$MODE" = "offline" ]; then
  EXTRA=(--arch resnet18 --init-weights weights/resnet18_a1_0-d63eafa0.pth)
else
  EXTRA=(--arch densenet121)
fi
python -m clinicalpulse.imaging.train_cxr train --dataset chestxray14 --root data/cxr14 --shards data/shards \
  "${EXTRA[@]}" --epochs "${EPOCHS:-15}" --warmup 2 --patience 5 --batch-size "${BATCH:-32}" \
  --workers "${WORKERS:-4}" --out "${OUT:-runs/clinicalpulse-cxr}"
echo "Done. Checkpoint + test_metrics.json are in ${OUT:-runs/clinicalpulse-cxr}"
