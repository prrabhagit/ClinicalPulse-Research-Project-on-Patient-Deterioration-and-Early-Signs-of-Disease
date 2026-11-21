"""Plot training curves from runs/history.json -> runs/curves.png"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
h = json.loads((run / "history.json").read_text())
ep = [r["epoch"] for r in h]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, [r["train_loss"] for r in h], label="train")
ax[0].plot(ep, [r["val_loss"] for r in h], label="val")
ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
ax[1].plot(ep, [r["val_acc"] for r in h], label="val acc")
ax[1].plot(ep, [r["val_auc"] for r in h], label="val AUC")
ax[1].set(title="Validation metrics", xlabel="epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(run / "curves.png", dpi=150)
print(f"Saved {run / 'curves.png'}")
