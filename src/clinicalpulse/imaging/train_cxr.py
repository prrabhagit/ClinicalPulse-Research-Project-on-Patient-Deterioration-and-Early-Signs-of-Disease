"""Train / evaluate / predict a multi-label chest X-ray model (ClinicalPulse imaging model).

Datasets (--dataset): `mimic` (MIMIC-CXR-JPG, CheXpert labels) or `chestxray14` (NIH ChestX-ray14 shards).

    python -m clinicalpulse.imaging.train_cxr train --dataset chestxray14 --root data/cxr14 \
        --shards ChestXray14_train_000.tar ChestXray14_val_002.tar ChestXray14_test_000.tar
    python -m clinicalpulse.imaging.train_cxr train --dataset mimic --root /path/to/mimic-cxr-jpg --max-images 60000
    python -m clinicalpulse.imaging.train_cxr evaluate --root data/cxr14 --ckpt runs/clinicalpulse-cxr/best.pt
    python -m clinicalpulse.imaging.train_cxr predict  --ckpt runs/clinicalpulse-cxr/best.pt --image xray.png --gradcam

Both datasets use patient-level train/validate/test splits (a leakage check runs on every index).
Research use only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import chestxray14, mimic_data
from .metrics import evaluate as compute_metrics
from .mimic_data import UNCERTAIN_POLICIES, DatasetError, make_targets
from .xray_classifier import (IMG_SIZE, build_model, eval_tfms, get_device, grad_cam, load_backbone_weights,
                              seed_everything, set_backbone_trainable, train_tfms)


class MimicDataset(Dataset):
    def __init__(self, paths, y, mask, tfm):
        self.paths, self.y, self.mask, self.tfm = list(paths), y, mask, tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i])
            img.load()
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Cannot read image {self.paths[i]}: {e}") from e
        return self.tfm(img.convert("L")), torch.from_numpy(self.y[i]), torch.from_numpy(self.mask[i])


def make_loader(df, labels, uncertain, training, batch_size, workers):
    y, m = make_targets(df, labels, uncertain, training=training)
    ds = MimicDataset(df["path"], y, m, train_tfms() if training else eval_tfms())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=workers,
                    pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return dl, y, m


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for x, _, _ in loader:
        out.append(torch.sigmoid(model(x.to(device)).float()).cpu())
    return torch.cat(out).numpy()


def checkpoint_config(args):
    """Plain-data copy of the CLI args (no callables), so checkpoints load with torch.load(weights_only=True)."""
    return {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))}


def masked_bce(logits, y, m, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight, reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1)


DATASETS = ("mimic", "chestxray14")


def resolve_labels(names, dataset):
    everything = mimic_data.ALL_LABELS if dataset == "mimic" else chestxray14.LABELS
    if names is None:
        return mimic_data.DEFAULT_LABELS if dataset == "mimic" else everything
    return everything if names == ["all"] else names


def index_for(args, labels):
    """Build the (dataframe, summary) index for the chosen dataset."""
    if args.dataset == "mimic":
        return mimic_data.build_index(args.root, labels, args.views, args.max_images, args.seed)
    if getattr(args, "shards", None):
        shards = chestxray14.expand_shards(args.shards)
        if chestxray14.already_extracted(shards, args.root):
            print(f"Using shards already extracted in {args.root}")
        else:
            print(f"Extracting {len(shards)} shard(s) into {args.root} ...")
            chestxray14.extract_shards(shards, args.root)
    return chestxray14.build_index(args.root, labels, args.max_images, args.seed)


def print_report(res, labels):
    print(f"{'label':<28}{'n':>7}{'AUROC':>8}{'sens':>7}{'spec':>7}{'prec':>7}{'F1':>7}{'ECE':>7}")
    for l in labels:
        r = res["per_label"][l]
        if "auroc" not in r:
            print(f"{l:<28}{r['n']:>7}  {r['note']}")
            continue
        print(f"{l:<28}{r['n']:>7}{r['auroc']:>8.3f}{r['sensitivity']:>7.3f}{r['specificity']:>7.3f}"
              f"{r['precision']:>7.3f}{r['f1']:>7.3f}{r['ece']:>7.3f}")
    print("macro:", {k: round(v, 4) for k, v in res["macro"].items()})


def train(args):
    seed_everything(args.seed)
    device = get_device()
    labels = resolve_labels(args.labels, args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, summary = index_for(args, labels)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("images_used", "patients", "per_split")}, indent=2))

    tr, va = df[df["split"] == "train"], df[df["split"] == "validate"]
    if tr.empty or va.empty:
        raise DatasetError("Need images in both the train and validate splits "
                           "(a partial download may have left one empty; download more folders).")
    train_dl, y_tr, m_tr = make_loader(tr, labels, args.uncertain, True, args.batch_size, args.workers)
    val_dl, y_va, m_va = make_loader(va, labels, args.uncertain, False, args.batch_size, args.workers)

    pos, neg = (y_tr * m_tr).sum(0), ((1 - y_tr) * m_tr).sum(0)
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32, device=device)

    model = build_model(args.arch, len(labels), pretrained=not (args.no_pretrained or args.init_weights))
    if args.init_weights:
        load_backbone_weights(model, args.init_weights)
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, bad, history, sched = -1.0, 0, [], None

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:  # stage 1: train only the new head
            set_backbone_trainable(model, args.arch, False)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
        if epoch == args.warmup + 1:  # stage 2: fine-tune everything at a lower LR
            set_backbone_trainable(model, args.arch, True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup, 1))
        model.train()
        run, seen = 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = masked_bce(logits, y, m, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            seen += x.size(0)
        if sched is not None:
            sched.step()

        p_va = predict_probs(model, val_dl, device)
        res = compute_metrics(y_va, p_va, m_va, labels)
        auc = res["macro"].get("auroc", float("nan"))
        history.append({"epoch": epoch, "train_loss": run / seen, "val_macro_auroc": auc})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {run / seen:.4f} | val macro-AUROC {auc:.4f}")
        if auc > best:
            best, bad = auc, 0
            torch.save({"model": model.state_dict(), "arch": args.arch, "labels": labels, "config": checkpoint_config(args)},
                       out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break
    (out / "history.json").write_text(json.dumps(history, indent=2))

    # Final: thresholds chosen on validation, applied unchanged to the test split.
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    val_res = compute_metrics(y_va, predict_probs(model, val_dl, device), m_va, labels)
    ckpt["thresholds"] = val_res["thresholds"]
    torch.save(ckpt, out / "best.pt")
    run_split(model, df, "test", labels, ckpt["thresholds"], args, device, out / "test_metrics.json")


def run_split(model, df, split, labels, thresholds, args, device, save_to=None):
    part = df[df["split"] == split]
    if part.empty:
        raise DatasetError(f"No images in split '{split}'.")
    dl, y, m = make_loader(part, labels, "ignore", False, args.batch_size, args.workers)
    res = compute_metrics(y, predict_probs(model, dl, device), m, labels, thresholds)
    print(f"\n{split.upper()} ({len(part)} images, {part['subject_id'].nunique()} patients; "
          "uncertain labels excluded)")
    print_report(res, labels)
    if save_to:
        Path(save_to).write_text(json.dumps(res, indent=2))
        print(f"Saved {save_to}")
    return res


def evaluate_cmd(args):
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    labels = ckpt["labels"]
    model = build_model(ckpt["arch"], len(labels), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    args.dataset = args.dataset or ckpt["config"].get("dataset", "mimic")
    df, _ = index_for(args, labels)
    run_split(model, df, args.split, labels, ckpt.get("thresholds"), args, device)


def load_for_inference(ckpt_path):
    """Return (model, labels, arch, thresholds, device) from a checkpoint written by `train`."""
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["labels"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["labels"], ckpt["arch"], ckpt.get("thresholds", {}), device


def predict_image(model, labels, thresholds, device, img):
    """img: PIL image. Returns [{label, probability, above_threshold}] sorted by probability."""
    x = eval_tfms()(img.convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()
    rows = [{"label": l, "probability": float(p),
             "above_threshold": (bool(p >= thresholds[l]) if l in thresholds else None)}
            for l, p in zip(labels, probs)]
    return sorted(rows, key=lambda r: -r["probability"])


def predict_cmd(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, labels, arch, thr, device = load_for_inference(args.ckpt)
    img = Image.open(args.image).convert("L")
    rows = predict_image(model, labels, thr, device, img)
    for r in rows:
        flag = "  <- above validation threshold" if r["above_threshold"] else ""
        print(f"{r['label']:<28}{r['probability']:.3f}{flag}")
    print("Model output is a research estimate, not a diagnosis.")
    if args.gradcam:
        top = labels.index(rows[0]["label"])
        x = eval_tfms()(img).unsqueeze(0).to(device)
        x.requires_grad_(True)
        cam = grad_cam(model, arch, x, top)
        base = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(base, cmap="gray")
        ax[1].imshow(base, cmap="gray")
        ax[1].imshow(cam, cmap="jet", alpha=0.45)
        ax[0].set_title("Input")
        ax[1].set_title(f"Grad-CAM: {rows[0]['label']} ({rows[0]['probability']:.2f})")
        for a_ in ax:
            a_.axis("off")
        out = Path(args.image).with_suffix("").as_posix() + "_gradcam.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p, dataset_default):
        p.add_argument("--dataset", choices=DATASETS, default=dataset_default)
        p.add_argument("--root", required=True,
                       help="mimic: folder with mimic-cxr-2.0.0-*.csv.gz and files/; chestxray14: extraction folder")
        p.add_argument("--shards", nargs="+", default=None, help="chestxray14: .tar shards to extract into --root first")
        p.add_argument("--views", nargs="+", default=["PA", "AP"], help="mimic only: ViewPosition values to keep")
        p.add_argument("--max-images", type=int, default=None, help="subset by whole patients to ~N images")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train")
    data_args(t, "mimic")
    t.add_argument("--labels", nargs="+", default=None, help="label names, or: all (default: dataset-specific)")
    t.add_argument("--uncertain", default="ones", choices=UNCERTAIN_POLICIES, help="how to treat -1 labels in TRAINING")
    t.add_argument("--arch", default="densenet121", choices=["resnet18", "resnet50", "densenet121", "efficientnet_b0"])
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--warmup", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--no-pretrained", action="store_true", help="random init")
    t.add_argument("--init-weights", default=None, help="path to an ImageNet state_dict (offline alternative to torchvision download)")
    t.add_argument("--out", default="runs/clinicalpulse-cxr")
    t.set_defaults(fn=train)

    e = sub.add_parser("evaluate")
    data_args(e, None)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test", choices=["train", "validate", "test"])
    e.set_defaults(fn=evaluate_cmd)

    r = sub.add_parser("predict")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--gradcam", action="store_true")
    r.set_defaults(fn=predict_cmd)

    args = ap.parse_args()
    try:
        args.fn(args)
    except DatasetError as err:
        raise SystemExit(f"error: {err}")


if __name__ == "__main__":
    main()
