"""
Chest X-ray image classifier (PyTorch, transfer learning).

Expected dataset layout (ImageFolder style, e.g. Kaggle "Chest X-Ray Images (Pneumonia)"):

    data/
      train/  NORMAL/*.jpeg   PNEUMONIA/*.jpeg
      val/    NORMAL/*.jpeg   PNEUMONIA/*.jpeg     (optional, see --val-split)
      test/   NORMAL/*.jpeg   PNEUMONIA/*.jpeg

Works with any number of classes (e.g. NORMAL / PNEUMONIA / COVID / TB) -
class names come from the folder names.

Usage:
    python xray_classifier.py train    --data data --arch densenet121 --epochs 15
    python xray_classifier.py evaluate --data data --ckpt runs/best.pt
    python xray_classifier.py predict  --ckpt runs/best.pt --image some_xray.jpg --gradcam

NOTE: research/educational use only - not a medical device.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_tfms():
    # Mild augmentation only: X-rays are standardized, so heavy flips/crops
    # can destroy anatomy cues (e.g. heart on the left side), so no h-flip.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_tfms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(arch, num_classes, pretrained=True):
    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, num_classes))
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def load_backbone_weights(model, path):
    """Load an ImageNet state_dict (torchvision/timm ResNet naming) into everything except the new head.

    Uses torch.load(weights_only=True). Raises if anything other than the classifier head differs.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("fc.")] + list(unexpected)
    if bad:
        raise ValueError(f"Backbone weights do not match the model; offending keys: {bad[:5]}")


def set_backbone_trainable(model, arch, trainable):
    head = {"resnet18": "fc", "resnet50": "fc", "densenet121": "classifier", "efficientnet_b0": "classifier"}[arch]
    for name, p in model.named_parameters():
        p.requires_grad = True if name.startswith(head) else trainable


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_loaders(data_dir, batch_size, val_split, workers):
    root = Path(data_dir)
    train_full = datasets.ImageFolder(root / "train", transform=train_tfms())
    class_names = train_full.classes

    val_dir = root / "val"
    if val_split > 0 or not val_dir.exists():
        # Kaggle's val folder has only 16 images -> carve a real val set out of train.
        frac = val_split if val_split > 0 else 0.15
        idx = np.arange(len(train_full))
        np.random.shuffle(idx)
        n_val = int(len(idx) * frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        val_base = datasets.ImageFolder(root / "train", transform=eval_tfms())
        train_ds, val_ds = Subset(train_full, train_idx), Subset(val_base, val_idx)
        train_targets = [train_full.targets[i] for i in train_idx]
    else:
        train_ds = train_full
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms())
        train_targets = train_full.targets

    # Balance classes via weighted sampling (chest X-ray sets are usually imbalanced)
    counts = np.bincount(train_targets, minlength=len(class_names))
    class_w = 1.0 / np.maximum(counts, 1)
    sample_w = [class_w[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    test_dl = None
    if (root / "test").exists():
        test_ds = datasets.ImageFolder(root / "test", transform=eval_tfms())
        assert test_ds.classes == class_names, "train/test class folders differ"
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    print(f"Classes: {class_names} | train class counts: {counts.tolist()}")
    return train_dl, val_dl, test_dl, class_names


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_eval(model, loader, device, criterion=None):
    model.eval()
    all_probs, all_y, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        all_probs.append(F.softmax(logits, dim=1).cpu())
        all_y.append(y.cpu())
    probs, ys = torch.cat(all_probs).numpy(), torch.cat(all_y).numpy()
    preds = probs.argmax(1)
    acc = float((preds == ys).mean())
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(ys, probs[:, 1])
        else:
            auc = roc_auc_score(ys, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return total_loss / max(n, 1), acc, auc, probs, ys


def train(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl, class_names = make_loaders(args.data, args.batch_size, args.val_split, args.workers)
    model = build_model(args.arch, len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        # Stage 1: warm up the new head only. Stage 2: fine-tune everything at lower LR.
        if epoch == 1:
            set_backbone_trainable(model, args.arch, False)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        if epoch == args.warmup + 1:
            set_backbone_trainable(model, args.arch, True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup, 1))
        if epoch <= args.warmup:
            sched = None

        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        if sched is not None:
            sched.step()

        val_loss, val_acc, val_auc, _, _ = run_eval(model, val_dl, device, criterion)
        row = dict(epoch=epoch, train_loss=run_loss / seen, val_loss=val_loss, val_acc=val_acc, val_auc=val_auc)
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {row['train_loss']:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_auc {val_auc:.4f}")

        score = val_auc if not np.isnan(val_auc) else val_acc
        if score > best_auc:
            best_auc, bad_epochs = score, 0
            torch.save({"model": model.state_dict(), "arch": args.arch, "classes": class_names}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val score: {best_auc:.4f}. Checkpoint -> {out / 'best.pt'}")

    if test_dl is not None:
        ckpt = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        report(model, test_dl, device, class_names)


def report(model, loader, device, class_names):
    _, acc, auc, probs, ys = run_eval(model, loader, device)
    preds = probs.argmax(1)
    print(f"\nTEST  acc={acc:.4f}  auc={auc:.4f}")
    print(classification_report(ys, preds, target_names=class_names, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(ys, preds))


def evaluate(args):
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    ds = datasets.ImageFolder(Path(args.data) / args.split, transform=eval_tfms())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    report(model, dl, device, ckpt["classes"])


# --------------------------------------------------------------------------- #
# Inference + Grad-CAM
# --------------------------------------------------------------------------- #
def target_layer(model, arch):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features
    return model.features[-1]


def grad_cam(model, arch, x, class_idx):
    feats, grads = {}, {}
    layer = target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: feats.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
    model.zero_grad()
    logits = model(x)
    logits[0, class_idx].backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def predict(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    arch, classes = ckpt["arch"], ckpt["classes"]
    model = build_model(arch, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = eval_tfms()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    for c, p in sorted(zip(classes, probs), key=lambda t: -t[1]):
        print(f"{c:>15}: {p:.4f}")

    if args.gradcam:
        idx = int(probs.argmax())
        x.requires_grad_(True)
        cam = grad_cam(model, arch, x, idx)
        base = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(base)
        ax[0].set_title("Input")
        ax[1].imshow(base)
        ax[1].imshow(cam, cmap="jet", alpha=0.45)
        ax[1].set_title(f"Grad-CAM: {classes[idx]} ({probs[idx]:.2f})")
        for a in ax:
            a.axis("off")
        out = Path(args.image).with_suffix("").as_posix() + "_gradcam.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--arch", default="densenet121", choices=["resnet18", "resnet50", "densenet121", "efficientnet_b0"])
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--warmup", type=int, default=2, help="epochs training only the new head")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-split", type=float, default=0.0, help="fraction of train to hold out (0 = use val/ folder if present, else 0.15)")
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-pretrained", action="store_true", help="random init (offline / smoke tests)")
    t.add_argument("--out", default="runs/clinicalpulse-cxr")
    t.set_defaults(fn=train)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--workers", type=int, default=4)
    e.set_defaults(fn=evaluate)

    r = sub.add_parser("predict")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--gradcam", action="store_true")
    r.set_defaults(fn=predict)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
