"""Multi-label evaluation: AUROC, thresholded sensitivity/specificity/precision/F1/accuracy, calibration.

Thresholds are chosen on the validation set and then applied unchanged to the test set.
Uncertain/masked entries (mask == 0) are excluded from every metric.
"""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1 on (y, p). Returns 0.5 if undefined."""
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99))):
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        if sens + spec - 1 > best_j:
            best_j, best_t = sens + spec - 1, float(t)
    return best_t


def evaluate(y: np.ndarray, p: np.ndarray, mask: np.ndarray, labels: Sequence[str],
             thresholds: Dict[str, float] = None) -> Dict:
    """y, p, mask: [N, L]. Returns per-label metrics, macro means and the thresholds used."""
    per, used = {}, {}
    for j, name in enumerate(labels):
        m = mask[:, j] > 0
        yy, pp = y[m, j], p[m, j]
        if m.sum() == 0 or len(np.unique(yy)) < 2:
            per[name] = {"n": int(m.sum()), "note": "needs both classes; metrics undefined"}
            continue
        t = (thresholds or {}).get(name)
        if t is None:
            t = youden_threshold(yy, pp)
        used[name] = t
        pred = pp >= t
        tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
        tn = int((~pred & (yy == 0)).sum()); fn = int((~pred & (yy == 1)).sum())
        per[name] = {
            "n": int(m.sum()), "prevalence": float(yy.mean()),
            "auroc": float(roc_auc_score(yy, pp)),
            "threshold": t,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "ece": expected_calibration_error(yy, pp),
            "brier": float(np.mean((pp - yy) ** 2)),
        }
    valid = [v for v in per.values() if "auroc" in v]
    macro = {k: float(np.mean([v[k] for v in valid])) for k in
             ("auroc", "sensitivity", "specificity", "precision", "f1", "accuracy", "ece", "brier")} if valid else {}
    return {"per_label": per, "macro": macro, "thresholds": used}
