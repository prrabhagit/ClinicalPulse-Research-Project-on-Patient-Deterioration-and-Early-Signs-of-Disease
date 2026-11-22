"""MIMIC-CXR-JPG indexing: official patient-level splits, CheXpert labels, subsetting.

Needs only pandas/numpy (no torch). Expects the PhysioNet download layout:

    <root>/mimic-cxr-2.0.0-metadata.csv.gz
    <root>/mimic-cxr-2.0.0-split.csv.gz
    <root>/mimic-cxr-2.0.0-chexpert.csv.gz
    <root>/files/p10/p10000032/s50414267/<dicom_id>.jpg

The dataset is credentialed-access (PhysioNet account + CITI training + signed DUA); it is
never downloaded by this code.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding",
]
# The 5 CheXpert competition labels plus three more common, clinically relevant findings.
DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Lung Opacity",
]
SPLITS = ("train", "validate", "test")
UNCERTAIN_POLICIES = ("ones", "zeros", "ignore")


class DatasetError(Exception):
    pass


def _find(root: Path, stem: str) -> Path:
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        if (root / name).is_file():
            return root / name
    raise DatasetError(
        f"'{stem}.csv.gz' not found in {root}. Point --root at the folder that contains the "
        "mimic-cxr-2.0.0-*.csv.gz files and the files/ directory."
    )


def image_path(root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    return root / "files" / f"p{str(subject_id)[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def load_tables(root) -> pd.DataFrame:
    """Join split + metadata + CheXpert labels. Studies without a CheXpert row are dropped."""
    root = Path(root)
    split = pd.read_csv(_find(root, "mimic-cxr-2.0.0-split"))
    meta = pd.read_csv(_find(root, "mimic-cxr-2.0.0-metadata"), usecols=["dicom_id", "ViewPosition"])
    labels = pd.read_csv(_find(root, "mimic-cxr-2.0.0-chexpert"))
    for name, df, need in (("split", split, {"dicom_id", "study_id", "subject_id", "split"}),
                           ("metadata", meta, {"dicom_id", "ViewPosition"}),
                           ("chexpert", labels, {"subject_id", "study_id"})):
        missing = need - set(df.columns)
        if missing:
            raise DatasetError(f"{name} CSV is missing columns: {sorted(missing)}")
    bad = set(split["split"].unique()) - set(SPLITS)
    if bad:
        raise DatasetError(f"Unexpected split names: {sorted(bad)}")
    df = split.merge(meta, on="dicom_id", how="left").merge(labels, on=["subject_id", "study_id"], how="inner")
    return df


def check_no_patient_leakage(df: pd.DataFrame) -> None:
    sets = {s: set(df.loc[df["split"] == s, "subject_id"]) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and sets[a] & sets[b]:
                raise DatasetError(f"Patient leakage between '{a}' and '{b}' splits "
                                   f"({len(sets[a] & sets[b])} shared patients).")


def subset_by_patient(df: pd.DataFrame, max_images: int, seed: int = 42) -> pd.DataFrame:
    """Keep whole patients so that roughly max_images images remain, preserving split proportions."""
    if max_images >= len(df):
        return df
    rng = np.random.RandomState(seed)
    keep: List[int] = []
    for s in SPLITS:
        part = df[df["split"] == s]
        if part.empty:
            continue
        target = max(1, round(max_images * len(part) / len(df)))
        counts = part.groupby("subject_id").size()
        order = rng.permutation(counts.index.values)
        total = 0
        for sid in order:
            if total >= target:
                break
            keep.append(sid)
            total += counts[sid]
    return df[df["subject_id"].isin(keep)]


def summarize_index(df: pd.DataFrame, labels: Sequence[str], n_listed: int) -> Dict:
    """Counts per split/patient and train label counts, for logging and the data_summary.json."""
    return {
        "images_listed_for_views": n_listed,
        "images_used": len(df),
        "patients": int(df["subject_id"].nunique()),
        "per_split": {s: {"images": int((df["split"] == s).sum()),
                          "patients": int(df.loc[df["split"] == s, "subject_id"].nunique())} for s in SPLITS},
        "label_counts_train": {
            l: {"positive": int((df.loc[df["split"] == "train", l] == 1).sum()),
                "negative": int((df.loc[df["split"] == "train", l] == 0).sum()),
                "uncertain": int((df.loc[df["split"] == "train", l] == -1).sum()),
                "not_mentioned": int(df.loc[df["split"] == "train", l].isna().sum())}
            for l in labels},
    }


def build_index(root, labels: Sequence[str] = DEFAULT_LABELS, views: Sequence[str] = ("PA", "AP"),
                max_images: Optional[int] = None, seed: int = 42,
                only_existing: bool = True) -> Tuple[pd.DataFrame, Dict]:
    root = Path(root)
    unknown = [l for l in labels if l not in ALL_LABELS]
    if unknown:
        raise DatasetError(f"Unknown labels {unknown}. Valid: {ALL_LABELS}")
    df = load_tables(root)
    df = df[df["ViewPosition"].isin(list(views))].copy()
    df["path"] = [str(image_path(root, s, st, d)) for s, st, d in zip(df["subject_id"], df["study_id"], df["dicom_id"])]
    n_listed = len(df)
    if only_existing:  # lets you train on a partial download (e.g. only the p10/ folder)
        df = df[[Path(p).is_file() for p in df["path"]]]
    if df.empty:
        raise DatasetError(f"No images found on disk under {root / 'files'} ({n_listed} listed in the CSVs).")
    check_no_patient_leakage(df)
    if max_images:
        df = subset_by_patient(df, max_images, seed)
        check_no_patient_leakage(df)
    df = df.reset_index(drop=True)
    summary = summarize_index(df, labels, n_listed)
    return df, summary


def make_targets(df: pd.DataFrame, labels: Sequence[str], uncertain: str = "ones",
                 training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y, mask), both float32 [N, L]. mask=0 means the entry is excluded from loss/metrics.

    - 1.0 -> positive; 0.0 -> negative.
    - blank (label not mentioned in the report) -> negative (standard CheXpert practice; this is
      noisy because absence of a mention is not proof of absence).
    - -1.0 (uncertain): per `uncertain` policy when training; ALWAYS masked out when evaluating,
      so reported metrics never depend on how uncertainty was resolved.
    """
    if uncertain not in UNCERTAIN_POLICIES:
        raise ValueError(f"uncertain must be one of {UNCERTAIN_POLICIES}")
    raw = df[list(labels)].to_numpy(dtype="float64")
    unc = raw == -1.0
    y = np.where(np.isnan(raw), 0.0, raw)
    mask = np.ones_like(y)
    if training and uncertain == "ones":
        y = np.where(unc, 1.0, y)
    elif training and uncertain == "zeros":
        y = np.where(unc, 0.0, y)
    else:
        y = np.where(unc, 0.0, y)
        mask = np.where(unc, 0.0, 1.0)
    return y.astype("float32"), mask.astype("float32")
