
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding",
]
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



