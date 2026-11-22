"""NIH ChestX-ray14 WebDataset-style shards -> image folder + labels.csv -> index for training.

Shard format (e.g. ChestXray14_train_000.tar): pairs of members per image
    <patient8>_<followup3>.png   grayscale image
    <patient8>_<followup3>.pyd   pickled numpy float array of shape (14,) with 0/1 labels

The label VECTOR has no names inside the shard. LABELS below is the commonly used NIH order and
is an ASSUMPTION; verify it on your data (README explains a prevalence check) and edit LABELS if
your shards use another order. Pickles are read with a restricted unpickler that only allows numpy
array reconstruction, so a malicious .pyd cannot execute code.
"""
import io
import pickle
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .mimic_data import SPLITS, DatasetError, check_no_patient_leakage, summarize_index

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
_NAME = re.compile(r"^(\d{8})_(\d{3})\.(png|pyd)$")
_SHARD = re.compile(r"ChestXray14_(train|val|test)_\d+\.tar$")
_SPLIT_NAME = {"train": "train", "val": "validate", "test": "test"}
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype")}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a label file")


def read_label_vector(raw: bytes, n_labels: int = len(LABELS)) -> np.ndarray:
    arr = _NumpyOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(arr, np.ndarray) or arr.shape != (n_labels,):
        raise DatasetError(f"Label vector must be a numpy array of shape ({n_labels},), got {getattr(arr, 'shape', type(arr))}")
    if not np.isin(arr, (0, 1)).all():
        raise DatasetError(f"Label vector has values other than 0/1: {np.unique(arr)}")
    return arr.astype("float32")


def expand_shards(paths: Iterable) -> List[Path]:
    """Each entry may be a .tar file or a directory containing ChestXray14_*.tar shards."""
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(p.glob("ChestXray14_*.tar"))
            if not found:
                raise DatasetError(f"No ChestXray14_*.tar shards found in {p}")
            out += found
        elif p.is_file():
            out.append(p)
        else:
            raise DatasetError(f"Shard path not found: {p}")
    return out


def already_extracted(shards: List[Path], out_dir) -> bool:
    """True if out_dir/labels.csv exists and was built from exactly these shard names."""
    csv = Path(out_dir) / "labels.csv"
    if not csv.is_file():
        return False
    done = set(pd.read_csv(csv, usecols=["source_shard"])["source_shard"])
    return done == {p.name for p in shards}


def extract_shards(tar_paths: Iterable, out_dir) -> pd.DataFrame:
    """Unpack shards to out_dir/images/<split>/ and write out_dir/labels.csv. Returns the table."""
    out_dir = Path(out_dir)
    rows: List[dict] = []
    for tp in map(Path, tar_paths):
        m = _SHARD.search(tp.name)
        if not m:
            raise DatasetError(f"Unexpected shard name '{tp.name}' (expected ChestXray14_<train|val|test>_NNN.tar)")
        split = _SPLIT_NAME[m.group(1)]
        img_dir = out_dir / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        pngs: Dict[str, bytes] = {}
        labs: Dict[str, np.ndarray] = {}
        with tarfile.open(tp) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                nm = _NAME.match(Path(member.name).name)
                if not nm:
                    raise DatasetError(f"{tp.name}: unexpected member '{member.name}'")
                key, kind = f"{nm.group(1)}_{nm.group(2)}", nm.group(3)
                data = tf.extractfile(member).read()
                if kind == "png":
                    pngs[key] = data
                else:
                    labs[key] = read_label_vector(data)
        if set(pngs) != set(labs):
            raise DatasetError(f"{tp.name}: {len(set(pngs) ^ set(labs))} images without matching label file (or vice versa)")
        for key in sorted(pngs):
            dest = img_dir / f"{key}.png"
            dest.write_bytes(pngs[key])
            row = {"image": f"images/{split}/{key}.png", "subject_id": int(key[:8]), "follow_up": int(key[9:]),
                   "split": split, "source_shard": tp.name}
            row.update({l: float(v) for l, v in zip(LABELS, labs[key])})
            rows.append(row)
    if not rows:
        raise DatasetError("No shards given.")
    df = pd.DataFrame(rows).sort_values(["split", "subject_id", "follow_up"]).reset_index(drop=True)
    df.to_csv(out_dir / "labels.csv", index=False)
    return df


def build_index(root, labels: Sequence[str] = LABELS, max_images=None, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """Index an extracted folder (see extract_shards). Same return shape as mimic_data.build_index."""
    root = Path(root)
    csv = root / "labels.csv"
    if not csv.is_file():
        raise DatasetError(f"{csv} not found. Run: python -m clinicalpulse.imaging.chestxray14 extract --shards ... --out {root}")
    unknown = [l for l in labels if l not in LABELS]
    if unknown:
        raise DatasetError(f"Unknown labels {unknown}. Valid: {LABELS}")
    df = pd.read_csv(csv)
    df["path"] = [str(root / p) for p in df["image"]]
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise DatasetError(f"{len(missing)} images listed in labels.csv are missing, e.g. {missing[0]}")
    check_no_patient_leakage(df)
    n_listed = len(df)
    if max_images:
        from .mimic_data import subset_by_patient
        df = subset_by_patient(df, max_images, seed).reset_index(drop=True)
    return df, summarize_index(df, labels, n_listed)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="clinicalpulse.imaging.chestxray14")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="unpack .tar shards into an image folder + labels.csv")
    e.add_argument("--shards", nargs="+", required=True, help=".tar files and/or folders containing them")
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        df = extract_shards(expand_shards(a.shards), a.out)
        check_no_patient_leakage(df)
    except DatasetError as err:
        raise SystemExit(f"error: {err}")
    print(f"Wrote {len(df)} images to {a.out}")
    print(df.groupby("split").agg(images=("image", "size"), patients=("subject_id", "nunique")))
    print("Label prevalence per position (verify against your expected label order):")
    print(df[LABELS].mean().round(4).to_string())


if __name__ == "__main__":
    main()
