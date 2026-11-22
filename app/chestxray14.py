
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
