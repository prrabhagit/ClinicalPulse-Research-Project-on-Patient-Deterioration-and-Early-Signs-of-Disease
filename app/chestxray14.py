
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
    


