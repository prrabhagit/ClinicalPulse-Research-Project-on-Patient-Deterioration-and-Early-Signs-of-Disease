import io
import os
import pickle
import tarfile

import numpy as np
import pytest
from PIL import Image

from clinicalpulse.imaging import chestxray14 as cx
from clinicalpulse.imaging.mimic_data import DatasetError


def png_bytes(size=16):
    b = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (size, size), dtype="uint8"), "L").save(b, "PNG")
    return b.getvalue()


def make_shard(path, items):
    """items: list of (name_stem, label_vector_or_raw_bytes)."""
    with tarfile.open(path, "w") as tf:
        for stem, lab in items:
            for ext, data in ((".png", png_bytes()),
                              (".pyd", lab if isinstance(lab, bytes) else pickle.dumps(np.asarray(lab, dtype="float64")))):
                info = tarfile.TarInfo(stem + ext)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


def vec(*positive):
    v = np.zeros(14)
    v[list(positive)] = 1
    return v


@pytest.fixture
def shards(tmp_path):
    make_shard(tmp_path / "ChestXray14_train_000.tar", [("00000004_000", vec(0)), ("00000004_001", vec(0, 2)), ("00000007_000", vec())])
    make_shard(tmp_path / "ChestXray14_val_002.tar", [("00000100_000", vec(3))])
    make_shard(tmp_path / "ChestXray14_test_000.tar", [("00000200_000", vec(13))])
    return sorted(tmp_path.glob("*.tar"))


def test_extract_and_index_roundtrip(shards, tmp_path):
    out = tmp_path / "out"
    df = cx.extract_shards(shards, out)
    assert len(df) == 5 and set(df["split"]) == {"train", "validate", "test"}
    idx, summary = cx.build_index(out)
    assert summary["per_split"]["train"] == {"images": 3, "patients": 2}
    r = idx[(idx.subject_id == 4) & (idx.follow_up == 1)].iloc[0]
    assert r["Atelectasis"] == 1 and r["Effusion"] == 1 and r["Mass"] == 0
    assert idx[idx.split == "test"].iloc[0]["Hernia"] == 1
    assert all(os.path.isfile(p) for p in idx["path"])


def test_restricted_unpickler_refuses_code_execution():
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > /tmp/pwned_marker",))

    with pytest.raises(pickle.UnpicklingError):
        cx.read_label_vector(pickle.dumps(Evil()))
    assert not os.path.exists("/tmp/pwned_marker")


@pytest.mark.parametrize("arr", [np.zeros(13), np.full(14, 0.5), np.zeros((14, 1))])
def test_bad_label_vectors_rejected(arr):
    with pytest.raises(DatasetError):
        cx.read_label_vector(pickle.dumps(arr))


def test_image_without_label_rejected(tmp_path):
    p = tmp_path / "ChestXray14_train_000.tar"
    with tarfile.open(p, "w") as tf:
        data = png_bytes()
        info = tarfile.TarInfo("00000004_000.png")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(DatasetError, match="without matching label"):
        cx.extract_shards([p], tmp_path / "o")


def test_unexpected_member_and_shard_names_rejected(tmp_path):
    p = tmp_path / "ChestXray14_train_000.tar"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("../evil.png")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(DatasetError, match="unexpected member"):
        cx.extract_shards([p], tmp_path / "o")
    with pytest.raises(DatasetError, match="Unexpected shard name"):
        cx.extract_shards([tmp_path / "random.tar"], tmp_path / "o2")


def test_patient_leakage_across_shards_detected(tmp_path):
    make_shard(tmp_path / "ChestXray14_train_000.tar", [("00000004_000", vec(0))])
    make_shard(tmp_path / "ChestXray14_test_000.tar", [("00000004_001", vec(1))])  # same patient 4
    cx.extract_shards(sorted(tmp_path.glob("*.tar")), tmp_path / "o")
    with pytest.raises(DatasetError, match="leakage"):
        cx.build_index(tmp_path / "o")


def test_missing_labels_csv_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetError, match="extract"):
        cx.build_index(tmp_path)


def test_expand_shards_and_skip_reextraction(shards, tmp_path):
    folder = shards[0].parent
    assert cx.expand_shards([folder]) == sorted(folder.glob("ChestXray14_*.tar"))
    out = tmp_path / "out"
    assert not cx.already_extracted(shards, out)
    cx.extract_shards(shards, out)
    assert cx.already_extracted(shards, out)
    assert not cx.already_extracted(shards[:2], out)  # different shard set -> re-extract
    with pytest.raises(DatasetError):
        cx.expand_shards([tmp_path / "missing"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="No ChestXray14"):
        cx.expand_shards([tmp_path / "empty"])
