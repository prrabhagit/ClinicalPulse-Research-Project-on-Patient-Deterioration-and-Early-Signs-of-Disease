import gzip

import numpy as np
import pandas as pd
import pytest

from clinicalpulse.imaging import mimic_data as md


def make_fake_mimic(root, n_patients=30, seed=0, create_images=True):
    """Tiny synthetic dataset in the real PhysioNet layout (no real patient data)."""
    rng = np.random.RandomState(seed)
    rows_split, rows_meta, rows_lab = [], [], []
    for i in range(n_patients):
        sid = 10000000 + i
        split = "train" if i < 20 else ("validate" if i < 25 else "test")
        for k in range(2):  # two studies per patient
            stid = 50000000 + i * 10 + k
            lab = {l: rng.choice([1.0, 0.0, -1.0, np.nan]) for l in md.ALL_LABELS}
            rows_lab.append({"subject_id": sid, "study_id": stid, **lab})
            for v in ("PA", "LATERAL"):
                did = f"d{sid}-{stid}-{v}"
                rows_split.append({"dicom_id": did, "study_id": stid, "subject_id": sid, "split": split})
                rows_meta.append({"dicom_id": did, "ViewPosition": v})
                if create_images:
                    p = md.image_path(root, sid, stid, did)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"x")
    for name, rows in (("split", rows_split), ("metadata", rows_meta), ("chexpert", rows_lab)):
        with gzip.open(root / f"mimic-cxr-2.0.0-{name}.csv.gz", "wt") as f:
            pd.DataFrame(rows).to_csv(f, index=False)


@pytest.fixture
def root(tmp_path):
    make_fake_mimic(tmp_path)
    return tmp_path


def test_image_path_layout(tmp_path):
    p = md.image_path(tmp_path, 10000032, 50414267, "abc")
    assert p.as_posix().endswith("files/p10/p10000032/s50414267/abc.jpg")


def test_index_keeps_frontal_only_and_official_splits(root):
    df, s = md.build_index(root, md.DEFAULT_LABELS)
    assert set(df["ViewPosition"]) == {"PA"}
    assert s["per_split"]["train"]["patients"] == 20
    assert s["per_split"]["validate"]["patients"] == 5
    assert s["per_split"]["test"]["patients"] == 5
    md.check_no_patient_leakage(df)


def test_lateral_can_be_requested(root):
    df, _ = md.build_index(root, md.DEFAULT_LABELS, views=("PA", "LATERAL"))
    assert set(df["ViewPosition"]) == {"PA", "LATERAL"}


def test_partial_download_only_existing_images(root):
    # delete every image of the test patients: they must silently drop out of the index (documented)
    for p in (root / "files").rglob("*.jpg"):
        if int(p.parts[-3][1:]) >= 10000025:
            p.unlink()
    df, s = md.build_index(root, md.DEFAULT_LABELS)
    assert s["per_split"]["test"]["images"] == 0
    assert s["images_used"] < s["images_listed_for_views"]


def test_no_images_on_disk_raises(tmp_path):
    make_fake_mimic(tmp_path, create_images=False)
    with pytest.raises(md.DatasetError, match="No images found"):
        md.build_index(tmp_path)


def test_missing_csv_gives_clear_error(tmp_path):
    with pytest.raises(md.DatasetError, match="not found"):
        md.build_index(tmp_path)


def test_unknown_label_rejected(root):
    with pytest.raises(md.DatasetError, match="Unknown labels"):
        md.build_index(root, ["Cancer"])


def test_leakage_detected():
    df = pd.DataFrame({"subject_id": [1, 1], "split": ["train", "test"]})
    with pytest.raises(md.DatasetError, match="leakage"):
        md.check_no_patient_leakage(df)


def test_subset_keeps_whole_patients_and_all_splits(root):
    df, _ = md.build_index(root, md.DEFAULT_LABELS)
    sub = md.subset_by_patient(df, max_images=20, seed=1)
    assert 0 < len(sub) < len(df)
    # every kept patient keeps all of its images
    full = df.groupby("subject_id").size()
    assert (sub.groupby("subject_id").size() == full[sub["subject_id"].unique()]).all()
    assert set(sub["split"]) == {"train", "validate", "test"}
    md.check_no_patient_leakage(sub)
    # deterministic
    pd.testing.assert_frame_equal(sub, md.subset_by_patient(df, max_images=20, seed=1))


def test_subset_noop_when_large(root):
    df, _ = md.build_index(root, md.DEFAULT_LABELS)
    assert len(md.subset_by_patient(df, 10**6)) == len(df)


def _labels_df():
    # one row per case: positive, negative, uncertain, not-mentioned
    return pd.DataFrame({"A": [1.0, 0.0, -1.0, np.nan]})


@pytest.mark.parametrize("policy,expected_y,expected_m", [
    ("ones", [1, 0, 1, 0], [1, 1, 1, 1]),
    ("zeros", [1, 0, 0, 0], [1, 1, 1, 1]),
    ("ignore", [1, 0, 0, 0], [1, 1, 0, 1]),
])
def test_targets_training_policies(policy, expected_y, expected_m):
    y, m = md.make_targets(_labels_df(), ["A"], policy, training=True)
    assert y[:, 0].tolist() == expected_y and m[:, 0].tolist() == expected_m


@pytest.mark.parametrize("policy", md.UNCERTAIN_POLICIES)
def test_targets_eval_always_masks_uncertain(policy):
    y, m = md.make_targets(_labels_df(), ["A"], policy, training=False)
    assert m[:, 0].tolist() == [1, 1, 0, 1]


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        md.make_targets(_labels_df(), ["A"], "maybe")
