import numpy as np
import pytest

from clinicalpulse.imaging.metrics import evaluate, expected_calibration_error, youden_threshold


def test_perfect_separation():
    y = np.array([[0], [0], [1], [1]], dtype=float)
    p = np.array([[0.1], [0.2], [0.8], [0.9]])
    r = evaluate(y, p, np.ones_like(y), ["L"])
    m = r["per_label"]["L"]
    assert m["auroc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0


def test_random_scores_near_chance():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, (4000, 1)).astype(float)
    p = rng.rand(4000, 1)
    assert abs(evaluate(y, p, np.ones_like(y), ["L"])["per_label"]["L"]["auroc"] - 0.5) < 0.05


def test_masked_entries_excluded():
    y = np.array([[0], [0], [1], [1], [1]], dtype=float)
    p = np.array([[0.1], [0.2], [0.8], [0.9], [0.0]])  # last sample would hurt, but is masked
    mask = np.array([[1], [1], [1], [1], [0]], dtype=float)
    r = evaluate(y, p, mask, ["L"])["per_label"]["L"]
    assert r["n"] == 4 and r["auroc"] == 1.0


def test_single_class_label_reported_as_undefined_not_crashing():
    y = np.zeros((10, 1))
    r = evaluate(y, np.random.rand(10, 1), np.ones_like(y), ["L"])
    assert "note" in r["per_label"]["L"] and r["macro"] == {}


def test_given_thresholds_are_applied_unchanged():
    y = np.array([[0], [0], [1], [1]], dtype=float)
    p = np.array([[0.1], [0.2], [0.8], [0.9]])
    r = evaluate(y, p, np.ones_like(y), ["L"], thresholds={"L": 0.95})
    assert r["per_label"]["L"]["sensitivity"] == 0.0 and r["per_label"]["L"]["threshold"] == 0.95


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    y = np.array([0, 0, 0, 1] * 250, dtype=float)
    assert expected_calibration_error(y, np.full(1000, 0.25)) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(y, np.full(1000, 0.95)) > 0.6


def test_youden_threshold_between_classes():
    y = np.array([0] * 50 + [1] * 50, dtype=float)
    p = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert 0.35 <= youden_threshold(y, p) <= 0.65
