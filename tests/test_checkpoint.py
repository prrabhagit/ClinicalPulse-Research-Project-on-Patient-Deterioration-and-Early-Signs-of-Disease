import argparse

import pytest

torch = pytest.importorskip("torch")

from clinicalpulse.imaging.train_cxr import checkpoint_config, load_for_inference, predict_image  # noqa: E402
from clinicalpulse.imaging.xray_classifier import build_model  # noqa: E402


def test_checkpoint_roundtrip_loads_with_default_weights_only(tmp_path):
    from PIL import Image
    args = argparse.Namespace(dataset="chestxray14", arch="resnet18", fn=lambda a: None, root="x", labels=None, lr=1e-3)
    labels = ["A", "B", "C"]
    model = build_model("resnet18", len(labels), pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "arch": "resnet18", "labels": labels,
                "config": checkpoint_config(args), "thresholds": {"A": 0.3}}, path)
    assert "fn" not in checkpoint_config(args)
    m, lab, arch, thr, device = load_for_inference(path)  # default torch.load must succeed
    rows = predict_image(m, lab, thr, device, Image.new("L", (64, 64), 128))
    assert {r["label"] for r in rows} == set(labels) and all(0 <= r["probability"] <= 1 for r in rows)
    assert [r for r in rows if r["label"] == "B"][0]["above_threshold"] is None
