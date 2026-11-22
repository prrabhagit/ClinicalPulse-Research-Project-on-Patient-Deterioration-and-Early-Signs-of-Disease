import io

import pytest
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)
REPORT = b"FINDINGS: Right lower lobe consolidation. No pneumothorax. HR 112, SpO2 91% on RA, RR 24.\nIMPRESSION: Pneumonia."
CSV = (b"timestamp,heart_rate,respiratory_rate,spo2,systolic_bp,temperature_c,on_oxygen,consciousness\n"
       b"2026-01-01 08:00,72,16,97,122,37.0,no,A\n2026-01-01 10:00,95,20,94,112,37.8,no,A\n"
       b"2026-01-01 12:00,118,26,90,98,38.9,yes,C\n")


def up(name, data):
    return {"file": (name, io.BytesIO(data))}


def test_index_and_health(monkeypatch):
    monkeypatch.delenv("CLINICALPULSE_CKPT", raising=False)
    assert "ClinicalPulse" in client.get("/").text
    assert client.get("/api/health").json() == {"report_scanning": True, "vitals": True, "xray_model": False}


def test_report_endpoint_real_analysis():
    r = client.post("/api/report", files=up("r.txt", REPORT))
    j = r.json()
    assert r.status_code == 200 and j["findings"]["Consolidation"] == "present" and j["findings"]["Pneumothorax"] == "negated"
    assert j["news2_from_report"]["complete"] is False and j["news2_from_report"]["components"]["spo2"] == 3


def test_vitals_endpoint():
    r = client.post("/api/vitals?window_hours=6", files=up("v.csv", CSV))
    j = r.json()
    assert r.status_code == 200 and j["n_observations"] == 3 and j["latest"]["band"] == "high"
    assert any("spo2" in f for f in j["flags"]) or j["news2_change_in_window"] > 0


@pytest.mark.parametrize("path,name,data,code", [
    ("/api/report", "r.docx", b"x", 415), ("/api/report", "r.txt", b"", 400), ("/api/report", "r.txt", b"short", 422),
    ("/api/vitals", "v.csv", b"timestamp,notes\n2026-01-01,x\n", 422), ("/api/vitals", "v.txt", b"x", 415),
    ("/api/xray", "x.png", b"notimage", 503),
])
def test_validation_errors(monkeypatch, path, name, data, code):
    monkeypatch.delenv("CLINICALPULSE_CKPT", raising=False)
    assert client.post(path, files=up(name, data)).status_code == code


def test_xray_never_fabricates_without_checkpoint(monkeypatch):
    monkeypatch.setenv("CLINICALPULSE_CKPT", "/nonexistent/best.pt")
    r = client.post("/api/xray", files=up("x.png", b"\x89PNG"))
    assert r.status_code == 503 and "No X-ray model" in r.json()["detail"]
