import pytest
from fastapi.testclient import TestClient

from app.server import app
from clinicalpulse.fusion import summarize_case

V_BAD = {"latest": {"components": {"spo2": 3, "respiratory_rate": 2, "on_oxygen": 2}, "total": 9, "band": "high",
                    "complete": True, "missing": []}, "flags": ["spo2 falling (-5.0 over 4.0 h)"]}
V_OK = {"latest": {"components": {"spo2": 0, "respiratory_rate": 0}, "total": 0, "band": "low",
                   "complete": True, "missing": []}, "flags": []}
REP = {"findings": {"Pneumonia": "present", "Pneumothorax": "negated", "Cough": "present"}, "contradictions": []}
XR = [{"label": "Pneumonia", "probability": 0.7, "above_threshold": True},
      {"label": "Pneumothorax", "probability": 0.4, "above_threshold": True},
      {"label": "Effusion", "probability": 0.1, "above_threshold": False}]


def test_multimodal_review_flag_and_concordance():
    c = summarize_case(V_BAD, REP, XR)
    assert c["triage"]["band"] == "high" and c["flags"][0].startswith("Review suggested")
    verdicts = {x["finding"]: x["verdict"] for x in c["concordance"]}
    assert verdicts["Pneumonia"] == "concordant" and "negates" in verdicts["Pneumothorax"]


def test_no_review_flag_when_vitals_respiratory_normal():
    c = summarize_case(V_OK, REP, XR)
    assert not any(f.startswith("Review suggested") for f in c["flags"])


def test_missing_modalities_stated_and_no_numeric_risk():
    c = summarize_case(V_BAD, None, None)
    assert any("Not assessed" in f and "report" in f and "xray" in f for f in c["flags"])
    assert not any(k in c for k in ("risk", "probability", "risk_score"))
    assert c["limitations"]


def test_report_positive_image_negative_and_alias():
    c = summarize_case(None, {"findings": {"Effusion": "present"}, "contradictions": []},
                       [{"label": "Pleural Effusion", "probability": 0.1, "above_threshold": False}])
    assert c["concordance"][0]["verdict"] == "report positive, image model negative"


def test_missing_thresholds_flagged_and_contradictions_surfaced():
    rep = {"findings": {"Pneumonia": "conflicting"}, "contradictions": ["Pneumonia: report both negates and states present"]}
    c = summarize_case(None, rep, [{"label": "Pneumonia", "probability": 0.9, "above_threshold": None}])
    assert any("no validation thresholds" in f for f in c["flags"]) and any("contradiction" in f for f in c["flags"])


def test_case_endpoint_and_malformed_payload():
    cl = TestClient(app)
    r = cl.post("/api/case", json={"vitals": V_BAD, "report": REP, "xray": XR})
    assert r.status_code == 200 and r.json()["flags"][0].startswith("Review suggested")
    assert cl.post("/api/case", json={"vitals": {"oops": 1}}).status_code == 422
