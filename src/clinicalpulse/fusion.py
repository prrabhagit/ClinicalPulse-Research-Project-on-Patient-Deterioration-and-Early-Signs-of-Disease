"""Evidence aggregation across modalities (X-ray model, report scan, vital signs).

This is NOT a trained fusion model and does NOT output a risk probability. A learned deterioration
model needs linked image + report + vitals data with a defined outcome and horizon, which this
project does not have yet. Until then this module only (1) shows the established NEWS2 triage band,
(2) lists the evidence each modality contributes, (3) checks agreement between report and image
model, and (4) raises a review flag when a vital-sign respiratory abnormality coincides with
respiratory imaging/report evidence. The rules are transparent heuristics, unvalidated.

Inputs are the JSON-shaped outputs of the vitals, report and X-ray endpoints; any may be None.
"""
from typing import Dict, List, Optional

RESP_FINDINGS = {"Atelectasis", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Infiltration",
                 "Mass", "Nodule", "Opacity", "Pleural_Thickening", "Pneumonia", "Pneumothorax"}
_ALIAS = {"Pleural Effusion": "Effusion", "Lung Opacity": "Opacity", "Pleural Thickening": "Pleural_Thickening"}
LIMITATIONS = [
    "No validated multimodal risk model exists in this project; no deterioration probability is produced.",
    "NEWS2 is the only established score shown; image-model output is a research estimate from a small dataset.",
    "Report analysis is rule-based and can misread unusual wording; verify against the source report.",
]


def _canon(label: str) -> str:
    return _ALIAS.get(label, label.replace(" ", "_"))


def _vital_respiratory_evidence(v: Dict) -> List[str]:
    out = []
    comp = v["latest"]["components"]
    for k, name in (("spo2", "SpO2"), ("respiratory_rate", "respiratory rate")):
        if comp.get(k, 0) >= 1:
            out.append(f"{name} scores {comp[k]} on NEWS2")
    if comp.get("on_oxygen", 0):
        out.append("on supplemental oxygen")
    out += [f for f in v.get("flags", []) if f.startswith(("spo2", "respiratory_rate"))]
    return out


def summarize_case(vitals: Optional[Dict], report: Optional[Dict], xray: Optional[List[Dict]]) -> Dict:
    avail = {"vitals": bool(vitals), "report": bool(report), "xray": bool(xray)}
    out: Dict = {"data_availability": avail, "triage": None, "evidence": [], "concordance": [],
                 "flags": [], "limitations": LIMITATIONS}
    vit_resp: List[str] = []
    if vitals:
        L = vitals["latest"]
        out["triage"] = {"news2": L["total"], "band": L["band"], "complete": L["complete"], "missing": L["missing"]}
        vit_resp = _vital_respiratory_evidence(vitals)
        for e in vit_resp:
            out["evidence"].append({"source": "vitals", "statement": e})
        if not L["complete"]:
            out["flags"].append(f"NEWS2 incomplete (missing {', '.join(L['missing'])}): score is a lower bound.")

    rep_pos: Dict[str, str] = {}
    rep_neg = set()
    if report:
        for ent, st in report["findings"].items():
            if st in ("present", "uncertain", "conflicting") and (ent in RESP_FINDINGS):
                rep_pos[ent] = st
                out["evidence"].append({"source": "report", "statement": f"{ent.replace('_', ' ')}: {st}"})
            elif st == "negated":
                rep_neg.add(ent)
        out["flags"] += [f"Report contradiction: {c}" for c in report.get("contradictions", [])]

    img_pos: Dict[str, float] = {}
    if xray:
        for p in xray:
            lab = _canon(p["label"])
            if p.get("above_threshold") and lab in RESP_FINDINGS:
                img_pos[lab] = p["probability"]
                out["evidence"].append({"source": "xray_model",
                                        "statement": f"{lab.replace('_', ' ')} above validation threshold (p={p['probability']:.2f})"})
        if all(p.get("above_threshold") is None for p in xray):
            out["flags"].append("X-ray model has no validation thresholds; 'above threshold' cannot be judged.")

    if report and xray:
        known = {_canon(p["label"]): p for p in xray}
        for lab in sorted(set(rep_pos) | set(img_pos) | (rep_neg & set(img_pos))):
            if lab not in known:
                continue
            r = rep_pos.get(lab) or ("negated" if lab in rep_neg else None)
            i = "positive" if lab in img_pos else "negative"
            if r in ("present", "uncertain") and i == "positive":
                verdict = "concordant"
            elif r == "negated" and i == "positive":
                verdict = "image model positive, report negates"
            elif r in ("present", "uncertain") and i == "negative":
                verdict = "report positive, image model negative"
            else:
                continue
            out["concordance"].append({"finding": lab, "report": r, "image_model": i, "verdict": verdict})
        out["flags"] += [f"{c['finding']}: {c['verdict']}" for c in out["concordance"] if c["verdict"] != "concordant"]

    imaging_or_report = sorted(set(rep_pos) | set(img_pos))
    if vit_resp and imaging_or_report:
        out["flags"].insert(0, "Review suggested: respiratory vital-sign abnormality coincides with "
                               f"report/image evidence ({', '.join(f.replace('_', ' ') for f in imaging_or_report)}). "
                               "Not a diagnosis or a risk estimate.")
    missing = [k for k, v in avail.items() if not v]
    if missing:
        out["flags"].append(f"Not assessed (no data): {', '.join(missing)}.")
    return out
