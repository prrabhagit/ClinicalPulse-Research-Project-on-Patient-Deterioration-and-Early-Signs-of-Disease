import shutil

import pytest

from clinicalpulse.reports import ReportReadError, analyze_report, scan_report
from clinicalpulse.reports.schema import (CONFLICTING, HISTORICAL, NEGATED, PRESENT, UNCERTAIN)

PAD = "Chest radiograph report. "  # analyze_report rejects text shorter than 20 chars


def state(text, entity):
    return analyze_report(PAD + text).findings.get(entity)


def test_simple_present():
    assert state("Right lower lobe consolidation.", "Consolidation") == PRESENT


def test_negation_list_scope():
    t = "No focal consolidation, pleural effusion, or pneumothorax."
    for e in ("Consolidation", "Effusion", "Pneumothorax"):
        assert state(t, e) == NEGATED


def test_negation_scope_stops_at_but():
    t = "No pneumothorax but small pleural effusion is present."
    assert state(t, "Pneumothorax") == NEGATED
    assert state(t, "Effusion") == PRESENT


def test_pseudo_negation_no_change_keeps_finding_present():
    assert state("No change in the pleural effusion.", "Effusion") == PRESENT


def test_uncertainty_pre_and_post():
    assert state("Cannot exclude small nodule.", "Nodule") == UNCERTAIN
    assert state("Pneumonia cannot be excluded.", "Pneumonia") == UNCERTAIN
    assert state("Atelectasis versus pneumonia.", "Atelectasis") == UNCERTAIN


def test_not_excluded_is_not_negation():
    assert state("Pneumonia is not excluded.", "Pneumonia") == UNCERTAIN


def test_ruled_out_post_negation():
    assert state("Pneumothorax was ruled out.", "Pneumothorax") == NEGATED


def test_historical_and_resolved():
    assert state("History of pneumonia.", "Pneumonia") == HISTORICAL
    assert state("The effusion has resolved.", "Effusion") == HISTORICAL
    assert state("The effusion is no longer seen.", "Effusion") == HISTORICAL


def test_comparison_section_is_historical():
    r = analyze_report("COMPARISON: Prior film showed small pleural effusion.\nFINDINGS: Lungs clear.")
    assert r.findings["Effusion"] == HISTORICAL


def test_pericardial_effusion_not_pleural():
    assert state("Small pericardial effusion.", "Effusion") is None


def test_contradiction_flagged():
    r = analyze_report("FINDINGS: No pneumonia.\nIMPRESSION: Right lower lobe pneumonia.")
    assert r.findings["Pneumonia"] == CONFLICTING
    assert r.contradictions and "Pneumonia" in r.contradictions[0]
    assert r.features()["report_Pneumonia"] == -1.0


def test_features_omit_unmentioned_and_historical():
    r = analyze_report("FINDINGS: History of pneumonia. No pneumothorax. Mild cardiomegaly.")
    f = r.features()
    assert f == {"report_Pneumothorax": 0.0, "report_Cardiomegaly": 1.0}


def test_overlap_prefers_longer_match():
    r = analyze_report(PAD + "Patient reports coughing up blood.")
    assert r.findings.get("Hemoptysis") == PRESENT
    assert "Cough" not in r.findings


def test_vitals_extracted_and_converted():
    r = analyze_report(PAD + "HR 112, BP 128/76, SpO2 91% on RA, Temp 101.5 F, RR 24.")
    v = {x.name: x.value for x in r.vitals}
    assert v["heart_rate"] == 112 and v["spo2"] == 91 and v["respiratory_rate"] == 24
    assert (v["systolic_bp"], v["diastolic_bp"]) == (128, 76)
    assert v["temperature_c"] == pytest.approx(38.6, abs=0.1)


def test_implausible_vitals_rejected_with_warning():
    r = analyze_report(PAD + "SpO2 250%. BP 70/120. Temp 55.")
    assert not r.vitals
    assert len(r.warnings) >= 3


def test_pulse_ox_not_read_as_heart_rate():
    r = analyze_report(PAD + "Pulse ox 94% on room air.")
    names = [x.name for x in r.vitals]
    assert "spo2" in names and "heart_rate" not in names


def test_missing_sections_and_entities_warn():
    r = analyze_report("Patient seen today for routine paperwork only.")
    assert any("FINDINGS or IMPRESSION" in w for w in r.warnings)
    assert any("No supported clinical entities" in w for w in r.warnings)


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        analyze_report("   ")


# ---------------------------- file reading ----------------------------
def test_scan_txt(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("IMPRESSION: Mild cardiomegaly. No pneumothorax.", encoding="utf-8")
    r = scan_report(p)
    assert r.method == "text" and r.findings["Cardiomegaly"] == PRESENT


def test_non_utf8_txt_warns(tmp_path):
    p = tmp_path / "r.txt"
    p.write_bytes("IMPRESSION: Mild cardiomegaly – stable.".encode("cp1252"))
    assert any("Windows-1252" in w for w in scan_report(p).warnings)


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(ReportReadError):
        scan_report(tmp_path / "missing.txt")
    (tmp_path / "e.txt").write_text("")
    with pytest.raises(ReportReadError):
        scan_report(tmp_path / "e.txt")
    (tmp_path / "x.docx").write_bytes(b"abc")
    with pytest.raises(ReportReadError):
        scan_report(tmp_path / "x.docx")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ReportReadError):
        scan_report(tmp_path / "bad.pdf")


def _make_text_pdf(path, lines):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 740
    for ln in lines:
        c.drawString(72, y, ln)
        y -= 18
    c.save()


def test_scan_text_pdf(tmp_path):
    p = tmp_path / "r.pdf"
    _make_text_pdf(p, ["FINDINGS: Right lower lobe consolidation.", "No pleural effusion.",
                       "IMPRESSION: Pneumonia."])
    r = scan_report(p)
    assert r.method == "pdf_text"
    assert r.findings["Consolidation"] == PRESENT and r.findings["Effusion"] == NEGATED


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_scan_image_only_pdf_uses_ocr(tmp_path):
    pytest.importorskip("pytesseract")
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    img = Image.new("RGB", (1700, 500), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)
    except OSError:
        pytest.skip("no TrueType font available to render the scan")
    d.text((40, 40), "FINDINGS: Right lower lobe consolidation.", fill="black", font=font)
    d.text((40, 160), "No pneumothorax.", fill="black", font=font)
    d.text((40, 280), "IMPRESSION: Pneumonia.", fill="black", font=font)
    png = tmp_path / "scan.png"
    img.save(png)
    pdf = tmp_path / "scan.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.drawImage(str(png), 36, 400, width=540, height=159)
    c.save()
    r = scan_report(pdf)
    assert r.method == "ocr" and any("OCR" in w for w in r.warnings)
    assert r.findings.get("Consolidation") == PRESENT
    assert r.findings.get("Pneumothorax") == NEGATED
