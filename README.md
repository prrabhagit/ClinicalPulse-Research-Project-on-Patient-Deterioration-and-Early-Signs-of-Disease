# ClinicalPulse

Multimodal clinical decision-support / early-warning **research project** combining
chest X-ray analysis, medical-report analysis and vital-sign trends. It is not a medical
device and makes no diagnostic claims.

## Status (what actually exists)

| Module | State |
|---|---|
| Report scanning (`clinicalpulse.reports`) | Working, 23 passing tests. Rule-based baseline, not yet validated on real annotated reports. |
| X-ray classifier (`clinicalpulse.imaging`) | Code written, **never trained or run end-to-end** (no dataset or GPU in the build environment). Currently single-label `ImageFolder`. |
| Vital-sign time-series model | Not started. Report text vitals are extracted, but there is no trend model. |
| Multimodal fusion / deterioration risk | Not started. Needs a dataset with linked image, text and vitals (e.g. MIMIC-CXR + MIMIC-IV, credentialed access). |
| Dashboard / API (`app/`) | Not started. |

## Setup
    pip install -e ".[imaging,ocr,dev]"      # or: pip install -r requirements.txt && export PYTHONPATH=src
    pytest

## Scan a report (.txt, .pdf, scanned PDF, or image)
    python -m clinicalpulse.reports scan report.pdf          # readable summary
    python -m clinicalpulse.reports scan report.pdf --json   # full structured output

Output per report: sections found, each finding/symptom with an assertion status
(`present`, `negated`, `uncertain`, `historical`, `conflicting`), the cue that decided it,
vital signs, contradictions, and warnings. `features()` gives a CheXpert-style encoding
(1 present, 0 negated, -1 uncertain); unmentioned and historical findings are omitted, as
missing is not negative. Finding names follow ChestX-ray14 labels so they can later be
joined with image-model outputs.

Known limits: rule-based, so long or unusual sentences can be misread (e.g. "no acute
findings ... to suggest pneumonia" is read as uncertain, not negated). OCR output must be
verified. Evaluate on annotated reports before using it for anything real.

## Train the X-ray model
    python -m clinicalpulse.imaging.xray_classifier train --data data --arch densenet121 --epochs 15
    python -m clinicalpulse.imaging.xray_classifier evaluate --data data --ckpt runs/clinicalpulse-cxr/best.pt
    python -m clinicalpulse.imaging.xray_classifier predict --ckpt runs/clinicalpulse-cxr/best.pt --image xray.jpg --gradcam
    python -m clinicalpulse.imaging.plot_history runs/clinicalpulse-cxr

The data must currently be `data/{train,val,test}/<CLASS>/*.jpeg`. **NIH ChestX-ray14 is
multi-label with one CSV, so it needs a different loader (not written yet).** Splits must be
by patient, not by image.

## Layout
    src/clinicalpulse/reports/   report reading (txt/pdf/OCR), entity + assertion extraction, vitals
    src/clinicalpulse/imaging/   X-ray training / evaluation / Grad-CAM
    tests/                       pytest suite for reports
    app/                         (empty) future API + dashboard
    data/, runs/                 local only, git-ignored
