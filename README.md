# ClinicalPulse

Multimodal clinical decision-support / early-warning **research project** combining
chest X-ray analysis, medical-report analysis and vital-sign trends. It is not a medical
device and makes no diagnostic claims.

## Quickstart: train now

The three ChestX-ray14 shards you provided are in `data/shards/`, and ImageNet weights for offline use are in `weights/`.

    scripts/setup.sh              # Windows: scripts\setup.bat   (for an NVIDIA GPU install CUDA PyTorch first)
    scripts/train.sh              # DenseNet121, torchvision weights (downloads once), 15 epochs, GPU recommended
    scripts/train.sh offline      # ResNet18 with bundled weights, no download (works on CPU, slow)
    scripts/dashboard.sh          # http://localhost:8000, uses runs/clinicalpulse-cxr/best.pt

Put more `ChestXray14_*.tar` shards into `data/shards/` and re-run `train.sh`; they are extracted automatically (patient-level
split and leakage check included). Override with `EPOCHS=30 BATCH=64 WORKERS=8 OUT=runs/my-run scripts/train.sh`.
Results land in `OUT`: `best.pt`, `test_metrics.json`, `history.json`, `data_summary.json`.
`runs/baseline-resnet18-cxr14/` holds the weak CPU baseline trained here (test macro-AUROC 0.60) for comparison.
The offline script path was run end to end here (1 epoch on CPU); the GPU/DenseNet path was not run here.

## Status (what actually exists)

| Module | State |
|---|---|
| Report scanning (`clinicalpulse.reports`) | Working, 23 passing tests. Rule-based baseline, not yet validated on real annotated reports. |
| MIMIC-CXR-JPG data pipeline + metrics (`imaging/mimic_data.py`, `metrics.py`) | Working on synthetic data, covered by tests (official patient-level splits, leakage check, patient-level subsetting, uncertainty handling, AUROC/sens/spec/F1/ECE). Never run on the real dataset. |
| Multi-label trainer (`imaging/train_cxr.py`) | Ran end to end on 3 small ChestX-ray14 shards (CPU, ResNet18, ImageNet init, 4 epochs): val macro-AUROC 0.63, **test macro-AUROC 0.60** (1000 images; rare labels have a handful of positives). Weak and badly calibrated (ECE ~0.2-0.3 from positive-class weighting); a proof the pipeline works, not a usable model. MIMIC path never run on real data. |
| Dashboard + API (`app/`) | FastAPI endpoints tested. Page driven end to end in jsdom against the live server (report, vitals, X-ray with a real checkpoint, evidence summary): no errors. **Not yet viewed in a real browser** (layout unchecked). No risk score is shown, by design. |
| Single-label ImageFolder trainer (`imaging/xray_classifier.py`) | Older generic trainer, never run. Kept for folder-style datasets. |
| Vital-sign scoring + trends (`clinicalpulse.vitals`) | Working, tested. NEWS2 (RCP 2017 chart, rule-based, not ML) per observation, plus windowed trend flags. Trend cut-offs are untuned heuristics. No learned time-series model yet. |
| Evidence aggregation (`clinicalpulse.fusion`) | Working, tested. Rule-based: NEWS2 triage, per-source evidence, report-vs-image agreement, review flag when respiratory vitals coincide with imaging/report evidence. **Not a risk model; no probability is output.** |
| Learned fusion / deterioration risk | Not started; blocked on linked MIMIC-IV + MIMIC-CXR data. Plan: `docs/deterioration_model_plan.md`. |

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

## Score vital signs (NEWS2) and trends
    python -m clinicalpulse.vitals score observations.csv [--window 6] [--spo2-scale 2] [--json]

CSV columns: `timestamp` (required), `heart_rate, respiratory_rate, spo2, systolic_bp,
temperature_c, on_oxygen, consciousness (A/C/V/P/U)`. Bad rows raise clear errors; implausible or
non-numeric values become missing with a warning; **missing values are never imputed**, and an
incomplete NEWS2 is labelled a lower bound, not a clearance. SpO2 Scale 2 is only valid for
clinician-confirmed hypercapnic respiratory failure. Trend flags fire when the fitted change over the
window passes a configurable amount (`DEFAULT_CONCERN` in `vitals/trends.py`); those amounts are
project heuristics and need validation on real data. `scan` on a report also prints a partial NEWS2
from the vitals written in the report text.

NEWS2 is the baseline any learned deterioration model must beat; it is not a prediction of a future event.

## Train on NIH ChestX-ray14 shards (`ChestXray14_{train,val,test}_NNN.tar`)
    python -m clinicalpulse.imaging.train_cxr train --dataset chestxray14 --root data/cxr14 \
        --shards ChestXray14_train_000.tar ChestXray14_val_002.tar ChestXray14_test_000.tar --arch densenet121
Each shard holds `<patient>_<followup>.png` + a pickled 14-element 0/1 label vector, read with a restricted
unpickler. The label NAMES are not in the files: `chestxray14.LABELS` assumes the common NIH order, which matched
published prevalences on the shards (Infiltration ~18%, Effusion ~11%, Atelectasis ~10%, Hernia <1%); re-check if
your shards differ. Patient overlap across shards is checked. The three uploaded shards hold only 1000/896/1000
images, so per-label test metrics (rare labels: a handful of positives) are very noisy. Use `--arch densenet121`
with torchvision's pretrained weights and a GPU for a real run, or `--init-weights` to supply a state_dict offline.

## Dashboard
    pip install fastapi uvicorn python-multipart
    CLINICALPULSE_CKPT=runs/clinicalpulse-cxr/best.pt uvicorn app.server:app --port 8000   # then open http://localhost:8000
Three real panels (X-ray, report, vitals) plus an evidence summary (`POST /api/case`, from `clinicalpulse.fusion`). Without a checkpoint `/api/xray` returns 503
instead of a made-up prediction. There is no combined deterioration score yet.

## Train on MIMIC-CXR-JPG (multi-label, CheXpert labels)

MIMIC-CXR-JPG is **credentialed-access**: you need a PhysioNet account, credentialing, the
CITI "Data or Specimens Only Research" training and a signed data use agreement, and you must
not share the data. Request access at https://physionet.org/content/mimic-cxr-jpg/ (v2.0.0 and
the newer v2.1.0 both work with this code as long as the file names below exist). The full set
is hundreds of GB of images. To stay in the 30k-100k range, download only some `files/pXX/`
folders (each holds ~1/10 of the patients) plus the three CSVs; the indexer only uses images
present on disk.

Expected layout under `--root`:

    mimic-cxr-2.0.0-metadata.csv.gz   mimic-cxr-2.0.0-split.csv.gz   mimic-cxr-2.0.0-chexpert.csv.gz
    files/p10/p10000032/s50414267/<dicom_id>.jpg ...

    python -m clinicalpulse.imaging.train_cxr train --dataset mimic --root /data/mimic-cxr-jpg --max-images 60000
    python -m clinicalpulse.imaging.train_cxr evaluate --root /data/mimic-cxr-jpg --ckpt runs/clinicalpulse-cxr/best.pt
    python -m clinicalpulse.imaging.train_cxr predict  --ckpt runs/clinicalpulse-cxr/best.pt --image xray.jpg --gradcam

Design choices (all documented in the code):
- Official train/validate/test split, which is patient-level; a leakage check runs on every index.
  `--max-images` subsets by whole patients, deterministically.
- Frontal views only (PA, AP) by default. Images are not re-compressed or altered.
- Default labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumonia,
  Pneumothorax, Lung Opacity (`--labels all` for all 14).
- Uncertain (-1) labels: `--uncertain ones|zeros|ignore` for training only; they are always
  excluded from validation/test metrics. Labels not mentioned in the report count as negative,
  which is noisy.
- Labels come from an automatic report labeler (CheXpert), not from radiologists reading the
  images, so metrics measure agreement with the labeler.
- Per-label decision thresholds are picked on the validation split and applied unchanged to the
  test split. Reports AUROC, sensitivity, specificity, precision, F1, accuracy, ECE and Brier.
- Metrics are per image, not per patient. Studies with several frontal images count each image.

Do not expect 95% on every label: AUROC differs a lot between findings and is bounded by label
noise. Report what you measure on the test split and do not tune on it.

Older generic trainer for `data/{train,val,test}/<CLASS>/` folders:

    python -m clinicalpulse.imaging.xray_classifier train --data data --arch densenet121 --epochs 15

## Layout
    src/clinicalpulse/reports/   report reading (txt/pdf/OCR), entity + assertion extraction, vitals
    src/clinicalpulse/vitals/    NEWS2 scoring, observation validation, trend analysis
    src/clinicalpulse/imaging/   MIMIC-CXR pipeline, metrics, training, Grad-CAM
    tests/                       pytest suite (reports, vitals, MIMIC indexing, metrics)
    app/                         FastAPI server + static dashboard (app/static/index.html)
    data/, runs/                 local only, git-ignored
