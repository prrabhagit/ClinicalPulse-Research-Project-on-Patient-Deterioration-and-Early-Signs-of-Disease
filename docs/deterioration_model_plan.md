# Plan: learned deterioration model (not built)

Nothing here exists as code yet. It records the decisions needed before any "risk" number may be shown.

**Why it is blocked.** It needs patients with image + report + vitals linked and a defined outcome. MIMIC-CXR-JPG
and ChestX-ray14 have images/labels only; vitals and outcomes live in MIMIC-IV (credentialed, same PhysioNet
access route). Datasets without a valid patient-level linkage must not be merged.

**Target (to confirm with you).** Example: ICU transfer or in-hospital death within 24 h of an index time `t`
(binary). Detecting an abnormality, classifying a disease and predicting a future event are separate tasks, so
the X-ray model's AUROC says nothing about this target.

**Data.** Index times: every 4 h of a ward stay (MIMIC-IV `chartevents`/`icustays`/`transfers`), features only
from before `t` (vitals windows and trends, latest labs, report features from `clinicalpulse.reports`, image-model
outputs from the closest prior CXR from MIMIC-CXR via `subject_id`/`study_id` -> `hadm_id` by time window).
Missing modalities stay missing (mask), not imputed.

**Splits.** By patient (`subject_id`), never by stay or time point; keep the MIMIC-CXR official split where
images are used. Tune on validation only; report once on test.

**Baselines to beat.** NEWS2 at `t` (already implemented), then logistic regression on vitals + trends.
Only then a fusion model. Report AUROC, AUPRC, sensitivity/specificity at a fixed alert rate, and calibration
(ECE/reliability), with confidence intervals by patient bootstrap.

**Output rule.** The dashboard shows a probability only if the model beats NEWS2 on held-out patients with
acceptable calibration; otherwise it keeps showing the evidence summary.
