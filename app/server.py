"""ClinicalPulse API + dashboard.  Run from the project root:

    pip install fastapi uvicorn python-multipart
    CLINICALPULSE_CKPT=runs/clinicalpulse-cxr/best.pt uvicorn app.server:app --port 8000

Every endpoint calls the real module; nothing is mocked. If no X-ray checkpoint is configured,
/api/xray answers 503 instead of inventing a prediction. Research use only.
"""
import dataclasses
import io
import os
import tempfile
from pathlib import Path

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from clinicalpulse.fusion import summarize_case
from clinicalpulse.reports import ReportReadError, scan_report
from clinicalpulse.vitals import ObservationError, analyze, from_report_vitals, score_news2

MAX_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="ClinicalPulse", version="0.1.0")
_xray = {}  # lazily loaded: (model, labels, thresholds, device)


async def _read(upload: UploadFile, allowed: set) -> bytes:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}")
    data = await upload.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (limit 20 MB).")
    return data


def _xray_model():
    ckpt = os.environ.get("CLINICALPULSE_CKPT")
    if not ckpt or not Path(ckpt).is_file():
        raise HTTPException(503, "No X-ray model configured: train one and set CLINICALPULSE_CKPT to its best.pt.")
    if not _xray:
        from clinicalpulse.imaging.train_cxr import load_for_inference  # imports torch only when needed
        model, labels, _, thr, device = load_for_inference(ckpt)
        _xray.update(model=model, labels=labels, thr=thr, device=device)
    return _xray


@app.get("/api/health")
def health():
    ckpt = os.environ.get("CLINICALPULSE_CKPT")
    return {"report_scanning": True, "vitals": True, "xray_model": bool(ckpt and Path(ckpt).is_file())}


@app.post("/api/report")
async def report(file: UploadFile = File(...)):
    data = await _read(file, {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ("report" + Path(file.filename).suffix.lower())
        path.write_bytes(data)
        try:
            res = scan_report(path)
        except (ReportReadError, ValueError) as e:
            raise HTTPException(422, str(e))
    out = res.to_dict()
    out["source"] = file.filename
    if res.vitals:
        obs, vw = from_report_vitals(res.vitals)
        n2 = score_news2(obs)
        out["news2_from_report"] = dataclasses.asdict(n2)
        out["warnings"] += vw
    return out


@app.post("/api/vitals")
async def vitals(file: UploadFile = File(...), window_hours: float = 6.0, spo2_scale: int = 1):
    data = await _read(file, {".csv"})
    try:
        return dataclasses.asdict(analyze(io.BytesIO(data), window_hours, spo2_scale=spo2_scale))
    except (ObservationError, ValueError) as e:
        raise HTTPException(422, str(e))


@app.post("/api/xray")
async def xray(file: UploadFile = File(...)):
    data = await _read(file, {".png", ".jpg", ".jpeg"})
    m = _xray_model()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "Not a readable image.")
    from clinicalpulse.imaging.train_cxr import predict_image
    return {"predictions": predict_image(m["model"], m["labels"], m["thr"], m["device"], img),
            "note": "Research estimate from a model trained on a small public dataset; not a diagnosis."}


@app.post("/api/case")
def case(payload: Dict[str, Any] = Body(...)):
    """Aggregate evidence from the outputs of /api/vitals, /api/report and /api/xray (any may be null)."""
    try:
        return summarize_case(payload.get("vitals"), payload.get("report"), payload.get("xray"))
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"Malformed case payload: {e!r}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
