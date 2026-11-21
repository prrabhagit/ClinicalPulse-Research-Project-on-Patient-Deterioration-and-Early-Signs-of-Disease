"""Read an uploaded report (.txt, text-layer or scanned .pdf, or image) into text.

Inputs are validated; failures raise ReportReadError with a reason instead of returning
empty text. Scanned documents need OCR (pytesseract + the tesseract binary).
"""
import io
from pathlib import Path

from .schema import ReportDocument

MAX_BYTES = 20 * 1024 * 1024
TEXT_EXT = {".txt", ".text"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_TEXT_CHARS = 20  # below this a PDF is treated as having no usable text layer


class ReportReadError(Exception):
    pass


def _ocr(image) -> str:
    try:
        import pytesseract
    except ImportError as e:
        raise ReportReadError("OCR needed but pytesseract is not installed (pip install pytesseract).") from e
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as e:
        raise ReportReadError("OCR needed but the tesseract binary is not installed.") from e


def read_report(path) -> ReportDocument:
    p = Path(path)
    if not p.is_file():
        raise ReportReadError(f"File not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise ReportReadError(f"File is empty: {p.name}")
    if size > MAX_BYTES:
        raise ReportReadError(f"File too large ({size / 1e6:.1f} MB; limit {MAX_BYTES // 2**20} MB).")
    ext = p.suffix.lower()
    if ext in TEXT_EXT:
        return _read_text(p)
    if ext == ".pdf":
        return _read_pdf(p)
    if ext in IMAGE_EXT:
        return _read_image(p)
    raise ReportReadError(f"Unsupported file type '{ext}'. Supported: .txt, .pdf, {', '.join(sorted(IMAGE_EXT))}")


def _read_text(p: Path) -> ReportDocument:
    raw = p.read_bytes()
    warnings = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        warnings.append("File was not valid UTF-8; decoded as Windows-1252, some characters may be wrong.")
    return ReportDocument(text, p.name, "text", 1, warnings)


def _read_pdf(p: Path) -> ReportDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(p))
    except PdfReadError as e:
        raise ReportReadError(f"Could not parse PDF: {e}") from e
    if reader.is_encrypted:
        raise ReportReadError("PDF is encrypted; decrypt it before uploading.")
    pages = list(reader.pages)
    text = "\n".join((pg.extract_text() or "") for pg in pages).strip()
    if len(text) >= MIN_TEXT_CHARS:
        return ReportDocument(text, p.name, "pdf_text", len(pages))

    # No text layer: OCR the page images embedded in the PDF (typical for scans).
    from PIL import Image

    chunks = []
    for pg in pages:
        for img in pg.images:
            chunks.append(_ocr(Image.open(io.BytesIO(img.data))))
    text = "\n".join(chunks).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ReportReadError("No readable text found in PDF, even after OCR.")
    return ReportDocument(text, p.name, "ocr", len(pages), ["Text obtained via OCR; verify transcription."])


def _read_image(p: Path) -> ReportDocument:
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(p)
        img.load()
    except UnidentifiedImageError as e:
        raise ReportReadError(f"Not a valid image: {p.name}") from e
    text = _ocr(img).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ReportReadError("No readable text found in image (try a higher-resolution scan).")
    return ReportDocument(text, p.name, "ocr", 1, ["Text obtained via OCR; verify transcription."])
