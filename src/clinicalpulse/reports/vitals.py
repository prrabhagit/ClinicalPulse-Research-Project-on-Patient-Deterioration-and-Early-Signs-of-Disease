"""Extract vital-sign readings written in free text. Implausible values are rejected with a warning."""
import re
from typing import List, Tuple

from ..vitals.observations import PLAUSIBLE
from .schema import VitalReading

I = re.IGNORECASE
_GAP = r"[^\d\n]{0,15}"

_SPO2 = re.compile(
    r"\b(?:spo2|sp02|sao2|o2 sats?|o2 saturation|oxygen saturation|pulse ox(?:imetry)?|sats?)\b"
    + _GAP + r"(\d{2,3})(?:\s*%)?",
    I,
)
_HR = re.compile(r"\b(?:hr|heart rate|pulse(?! ?ox))\b" + _GAP + r"(\d{2,3})\b", I)
_BP = re.compile(r"\b(?:bp|blood pressure)\b" + _GAP + r"(\d{2,3})\s*/\s*(\d{2,3})\b", I)
_RR = re.compile(r"\b(?:rr|resp(?:iratory)? rate|resps?)\b" + _GAP + r"(\d{1,2})\b", I)
_TEMP = re.compile(
    r"\b(?:temp(?:erature)?)\b" + _GAP + r"(\d{2,3}(?:\.\d)?)\s*(?:°|º|degrees?)?\s*([CF])?\b", I
)


def _f_to_c(f: float) -> float:
    return round((f - 32.0) * 5.0 / 9.0, 1)


def extract_vitals(text: str) -> Tuple[List[VitalReading], List[str]]:
    out: List[VitalReading] = []
    warns: List[str] = []

    def add(name, value, unit, raw):
        lo, hi = PLAUSIBLE[name]
        if lo <= value <= hi:
            out.append(VitalReading(name, value, unit, raw))
        else:
            warns.append(f"Rejected implausible {name} value {value} in '{raw}'")

    for m in _SPO2.finditer(text):
        add("spo2", float(m.group(1)), "%", m.group(0).strip())
    for m in _HR.finditer(text):
        add("heart_rate", float(m.group(1)), "bpm", m.group(0).strip())
    for m in _BP.finditer(text):
        sys_, dia = float(m.group(1)), float(m.group(2))
        raw = m.group(0).strip()
        if sys_ > dia:
            add("systolic_bp", sys_, "mmHg", raw)
            add("diastolic_bp", dia, "mmHg", raw)
        else:
            warns.append(f"Rejected blood pressure with systolic <= diastolic in '{raw}'")
    for m in _RR.finditer(text):
        add("respiratory_rate", float(m.group(1)), "breaths/min", m.group(0).strip())
    for m in _TEMP.finditer(text):
        v, unit, raw = float(m.group(1)), (m.group(2) or "").upper(), m.group(0).strip()
        if not unit:  # infer only when the range is unambiguous
            unit = "C" if 30 <= v <= 43 else "F" if 86 <= v <= 109 else ""
        if unit == "F":
            v = _f_to_c(v)
        if unit:
            add("temperature_c", v, "C", raw)
        else:
            warns.append(f"Could not determine temperature unit in '{raw}'")
    return out, warns
