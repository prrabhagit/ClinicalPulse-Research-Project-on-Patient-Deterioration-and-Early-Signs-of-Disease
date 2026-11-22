"""Load and validate a patient's vital-sign observations (one row per measurement time).

Columns (only `timestamp` is mandatory; missing values stay missing, never imputed):
    timestamp, heart_rate, respiratory_rate, spo2, systolic_bp, temperature_c,
    on_oxygen (true/false), consciousness (A/C/V/P/U or alert/confused/voice/pain/unresponsive)
"""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Values outside these ranges are treated as measurement/entry errors, not as real readings.
PLAUSIBLE = {
    "heart_rate": (20, 250), "respiratory_rate": (4, 70), "spo2": (50, 100),
    "systolic_bp": (50, 300), "diastolic_bp": (20, 200), "temperature_c": (30, 43),
}
NUMERIC = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "temperature_c"]
_TRUE = {"true", "1", "yes", "y", "oxygen", "o2", "on oxygen"}
_FALSE = {"false", "0", "no", "n", "air", "ra", "room air"}
_AVPU = {"a": "A", "alert": "A", "c": "C", "confused": "C", "confusion": "C", "new confusion": "C",
         "v": "V", "voice": "V", "p": "P", "pain": "P", "u": "U", "unresponsive": "U"}


class ObservationError(Exception):
    pass


def _parse_map(series: pd.Series, mapping: Dict[str, object], column: str):
    out = []
    for i, v in series.items():
        if pd.isna(v) or str(v).strip() == "":
            out.append(None)
            continue
        key = str(v).strip().lower()
        if key not in mapping:
            raise ObservationError(f"Row {i + 2}: unrecognized {column} value '{v}'.")
        out.append(mapping[key])
    return out


def load_observations(source: Union[str, "pd.DataFrame"]) -> Tuple[pd.DataFrame, List[str]]:
    """Return (clean dataframe sorted by time, warnings). Raises ObservationError on unusable input."""
    try:
        raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ObservationError(f"Could not read observations: {e}") from e
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "timestamp" not in raw.columns:
        raise ObservationError("Observations need a 'timestamp' column.")
    if not (set(NUMERIC) | {"on_oxygen", "consciousness"}) & set(raw.columns):
        raise ObservationError(f"No vital-sign columns found. Expected some of: {NUMERIC}.")
    if raw.empty:
        raise ObservationError("Observation table has no rows.")

    warnings: List[str] = []
    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    if ts.isna().any():
        bad = [int(i) + 2 for i in raw.index[ts.isna()]][:5]
        raise ObservationError(f"Unparseable timestamps in rows {bad} (first 5 shown).")
    df = pd.DataFrame({"timestamp": ts})
    for col in NUMERIC:
        if col not in raw.columns:
            df[col] = np.nan
            continue
        num = pd.to_numeric(raw[col], errors="coerce")
        n_bad = int((num.isna() & raw[col].notna() & (raw[col].astype(str).str.strip() != "")).sum())
        if n_bad:
            warnings.append(f"{col}: {n_bad} non-numeric value(s) treated as missing.")
        lo, hi = PLAUSIBLE[col]
        out = (num < lo) | (num > hi)
        if out.any():
            warnings.append(f"{col}: {int(out.sum())} implausible value(s) outside [{lo}, {hi}] treated as missing.")
            num = num.mask(out)
        df[col] = num
    df["on_oxygen"] = _parse_map(raw["on_oxygen"], {**{k: True for k in _TRUE}, **{k: False for k in _FALSE}},
                                 "on_oxygen") if "on_oxygen" in raw.columns else None
    df["consciousness"] = _parse_map(raw["consciousness"], _AVPU, "consciousness") \
        if "consciousness" in raw.columns else None
    if df["timestamp"].duplicated().any():
        warnings.append("Duplicate timestamps present; rows were kept, check for double entry.")
    if not df["timestamp"].is_monotonic_increasing:
        warnings.append("Rows were not in time order and have been sorted.")
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True), warnings


def from_report_vitals(readings: Iterable) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Collapse vital readings extracted from a report (objects with .name/.value) into one observation.

    Reports carry no per-reading timestamps, so when a vital is mentioned with different values the
    LAST mention is used and a warning is returned. Oxygen use and consciousness are not extracted,
    so they stay missing.
    """
    by_name: Dict[str, List[float]] = {}
    for r in readings:
        by_name.setdefault(r.name, []).append(float(r.value))
    obs: Dict[str, Optional[float]] = {}
    warns = []
    for name in NUMERIC:
        vals = by_name.get(name)
        obs[name] = vals[-1] if vals else None
        if vals and len(set(vals)) > 1:
            warns.append(f"{name} appears with several values {vals} in the report; using the last ({vals[-1]}).")
    return obs, warns
