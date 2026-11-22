"""NEWS2 (Royal College of Physicians, 2017) aggregate early-warning score.

This is an established, rule-based clinical score, not a machine-learning model. Any learned
deterioration model should be compared against it. Thresholds follow the RCP NEWS2 chart:
SpO2 Scale 1 by default; Scale 2 (target 88-92%) only for patients a clinician has confirmed
as hypercapnic. Values are rounded to the chart's resolution (integers; temperature 0.1 C).
Missing parameters are never imputed: the result is flagged incomplete and its score and band are
lower bounds.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

AVPU_SCORE = {"A": 0, "C": 3, "V": 3, "P": 3, "U": 3}


def _band(value, cuts):
    """cuts: [(upper_inclusive, score), ...] evaluated in order; the last entry's bound is ignored."""
    for upper, score in cuts[:-1]:
        if value <= upper:
            return score
    return cuts[-1][1]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


@dataclass
class News2Result:
    components: Dict[str, int]
    total: int
    missing: List[str]
    complete: bool
    any_single_3: bool
    band: str  # low | low-medium | medium | high  (lower bound if not complete)
    spo2_scale: int
    notes: List[str] = field(default_factory=list)


def _spo2_score(spo2: float, on_oxygen: Optional[bool], scale: int) -> Optional[int]:
    s = round(spo2)
    if scale == 1:
        return _band(s, [(91, 3), (93, 2), (95, 1), (None, 0)])
    if on_oxygen is None:
        return None
    if s <= 83: return 3
    if s <= 85: return 2
    if s <= 87: return 1
    if s <= 92: return 0
    if not on_oxygen: return 0  # >= 93 on air
    return {93: 1, 94: 1, 95: 2, 96: 2}.get(s, 3)  # on oxygen: 93-94 ->1, 95-96 ->2, >=97 ->3


def score_news2(obs: Mapping, spo2_scale: int = 1) -> News2Result:
    if spo2_scale not in (1, 2):
        raise ValueError("spo2_scale must be 1 or 2")
    g = lambda k: obs.get(k)  # noqa: E731
    comp: Dict[str, int] = {}
    missing: List[str] = []
    notes: List[str] = []

    def put(name, value, fn):
        if _missing(value):
            missing.append(name)
        else:
            comp[name] = fn(value)

    put("respiratory_rate", g("respiratory_rate"), lambda v: _band(round(v), [(8, 3), (11, 1), (20, 0), (24, 2), (None, 3)]))
    if _missing(g("spo2")):
        missing.append("spo2")
    else:
        sc = _spo2_score(float(g("spo2")), g("on_oxygen") if not _missing(g("on_oxygen")) else None, spo2_scale)
        if sc is None:
            missing.append("spo2 (scale 2 needs on_oxygen)")
        else:
            comp["spo2"] = sc
    put("on_oxygen", g("on_oxygen"), lambda v: 2 if v else 0)
    put("systolic_bp", g("systolic_bp"), lambda v: _band(round(v), [(90, 3), (100, 2), (110, 1), (219, 0), (None, 3)]))
    put("heart_rate", g("heart_rate"), lambda v: _band(round(v), [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (None, 3)]))
    put("consciousness", g("consciousness"), lambda v: AVPU_SCORE[v])
    put("temperature_c", g("temperature_c"), lambda v: _band(round(v, 1), [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (None, 2)]))

    total = sum(comp.values())
    single3 = any(v == 3 for v in comp.values())
    band = "high" if total >= 7 else "medium" if total >= 5 else "low-medium" if single3 else "low"
    complete = not missing
    if not complete:
        notes.append(f"Incomplete: missing {', '.join(missing)}. Score and band are lower bounds, not a clearance.")
    if spo2_scale == 2:
        notes.append("SpO2 Scale 2 used; it is only valid for clinician-confirmed hypercapnic respiratory failure.")
    return News2Result(comp, total, missing, complete, single3, band, spo2_scale, notes)
