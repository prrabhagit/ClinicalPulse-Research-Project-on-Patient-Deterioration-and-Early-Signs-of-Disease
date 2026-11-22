"""Time-series analysis of vital signs: per-observation NEWS2 plus windowed trends.

Trends are least-squares slopes over the last `window_hours` before the final observation. A
trend is FLAGGED only when the fitted change across the observed span moves in the concerning
direction by at least a configurable amount. The default amounts are transparent heuristics
chosen for this project, NOT validated clinical cut-offs; tune and validate them on real data.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .news2 import News2Result, score_news2
from .observations import load_observations

# vital -> (concerning direction, minimum fitted change across the window)
DEFAULT_CONCERN: Dict[str, Tuple[str, float]] = {
    "spo2": ("down", 3.0),
    "heart_rate": ("up", 15.0),
    "respiratory_rate": ("up", 4.0),
    "systolic_bp": ("down", 20.0),
    "temperature_c": ("up", 1.0),
}


@dataclass
class Trend:
    name: str
    n: int
    slope_per_hour: Optional[float]
    fitted_change: Optional[float]
    flag: Optional[str]  # None, or e.g. "spo2 falling"
    note: str = ""


@dataclass
class VitalsAnalysis:
    n_observations: int
    first_time: str
    last_time: str
    latest: News2Result
    news2_series: List[dict]
    news2_change_in_window: Optional[int]
    trends: List[Trend]
    flags: List[str]
    warnings: List[str] = field(default_factory=list)


def _trend(df: pd.DataFrame, name: str, min_points: int, concern) -> Trend:
    s = df[["timestamp", name]].dropna()
    if len(s) < min_points:
        return Trend(name, len(s), None, None, None, f"needs >= {min_points} readings in window")
    hours = (s["timestamp"] - s["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    span = float(hours[-1] - hours[0])
    if span <= 0:
        return Trend(name, len(s), None, None, None, "all readings share one timestamp")
    slope = float(np.polyfit(hours, s[name].to_numpy(dtype=float), 1)[0])
    change = slope * span
    flag = None
    if name in concern:
        direction, amount = concern[name]
        if (direction == "down" and change <= -amount) or (direction == "up" and change >= amount):
            flag = f"{name} {'falling' if direction == 'down' else 'rising'} ({change:+.1f} over {span:.1f} h)"
    return Trend(name, len(s), slope, change, flag)


def analyze(source, window_hours: float = 6.0, min_points: int = 3, spo2_scale: int = 1,
            concern: Mapping[str, Tuple[str, float]] = DEFAULT_CONCERN) -> VitalsAnalysis:
    df, warnings = load_observations(source)
    rows = []
    for _, r in df.iterrows():
        res = score_news2(r.to_dict(), spo2_scale)
        rows.append({"timestamp": r["timestamp"].isoformat(), "news2": res.total,
                     "complete": res.complete, "band": res.band})
    last = df["timestamp"].iloc[-1]
    win = df[df["timestamp"] >= last - pd.Timedelta(hours=window_hours)]
    latest = score_news2(df.iloc[-1].to_dict(), spo2_scale)

    change = None
    if len(win) >= 2 and rows[len(df) - len(win)]["complete"] and rows[-1]["complete"]:
        change = rows[-1]["news2"] - rows[len(df) - len(win)]["news2"]

    trends = [_trend(win, n, min_points, concern) for n in
              ("heart_rate", "respiratory_rate", "spo2", "systolic_bp", "temperature_c")]
    flags = [t.flag for t in trends if t.flag]
    if not latest.complete:
        warnings.append(f"Latest observation incomplete (missing {', '.join(latest.missing)}).")
    return VitalsAnalysis(len(df), df["timestamp"].iloc[0].isoformat(), last.isoformat(), latest,
                          rows, change, trends, flags, warnings)
