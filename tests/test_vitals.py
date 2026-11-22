import numpy as np
import pandas as pd
import pytest

from clinicalpulse.reports import analyze_report
from clinicalpulse.vitals import (ObservationError, analyze, from_report_vitals, load_observations,
                                  score_news2)

NORMAL = dict(respiratory_rate=16, spo2=98, on_oxygen=False, systolic_bp=120, heart_rate=70,
              consciousness="A", temperature_c=37.0)


def comp(**over):
    return score_news2({**NORMAL, **over}).components


def test_normal_patient_scores_zero_low():
    r = score_news2(NORMAL)
    assert r.total == 0 and r.band == "low" and r.complete


def test_published_worked_example_is_13():  # RCP NEWS2 worked example used in trial protocols
    r = score_news2(dict(respiratory_rate=26, spo2=95, on_oxygen=True, systolic_bp=95, heart_rate=109,
                         consciousness="C", temperature_c=39.0))
    assert r.total == 13 and r.band == "high"


@pytest.mark.parametrize("name,value,expected", [
    ("respiratory_rate", 8, 3), ("respiratory_rate", 9, 1), ("respiratory_rate", 11, 1),
    ("respiratory_rate", 12, 0), ("respiratory_rate", 20, 0), ("respiratory_rate", 21, 2),
    ("respiratory_rate", 24, 2), ("respiratory_rate", 25, 3),
    ("spo2", 91, 3), ("spo2", 92, 2), ("spo2", 93, 2), ("spo2", 94, 1), ("spo2", 95, 1), ("spo2", 96, 0),
    ("systolic_bp", 90, 3), ("systolic_bp", 91, 2), ("systolic_bp", 100, 2), ("systolic_bp", 101, 1),
    ("systolic_bp", 110, 1), ("systolic_bp", 111, 0), ("systolic_bp", 219, 0), ("systolic_bp", 220, 3),
    ("heart_rate", 40, 3), ("heart_rate", 41, 1), ("heart_rate", 50, 1), ("heart_rate", 51, 0),
    ("heart_rate", 90, 0), ("heart_rate", 91, 1), ("heart_rate", 110, 1), ("heart_rate", 111, 2),
    ("heart_rate", 130, 2), ("heart_rate", 131, 3),
    ("temperature_c", 35.0, 3), ("temperature_c", 35.1, 1), ("temperature_c", 36.0, 1),
    ("temperature_c", 36.1, 0), ("temperature_c", 38.0, 0), ("temperature_c", 38.1, 1),
    ("temperature_c", 39.0, 1), ("temperature_c", 39.1, 2),
])
def test_component_boundaries(name, value, expected):
    assert comp(**{name: value})[name] == expected


def test_oxygen_and_consciousness():
    assert comp(on_oxygen=True)["on_oxygen"] == 2
    assert comp(consciousness="A")["consciousness"] == 0
    for c in "CVPU":
        assert comp(consciousness=c)["consciousness"] == 3


@pytest.mark.parametrize("spo2,oxy,expected", [
    (83, False, 3), (84, False, 2), (86, False, 1), (88, False, 0), (92, False, 0), (95, False, 0),
    (93, True, 1), (94, True, 1), (95, True, 2), (96, True, 2), (97, True, 3), (90, True, 0),
])
def test_spo2_scale_2(spo2, oxy, expected):
    r = score_news2({**NORMAL, "spo2": spo2, "on_oxygen": oxy}, spo2_scale=2)
    assert r.components["spo2"] == expected


def test_scale_2_without_oxygen_status_is_missing_not_guessed():
    r = score_news2({**NORMAL, "on_oxygen": None}, spo2_scale=2)
    assert "spo2" not in r.components and not r.complete


def test_bands():
    assert score_news2({**NORMAL, "heart_rate": 135}).band == "low-medium"  # single 3
    assert score_news2({**NORMAL, "heart_rate": 120, "respiratory_rate": 22, "spo2": 94}).band == "medium"  # 2+2+1=5
    assert score_news2({**NORMAL, "heart_rate": 135, "respiratory_rate": 26, "spo2": 90}).band == "high"


def test_missing_parameters_flagged_and_never_imputed():
    r = score_news2({"heart_rate": 135, "spo2": 90})
    assert not r.complete and r.total == 6 and "respiratory_rate" in r.missing
    assert any("lower bound" in n for n in r.notes)
    assert score_news2({"heart_rate": np.nan}).missing.count("heart_rate") == 1


def test_bad_scale_rejected():
    with pytest.raises(ValueError):
        score_news2(NORMAL, spo2_scale=3)


# ------------------------------ observations ------------------------------
def df_of(rows):
    return pd.DataFrame(rows)


def test_load_cleans_and_warns():
    df, w = load_observations(df_of([
        {"timestamp": "2026-01-01 08:00", "spo2": 95, "heart_rate": 900, "on_oxygen": "yes", "consciousness": "alert"},
        {"timestamp": "2026-01-01 06:00", "spo2": "abc", "heart_rate": 80, "on_oxygen": "RA", "consciousness": "C"},
    ]))
    assert df["timestamp"].is_monotonic_increasing and any("sorted" in x for x in w)
    assert any("implausible" in x for x in w) and any("non-numeric" in x for x in w)
    assert np.isnan(df.loc[1, "heart_rate"]) and bool(df.loc[1, "on_oxygen"]) is True
    assert df.loc[0, "consciousness"] == "C"


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"spo2": [95]}),                                         # no timestamp
    pd.DataFrame({"timestamp": ["2026-01-01"], "notes": ["x"]}),          # no vitals
    pd.DataFrame({"timestamp": ["not a time"], "spo2": [95]}),            # bad timestamp
    pd.DataFrame({"timestamp": ["2026-01-01"], "consciousness": ["zzz"]}),  # bad AVPU
    pd.DataFrame({"timestamp": [], "spo2": []}),                          # empty
])
def test_load_rejects_unusable_input(bad):
    with pytest.raises(ObservationError):
        load_observations(bad)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ObservationError):
        load_observations(str(tmp_path / "nope.csv"))


# ---------------------------------- trends ----------------------------------
def series(spo2, hr, start="2026-01-01 08:00", step_h=1.0):
    t0 = pd.Timestamp(start)
    return df_of([{"timestamp": t0 + pd.Timedelta(hours=i * step_h), "spo2": s, "heart_rate": h,
                   "respiratory_rate": 16, "systolic_bp": 120, "temperature_c": 37.0,
                   "on_oxygen": False, "consciousness": "A"} for i, (s, h) in enumerate(zip(spo2, hr))])


def test_deteriorating_patient_is_flagged():
    res = analyze(series([97, 96, 94, 93, 91, 90], [72, 78, 86, 95, 104, 112]))
    names = " ".join(res.flags)
    assert "spo2 falling" in names and "heart_rate rising" in names
    assert res.news2_change_in_window is not None and res.news2_change_in_window > 0
    assert res.latest.total > res.news2_series[0]["news2"]


def test_stable_patient_not_flagged():
    res = analyze(series([97, 97, 96, 97, 97, 97], [72, 74, 71, 73, 72, 72]))
    assert res.flags == [] and res.news2_change_in_window == 0


def test_too_few_points_reports_insufficient_data_not_a_trend():
    res = analyze(series([97, 90], [70, 120]))
    t = {x.name: x for x in res.trends}
    assert t["spo2"].slope_per_hour is None and "needs" in t["spo2"].note and res.flags == []


def test_window_limits_trend_to_recent_data():
    # old decline outside the 2h window must not be flagged
    res = analyze(series([99, 95, 91, 97, 97, 97], [70] * 6), window_hours=2.0)
    assert not any("spo2" in f for f in res.flags)


def test_incomplete_latest_observation_warns():
    df = series([97, 97, 97], [70, 70, 70]).drop(columns=["consciousness"])
    res = analyze(df)
    assert not res.latest.complete and any("incomplete" in w.lower() for w in res.warnings)


# ------------------------------ report -> NEWS2 ------------------------------
def test_report_vitals_feed_partial_news2():
    rep = analyze_report("Chest radiograph report. HR 112, BP 128/76, SpO2 91% on RA, Temp 38.6 C, RR 24.")
    obs, warns = from_report_vitals(rep.vitals)
    r = score_news2(obs)
    assert r.components["heart_rate"] == 2 and r.components["spo2"] == 3 and r.components["respiratory_rate"] == 2
    assert not r.complete and {"on_oxygen", "consciousness"} <= set(r.missing)


def test_conflicting_report_vitals_warn_and_use_last():
    rep = analyze_report("Chest radiograph report. HR 90 on arrival. Later HR 120.")
    obs, warns = from_report_vitals(rep.vitals)
    assert obs["heart_rate"] == 120 and warns
