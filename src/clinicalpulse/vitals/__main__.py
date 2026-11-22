"""CLI: python -m clinicalpulse.vitals score observations.csv [--window 6] [--spo2-scale 2] [--json]"""
import argparse
import dataclasses
import json
import sys

from . import ObservationError, analyze


def main() -> int:
    ap = argparse.ArgumentParser(prog="clinicalpulse.vitals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="NEWS2 + trends from a CSV of observations")
    s.add_argument("file")
    s.add_argument("--window", type=float, default=6.0, help="trend window in hours")
    s.add_argument("--spo2-scale", type=int, choices=[1, 2], default=1)
    s.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        res = analyze(a.file, a.window, spo2_scale=a.spo2_scale)
    except (ObservationError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(dataclasses.asdict(res), indent=2, default=str))
        return 0
    L = res.latest
    print(f"{res.n_observations} observations, {res.first_time} -> {res.last_time}")
    print(f"Latest NEWS2: {L.total} ({L.band}){'' if L.complete else ' [INCOMPLETE - lower bound]'}")
    print("  components:", ", ".join(f"{k}={v}" for k, v in L.components.items()))
    if res.news2_change_in_window is not None:
        print(f"  change over last {a.window:g} h: {res.news2_change_in_window:+d}")
    print("Trends:")
    for t in res.trends:
        body = f"{t.fitted_change:+.1f} fitted change ({t.slope_per_hour:+.2f}/h)" if t.slope_per_hour is not None else t.note
        print(f"  {t.name:<17} n={t.n}  {body}{'  <-- ' + t.flag if t.flag else ''}")
    for title, items in (("Notes", L.notes), ("Warnings", res.warnings)):
        if items:
            print(f"{title}:")
            for i in items:
                print(f"  - {i}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
