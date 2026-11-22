"""CLI: python -m clinicalpulse.reports scan <file> [--json]"""
import argparse
import json
import sys

from ..vitals import from_report_vitals, score_news2
from . import ReportReadError, scan_report


def main() -> int:
    ap = argparse.ArgumentParser(prog="clinicalpulse.reports")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="analyze a report file")
    s.add_argument("file")
    s.add_argument("--json", action="store_true", help="print full JSON instead of a summary")
    args = ap.parse_args()
    try:
        res = scan_report(args.file)
    except (ReportReadError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0
    print(f"Source: {res.source}  (read via {res.method})")
    print(f"Sections: {', '.join(res.sections_found) or 'none'}")
    print("\nFindings / symptoms:")
    for ent, state in res.findings.items():
        cues = sorted({m.trigger for m in res.mentions if m.entity == ent and m.trigger})
        print(f"  {ent:<20} {state:<12} {('cue: ' + ', '.join(cues)) if cues else ''}")
    if res.vitals:
        print("\nVitals:")
        for v in res.vitals:
            print(f"  {v.name:<18} {v.value:g} {v.unit}")
        obs, vwarn = from_report_vitals(res.vitals)
        n2 = score_news2(obs)
        print(f"  NEWS2 from report vitals: {n2.total} ({n2.band})" + ("" if n2.complete else " [INCOMPLETE - lower bound]"))
        res.warnings.extend(vwarn + n2.notes)
    for title, items in (("Contradictions", res.contradictions), ("Warnings", res.warnings)):
        if items:
            print(f"\n{title}:")
            for i in items:
                print(f"  - {i}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
