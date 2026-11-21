"""Medical report scanning: file -> text -> structured clinical information."""
from .extractor import analyze_report
from .reader import ReportReadError, read_report
from .schema import ReportAnalysis, ReportDocument


def scan_report(path) -> ReportAnalysis:
    """Read an uploaded report file and return its structured analysis."""
    doc = read_report(path)
    result = analyze_report(doc.text, source=doc.source, method=doc.method)
    result.warnings = doc.warnings + result.warnings
    return result


__all__ = ["scan_report", "analyze_report", "read_report", "ReportReadError", "ReportAnalysis", "ReportDocument"]
