"""Data structures returned by the report-analysis pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Assertion status of a clinical entity in a given sentence.
PRESENT = "present"
NEGATED = "negated"
UNCERTAIN = "uncertain"
HISTORICAL = "historical"  # resolved / prior / past history, not a current finding
CONFLICTING = "conflicting"  # aggregate only: report both asserts and negates the entity


@dataclass
class ReportDocument:
    """Text obtained from an uploaded report, plus how it was obtained."""

    text: str
    source: str
    method: str  # "text" | "pdf_text" | "ocr"
    pages: int = 1
    warnings: List[str] = field(default_factory=list)


@dataclass
class Mention:
    entity: str  # canonical name, e.g. "Pleural effusion" key from lexicon
    category: str  # "finding" | "symptom"
    assertion: str  # PRESENT / NEGATED / UNCERTAIN / HISTORICAL
    text: str  # matched surface form
    trigger: Optional[str]  # cue that decided the assertion (None => default present)
    sentence: str
    section: str


@dataclass
class VitalReading:
    name: str  # heart_rate, spo2, systolic_bp, diastolic_bp, temperature_c, respiratory_rate
    value: float
    unit: str
    text: str  # raw matched text


@dataclass
class ReportAnalysis:
    source: str
    method: str
    sections_found: List[str]
    mentions: List[Mention]
    findings: Dict[str, str]  # entity -> aggregate assertion (incl. CONFLICTING)
    vitals: List[VitalReading]
    statements: List[str]  # e.g. "normal_chest_statement"
    contradictions: List[str]
    warnings: List[str]

    def features(self) -> Dict[str, float]:
        """CheXpert-style numeric encoding of *current* findings.

        1.0 = present, 0.0 = negated, -1.0 = uncertain/conflicting.
        Entities that are not mentioned, or only mentioned as historical, are omitted
        (missing is not the same as negative).
        """
        enc = {PRESENT: 1.0, NEGATED: 0.0, UNCERTAIN: -1.0, CONFLICTING: -1.0}
        return {f"report_{k}": enc[v] for k, v in self.findings.items() if v in enc}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = self.features()
        return d
