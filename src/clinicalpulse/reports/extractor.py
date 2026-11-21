"""Rule-based clinical report analysis: sections, entities, assertion status, vitals.

This is a transparent baseline (NegEx-style context handling), not a trained model. It
handles negation, uncertainty, resolved/prior findings and contradictions, but it can be
wrong on long or unusual constructions, so every mention keeps the sentence and the cue
that decided it, to let a clinician audit the result. Validate against annotated reports
before relying on it.
"""
import re
from typing import Dict, List, Tuple

from . import lexicon as lx
from .schema import (CONFLICTING, HISTORICAL, NEGATED, PRESENT, UNCERTAIN, Mention,
                     ReportAnalysis)
from .vitals import extract_vitals

WINDOW = 10  # max words between a cue and the entity it governs
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _blank(pattern: "re.Pattern", text: str) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


def split_sections(text: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return [(section, sentence)] and the list of section names seen."""
    section, seen, out = "BODY", [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = lx.SECTION_HEADER.match(line)
        if m:
            section = lx.SECTION_ALIASES[m.group(1).upper()]
            if section not in seen:
                seen.append(section)
            line = line[m.end():]
        for sent in _SENT_SPLIT.split(line):
            if sent.strip():
                out.append((section, sent.strip()))
    return out, seen


def _assertion(sentence: str, start: int, end: int) -> Tuple[str, str]:
    """Decide the assertion status of the entity at sentence[start:end].

    The nearest cue (within WINDOW words, not crossing a termination word) wins.
    """
    pre, post = sentence[:start], sentence[end:]
    terms = list(lx.TERMINATION.finditer(pre))
    if terms:
        pre = pre[terms[-1].end():]
    m = lx.TERMINATION.search(post)
    if m:
        post = post[: m.start()]
    pre = " ".join(pre.split()[-WINDOW:])
    post = " ".join(post.split()[:WINDOW])

    cands = []  # (distance, order, assertion, cue)
    pre_for_neg = _blank(lx.PSEUDO_NEG, _blank(lx.NEG_POST, pre))
    for order, (pat, src, label) in enumerate(
        ((lx.NEG_PRE, pre_for_neg, NEGATED), (lx.UNC_PRE, pre, UNCERTAIN), (lx.HIST_PRE, pre, HISTORICAL))
    ):
        last = None
        for mm in pat.finditer(src):
            last = mm
        if last:
            cands.append((len(src) - last.end(), order, label, last.group().strip()))
    q = lx.UNC_PRE_Q.search(pre)
    if q:
        cands.append((len(q.group()), 1, UNCERTAIN, "?"))

    post_for_neg = _blank(lx.UNC_POST, post)
    for order, (pat, src, label) in enumerate(
        ((lx.NEG_POST, post_for_neg, NEGATED), (lx.UNC_POST, post, UNCERTAIN), (lx.HIST_POST, post, HISTORICAL))
    ):
        mm = pat.search(src)
        if mm:
            cands.append((mm.start(), order, label, mm.group().strip()))

    if not cands:
        return PRESENT, ""
    _, _, label, cue = min(cands, key=lambda c: (c[0], c[1]))
    return label, cue


def _find_mentions(section: str, sentence: str) -> List[Mention]:
    spans = []
    for name, (cat, rx) in lx.ENTITY_RES.items():
        for m in rx.finditer(sentence):
            spans.append((m.start(), m.end(), name, cat))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))  # longest first
    kept = []
    for s in spans:  # drop overlaps, e.g. "coughing up blood" beats "coughing"
        if not any(s[0] < k[1] and k[0] < s[1] for k in kept):
            kept.append(s)
    mentions = []
    for start, end, name, cat in sorted(kept):
        label, cue = _assertion(sentence, start, end)
        if section == "COMPARISON" and label in (PRESENT, UNCERTAIN):
            label, cue = HISTORICAL, "comparison section"
        mentions.append(Mention(name, cat, label, sentence[start:end], cue or None, sentence, section))
    return mentions


def _aggregate(mentions: List[Mention]) -> Tuple[Dict[str, str], List[str]]:
    by_entity: Dict[str, set] = {}
    for m in mentions:
        by_entity.setdefault(m.entity, set()).add(m.assertion)
    findings, contradictions = {}, []
    for ent, labels in by_entity.items():
        if NEGATED in labels and (PRESENT in labels or UNCERTAIN in labels):
            findings[ent] = CONFLICTING
            other = PRESENT if PRESENT in labels else UNCERTAIN
            contradictions.append(f"{ent}: report both negates it and states it as {other}")
        elif PRESENT in labels:
            findings[ent] = PRESENT
        elif UNCERTAIN in labels:
            findings[ent] = UNCERTAIN
        elif NEGATED in labels:
            findings[ent] = NEGATED
        else:
            findings[ent] = HISTORICAL
    return findings, contradictions


def analyze_report(text: str, source: str = "<text>", method: str = "text") -> ReportAnalysis:
    if not text or len(text.strip()) < 20:
        raise ValueError("Report text is empty or too short to analyze (< 20 characters).")
    sentences, sections = split_sections(text)
    mentions: List[Mention] = []
    for section, sent in sentences:
        mentions.extend(_find_mentions(section, sent))
    findings, contradictions = _aggregate(mentions)
    vitals, vital_warnings = extract_vitals(text)
    statements = [k for k, rx in lx.NORMAL_STATEMENTS.items() if rx.search(text)]

    warnings = list(vital_warnings)
    if not ({"FINDINGS", "IMPRESSION"} & set(sections)):
        warnings.append("No FINDINGS or IMPRESSION section recognized; section context may be missing.")
    if not mentions:
        warnings.append("No supported clinical entities recognized; the report may be outside the vocabulary.")
    return ReportAnalysis(source, method, sections, mentions, findings, vitals, statements,
                          contradictions, warnings)
