"""Clinical vocabulary and assertion cues.

Finding names that exist in NIH ChestX-ray14 use that dataset's label spelling so report
features and image-model outputs can be joined later. "Opacity" and the symptoms are extra.
"""
import re

I = re.IGNORECASE

# canonical name -> (category, regex)
ENTITIES = {
    "Atelectasis": ("finding", r"\batelecta(?:sis|tic)\b"),
    "Cardiomegaly": (
        "finding",
        r"\bcardiomegaly\b|\benlarged (?:heart|cardiac silhouette)\b|\bcardiac enlargement\b"
        r"|\bheart (?:size )?(?:is |appears )?enlarged\b",
    ),
    "Effusion": ("finding", r"(?<!pericardial )\b(?:pleural\s+)?effusions?\b"),
    "Infiltration": ("finding", r"\binfiltrat(?:e|es|ion|ions|ive)\b"),
    "Mass": ("finding", r"\bmass(?:es)?\b(?!\s+effect)"),
    "Nodule": ("finding", r"\bnodul(?:e|es|ar)\b|\bnodularity\b"),
    "Pneumonia": ("finding", r"\bpneumonia\b|\bpneumonic\b"),
    "Pneumothorax": ("finding", r"\bpneumothora(?:x|ces)\b"),
    "Consolidation": ("finding", r"\bconsolidat(?:ion|ions|ive|ed)\b"),
    "Edema": (
        "finding",
        r"\b(?:pulmonary |interstitial )?(?:edema|oedema)\b|\b(?:pulmonary|vascular) congestion\b",
    ),
    "Emphysema": ("finding", r"\bemphysema(?:tous)?\b|\bhyperinflat(?:ion|ed)\b"),
    "Fibrosis": ("finding", r"\b(?:pulmonary )?fibro(?:sis|tic)\b|\bscarring\b"),
    "Pleural_Thickening": ("finding", r"\bpleural (?:thickening|thickened)\b"),
    "Hernia": ("finding", r"\b(?:hiatal )?hernia\b"),
    "Opacity": ("finding", r"\b(?:airspace |lung )?opacit(?:y|ies)\b|\bopacification\b"),
    "Cough": ("symptom", r"\bcough(?:s|ing)?\b"),
    "Dyspnea": ("symptom", r"\bdyspn(?:ea|oea)\b|\bshortness of breath\b|\bSOB\b|\bbreathless(?:ness)?\b"),
    "Fever": ("symptom", r"\bfever(?:s|ish)?\b|\bfebrile\b|\bpyrexi(?:a|al)\b"),
    "Chest_pain": ("symptom", r"\bchest pain\b|\bpleuritic (?:chest )?pain\b"),
    "Hemoptysis": ("symptom", r"\bhemoptysis\b|\bcough(?:ing)? up blood\b"),
    "Wheeze": ("symptom", r"\bwheez(?:e|es|ing)\b"),
}
ENTITY_RES = {name: (cat, re.compile(rx, I)) for name, (cat, rx) in ENTITIES.items()}

# Boundaries that also treat "/" as part of a token so "r/o" and "s/p" match as units.
_L, _R = r"(?<![\w/])", r"(?![\w/])"


def _cue(alts: str) -> "re.Pattern":
    return re.compile(_L + "(?:" + alts + ")" + _R, I)


# --- assertion cues (NegEx-style) -------------------------------------------------
NEG_PRE = _cue(
    r"no|without|negative for|absence of|free of|denies|denied|denying|not|never|nor"
    r"|fails? to (?:show|demonstrate|reveal)"
)
NEG_POST = _cue(
    r"(?:is |are |was |were )?not (?:seen|identified|present|demonstrated|visualized|evident|appreciated|noted|detected)"
    r"|(?:is |are )?absent|(?:has been |was |is )?ruled out|(?:has been |was |is )?excluded"
)
# "no change in effusion" does NOT negate the effusion; "no longer seen" is resolution.
PSEUDO_NEG = _cue(
    r"no (?:significant |interval |appreciable |substantial |definite )?"
    r"(?:change|increase|decrease|worsening|progression|further)"
    r"|no (?:new |significant )?interval|no longer|not (?:only|necessarily)"
)
UNC_PRE = _cue(
    r"possible|possibly|probable|probably|likely|suspected|suspect|suspicious (?:for|of)"
    r"|concern(?:ing)? for|cannot exclude|can not exclude|cannot rule out|can not rule out"
    r"|may represent|may reflect|may be|might be|could represent|could be|questionable|equivocal"
    r"|versus|vs\.?|differential (?:diagnosis )?includes?|suggestive of|suggest(?:s|ing)?"
    r"|compatible with|rule out|r/o|evaluate for|assess for|question of"
)
UNC_PRE_Q = re.compile(r"\?\s*(?:\w+\s+){0,2}$")
UNC_POST = _cue(
    r"cannot be (?:entirely |completely |totally )?(?:excluded|ruled out)|can not be (?:excluded|ruled out)"
    r"|(?:is )?not excluded|may be present|is possible|is suspected|is unlikely|unlikely|versus|vs\.?"
    r"|(?:is )?questionable|is likely|is probable|is suspicious|suspected|possible"
)
HIST_PRE = _cue(
    r"history of|hx of|h/o|prior|previous|previously|status post|s/p|remote|resolution of|resolved"
)
HIST_POST = _cue(
    r"(?:has |have |had |is |are |was |were )?resolved"
    r"|(?:is |are |was |were )?no longer (?:seen|present|visible|identified|evident)"
)
# Ends the scope of a cue.
TERMINATION = re.compile(
    _L + r"(?:but|however|although|though|except|apart from|aside from|other than|while|whereas|which)" + _R
    + r"|[;:]",
    I,
)

SECTION_HEADER = re.compile(
    r"^\s*(FINDINGS?|IMPRESSIONS?|CONCLUSIONS?|INDICATIONS?|CLINICAL (?:HISTORY|INFORMATION)|HISTORY"
    r"|REASON FOR EXAM(?:INATION)?|COMPARISONS?|TECHNIQUE|EXAM(?:INATION)?|ASSESSMENT|PLAN)\s*:\s*",
    I,
)
SECTION_ALIASES = {
    "FINDING": "FINDINGS", "FINDINGS": "FINDINGS",
    "IMPRESSION": "IMPRESSION", "IMPRESSIONS": "IMPRESSION", "CONCLUSION": "IMPRESSION", "CONCLUSIONS": "IMPRESSION",
    "INDICATION": "HISTORY", "INDICATIONS": "HISTORY", "HISTORY": "HISTORY",
    "CLINICAL HISTORY": "HISTORY", "CLINICAL INFORMATION": "HISTORY",
    "REASON FOR EXAM": "HISTORY", "REASON FOR EXAMINATION": "HISTORY",
    "COMPARISON": "COMPARISON", "COMPARISONS": "COMPARISON",
    "TECHNIQUE": "TECHNIQUE", "EXAM": "TECHNIQUE", "EXAMINATION": "TECHNIQUE",
    "ASSESSMENT": "IMPRESSION", "PLAN": "PLAN",
}

NORMAL_STATEMENTS = {
    "normal_chest_statement": re.compile(
        r"\bno acute (?:cardiopulmonary|cardiac or pulmonary|cardiothoracic)\b"
        r"|\bnormal chest\b|\blungs? (?:are|is) clear\b|\bclear lungs\b",
        I,
    ),
}
