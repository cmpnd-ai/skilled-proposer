"""Committee name metric with feedback for the reflection model.

The alias table and the normalize functions are vendored from Derek Willis's
political-fundraising-emails project. See ATTRIBUTION.md.
"""

from __future__ import annotations

import re
import string

import dspy
from rapidfuzz import fuzz

MATCH_THRESHOLD = 0.80

INVALID_COMMITTEES = {
    "", "none", "na", "n/a", "unknown", "not found", "not available",
    "no political committee identified", "cannot determine",
    "unable to determine",
}

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

COMMITTEE_ALIASES = {
    "NRSC": "NATIONAL REPUBLICAN SENATORIAL COMMITTEE",
    "THE NATIONAL REPUBLICAN SENATORIAL COMMITTEE": "NATIONAL REPUBLICAN SENATORIAL COMMITTEE",
    "NRCC": "NATIONAL REPUBLICAN CONGRESSIONAL COMMITTEE",
    "THE NATIONAL REPUBLICAN CONGRESSIONAL COMMITTEE": "NATIONAL REPUBLICAN CONGRESSIONAL COMMITTEE",
    "VPP": "VOTER PROTECTION PROJECT",
    "THE VOTER PROTECTION PROJECT": "VOTER PROTECTION PROJECT",
    "VPP (WWW.PROTECTVOTING.ORG)": "VOTER PROTECTION PROJECT",
    "VPP (www.protectvoting.org)": "VOTER PROTECTION PROJECT",
    "DSCC": "DEMOCRATIC SENATORIAL CAMPAIGN COMMITTEE",
    "THE DEMOCRATIC SENATORIAL CAMPAIGN COMMITTEE": "DEMOCRATIC SENATORIAL CAMPAIGN COMMITTEE",
    "DCCC": "DEMOCRATIC CONGRESSIONAL CAMPAIGN COMMITTEE",
    "THE DEMOCRATIC CONGRESSIONAL CAMPAIGN COMMITTEE": "DEMOCRATIC CONGRESSIONAL CAMPAIGN COMMITTEE",
    "DNC": "DEMOCRATIC NATIONAL COMMITTEE",
    "THE DEMOCRATIC NATIONAL COMMITTEE": "DEMOCRATIC NATIONAL COMMITTEE",
    "THE DEMOCRATS": "DEMOCRATIC NATIONAL COMMITTEE",
    "HMP": "HOUSE MAJORITY PAC",
    "THE HOUSE MAJORITY PAC": "HOUSE MAJORITY PAC",
    "SMP": "SENATE MAJORITY PAC",
    "THE SENATE MAJORITY PAC": "SENATE MAJORITY PAC",
    "AB PAC": "AMERICAN BRIDGE 21ST CENTURY",
    "AMERICAN BRIDGE": "AMERICAN BRIDGE 21ST CENTURY",
    "AMERICAN BRIDGE PAC": "AMERICAN BRIDGE 21ST CENTURY",
    "NHGOP": "NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE",
    "NH GOP": "NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE",
    "NH REPUBLICAN PARTY": "NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE",
    "REPUBLICAN PARTY OF NEW HAMPSHIRE": "NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE",
    "MIGOP": "MICHIGAN REPUBLICAN PARTY",
    "MICHIGAN GOP": "MICHIGAN REPUBLICAN PARTY",
    "TRUMP SAVE AMERICA JFC": "TRUMP SAVE AMERICA JOINT FUNDRAISING COMMITTEE",
    "INDIANA GOP": "INDIANA REPUBLICAN STATE COMMITTEE",
    "DLCC PAC": "DEMOCRATIC LEGISLATIVE CAMPAIGN COMMITTEE",
    "ANDREW GILLUM, DEMOCRAT, FOR GOVERNOR": "ANDREW GILLUM FOR GOVERNOR",
    "THE ARIZONA DEMOCRATIC PARTY": "ARIZONA DEMOCRATIC PARTY",
    "ARIZONA DEMOCRATS": "ARIZONA DEMOCRATIC PARTY",
    "BECCA FOR VERMONT": "BECCA BALINT FOR VERMONT",
    "CAVPAC": "CHAMPION AMERICAN VALUES",
    "WISGOP": "REPUBLICAN PARTY OF WISCONSIN",
    "THE NEBRASKA REPUBLICAN PARTY": "NEBRASKA REPUBLICAN PARTY",
}


def _apply_aliases(value):
    return COMMITTEE_ALIASES.get(value, value)


def normalize_exact(value):
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in INVALID_COMMITTEES:
        return None
    cleaned = stripped.upper().translate(PUNCT_TABLE)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    return _apply_aliases(cleaned)


def normalize_fuzzy(value):
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in INVALID_COMMITTEES:
        return None
    cleaned = " ".join(stripped.upper().split())
    if not cleaned:
        return None
    return _apply_aliases(cleaned)


_TAG_RE = re.compile(r"<[^>]+>")
_LABEL_RE = re.compile(r"^\s*(committee|answer|name|paid for by)\b\s*[:>\-]+", re.I)


def _formatting_artifacts(value: str) -> str:
    """Name the stray format pieces in a value, or return an empty string."""
    notes = []
    if _TAG_RE.search(value):
        notes.append("XML/HTML tags")
    if "[[" in value or "]]" in value or "##" in value:
        notes.append("field markers like '[[ ## ... ## ]]'")
    if len(value) >= 2 and value[0] in "\"'" and value[-1] in "\"'":
        notes.append("wrapping quotes")
    if _LABEL_RE.search(value):
        notes.append("a leading label or preface")
    return ", ".join(notes)


def committee_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    gold_name = (getattr(gold, "committee", "") or "").strip()
    pred_value = (getattr(pred, "committee", "") or "").strip()

    if not pred_value:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"No committee extracted. Expected '{gold_name}'. The model must "
                "return the committee named in the 'Paid for by' disclaimer or sign-off."
            ),
        )

    g_exact = normalize_exact(gold_name)
    p_exact = normalize_exact(pred_value)
    if g_exact is not None and g_exact == p_exact:
        return dspy.Prediction(score=1.0, feedback=f"Correct. Matched '{gold_name}'.")

    g_fuzzy = normalize_fuzzy(gold_name) or ""
    p_fuzzy = normalize_fuzzy(pred_value) or ""
    ratio = (fuzz.ratio(g_fuzzy, p_fuzzy) / 100.0) if (g_fuzzy and p_fuzzy) else 0.0

    if ratio >= MATCH_THRESHOLD:
        verdict = f"Close (fuzzy {ratio:.2f} at or above {MATCH_THRESHOLD}) but not an exact normalized match."
    else:
        verdict = f"Wrong (fuzzy {ratio:.2f} below {MATCH_THRESHOLD})."
    feedback = (
        f"{verdict} Predicted '{pred_value}', expected '{gold_name}'. "
        "Return only the committee name exactly as written in the official disclaimer. "
        "Drop the 'Paid for by' preface, a leading 'The', and any address or URL after the name. "
        "If several committees appear, pick the primary one funding the email."
    )
    artifacts = _formatting_artifacts(pred_value)
    if artifacts:
        feedback += (
            f" The answer also contained {artifacts}. The committee value must be the "
            "bare name only, with no tags, field markers, quotes, or labels around it."
        )
    return dspy.Prediction(score=ratio, feedback=feedback)


# (gold, predicted, expected score or None when only the shape is checked)
SELFTEST_CASES = [
    ("Ted Cruz for Senate", "Ted Cruz for Senate", 1.0),
    ("NRSC", "National Republican Senatorial Committee", 1.0),
    ("Rosen for Nevada", "Jacky Rosen for Nevada", None),
    ("Adam for Colorado", "", 0.0),
    ("Adam for Colorado", "Some PAC", None),
]
