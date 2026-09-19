"""Checkers for the 24 IFEval constraint types used by RLVR-IFeval.

Each checker takes the response text and the constraint arguments and
returns whether the constraint holds plus a short measurement the metric
puts in its feedback. The pass or fail logic follows the AllenAI
open-instruct verifiers so scores match the reference implementation.
"""

from __future__ import annotations

import json
import re
from typing import Callable

Result = tuple[bool, str]

_SENTENCE_SPLIT = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")


def _quantified(actual: int, N: int, quantifier: str, tolerance: int) -> bool:
    if quantifier == "at least":
        return actual >= N
    if quantifier == "at most":
        return actual <= N
    if quantifier == "around":
        return abs(actual - N) <= tolerance
    return False


def verify_keywords(text: str, keyword_list: list[str]) -> Result:
    lower = text.lower()
    missing = [k for k in keyword_list if k.lower() not in lower]
    return not missing, f"missing keywords: {missing}" if missing else "all keywords present"


def verify_keyword_frequency(text: str, word: str, N: int) -> Result:
    words = re.findall(r"\b\w+\b", text.lower())
    actual = sum(1 for w in words if w == word.lower())
    return actual == N, f"'{word}' appears {actual} times, required exactly {N}"


def validate_forbidden_words(text: str, forbidden_words: list[str]) -> Result:
    lower = text.lower()
    found = [w for w in forbidden_words if w.lower() in lower]
    return not found, f"forbidden words present: {found}" if found else "no forbidden words"


def verify_letter_frequency(text: str, letter: str, N: int) -> Result:
    actual = text.count(letter)
    return actual == N, f"letter '{letter}' appears {actual} times, required exactly {N}"


def verify_paragraph_count(text: str, N: int) -> Result:
    cleaned = "\n".join(line.strip() for line in text.splitlines()).strip()
    parts = cleaned.split("* * *")
    nonempty = [p for p in parts if p.strip()]
    if len(nonempty) != len(parts):
        return False, f"{len(parts)} parts split by '* * *' but some are empty"
    return len(parts) == N, f"{len(parts)} paragraphs split by '* * *', required {N}"


def validate_word_constraint(text: str, N: int, quantifier: str) -> Result:
    actual = len(text.strip().split())
    ok = _quantified(actual, N, quantifier, max(round(N * 0.1), 1))
    return ok, f"{actual} words, required {quantifier} {N}"


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> Result:
    actual = len(_SENTENCE_SPLIT.split(text))
    return _quantified(actual, N, quantifier, 1), f"{actual} sentences, required {quantifier} {N}"


def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> Result:
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False, f"{len(paragraphs)} paragraphs split by blank lines, required {N}"
    starts = paragraphs[i - 1].strip().startswith(first_word)
    return starts, f"{N} paragraphs; paragraph {i} {'starts' if starts else 'does not start'} with '{first_word}'"


def verify_postscript(text: str, postscript_marker: str) -> Result:
    if postscript_marker not in text:
        return False, f"no postscript starting with '{postscript_marker}'"
    remaining = text[text.find(postscript_marker):].strip()
    ok = len(remaining) > len(postscript_marker)
    return ok, "postscript present" if ok else f"'{postscript_marker}' present but nothing follows it"


def validate_placeholders(text: str, N: int) -> Result:
    actual = len(re.findall(r"\[(.*?)\]", text))
    return actual >= N, f"{actual} square bracket placeholders, required at least {N}"


def verify_bullet_points(text: str, N: int) -> Result:
    actual = sum(1 for line in text.split("\n") if line.strip().startswith(("*", "-")))
    return actual == N, f"{actual} markdown bullet lines, required exactly {N}"


def validate_title(text: str) -> Result:
    ok = bool(re.findall(r"<<(.*?)>>", text))
    return ok, "title in << >> present" if ok else "no title wrapped in << >>"


def validate_choice(text: str, options: list[str]) -> Result:
    ok = any(option in text for option in options)
    return ok, f"response {'contains' if ok else 'contains none of'} the options {options}"


def validate_highlighted_sections(text: str, N: int) -> Result:
    actual = len(re.findall(r"\*(.*?)\*", text))
    return actual >= N, f"{actual} sections highlighted with *asterisks*, required at least {N}"


def validate_sections(text: str, N: int, section_splitter: str) -> Result:
    sections = text.split(section_splitter)
    if sections and sections[0] == "":
        sections.pop(0)
    return len(sections) == N, f"{len(sections)} sections marked by '{section_splitter}', required exactly {N}"


def validate_json_format(text: str) -> Result:
    try:
        json.loads(text)
    except ValueError:
        return False, "the whole response is not valid JSON"
    return True, "valid JSON"


def validate_repeat_prompt(text: str, original_prompt: str) -> Result:
    ok = text.startswith(original_prompt)
    return ok, "response starts with the request verbatim" if ok else "response does not start with the request repeated verbatim"


def validate_two_responses(text: str) -> Result:
    count = text.count("******")
    if count != 1:
        return False, f"found {count} '******' separators, required exactly one"
    first, second = (part.strip() for part in text.split("******"))
    ok = first != second
    return ok, "two different responses" if ok else "the two responses are identical"


def validate_uppercase(text: str) -> Result:
    ok = text == text.upper()
    return ok, "all uppercase" if ok else f"{sum(1 for c in text if c.islower())} lowercase letters present"


def validate_lowercase(text: str) -> Result:
    ok = text == text.lower()
    return ok, "all lowercase" if ok else f"{sum(1 for c in text if c.isupper())} uppercase letters present"


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> Result:
    actual = len(re.findall(r"\b[A-Z]+\b", text))
    ok = _quantified(actual, N, quantifier, max(round(N * 0.1), 1))
    return ok, f"{actual} all-capital words, required {quantifier} {N}"


def validate_end(text: str, end_phrase: str) -> Result:
    ok = text.endswith(end_phrase)
    return ok, f"response {'ends' if ok else 'does not end'} with '{end_phrase}'"


def validate_quotation(text: str) -> Result:
    ok = text.startswith('"') and text.endswith('"')
    return ok, "wrapped in double quotes" if ok else "not wrapped in double quotation marks"


def validate_no_commas(text: str) -> Result:
    count = text.count(",")
    return count == 0, "no commas" if count == 0 else f"{count} commas present"


CHECKERS: dict[str, Callable[..., Result]] = {
    f.__name__: f
    for f in (
        verify_keywords, verify_keyword_frequency, validate_forbidden_words, verify_letter_frequency,
        verify_paragraph_count, validate_word_constraint, verify_sentence_constraint, validate_paragraphs,
        verify_postscript, validate_placeholders, verify_bullet_points, validate_title, validate_choice,
        validate_highlighted_sections, validate_sections, validate_json_format, validate_repeat_prompt,
        validate_two_responses, validate_uppercase, validate_lowercase, validate_frequency_capital_words,
        validate_end, validate_quotation, validate_no_commas,
    )
}


def check(text: str, ground_truth: dict) -> Result:
    """Run the checker named in ground_truth with its non-null arguments."""
    func = CHECKERS[ground_truth["func_name"]]
    args = {k: v for k, v in ground_truth.items() if k != "func_name" and v is not None}
    return func(text, **args)
