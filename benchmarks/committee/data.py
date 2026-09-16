"""Load and split the committee extraction data."""

from __future__ import annotations

import csv
import random
import re
import unicodedata
from pathlib import Path

import dspy

DATA_PATH = Path(__file__).with_name("training.csv")

_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")


def clean_body(text: str) -> str:
    """Remove invisible padding characters that email preheaders use.

    Marketing emails pad the preview text with hundreds of combining
    joiners and zero width characters. They carry no content, they
    inflate the prompt, and a long run of them crashed the local student
    model. Format characters and the combining grapheme joiner are
    dropped, then repeated spaces are collapsed.
    """
    kept = []
    for ch in text:
        if ch == "͏" or unicodedata.category(ch) == "Cf":
            continue
        kept.append(ch)
    return _WHITESPACE_RUN.sub(" ", "".join(kept)).strip()


def load_rows(path: str | Path = DATA_PATH) -> list[dict]:
    """Clean each body and drop repeats so one email cannot appear in two splits."""
    seen_bodies: set[str] = set()
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            body = clean_body(r.get("body", ""))
            if not (r.get("committee", "").strip() and body):
                continue
            if body in seen_bodies:
                continue
            seen_bodies.add(body)
            rows.append({**r, "body": body})
    return rows


def make_examples(rows: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(email_body=r["body"].strip(), committee=r["committee"].strip())
        .with_inputs("email_body")
        for r in rows
    ]


def split(
    examples: list[dspy.Example],
    sizes: tuple[int, int, int] = (556, 200, 200),
    seed: int = 13,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Seeded shuffle, then train, validation, and test slices in that order."""
    if sum(sizes) > len(examples):
        raise ValueError(f"split sizes {sizes} exceed {len(examples)} examples")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_train, n_val, n_test = sizes
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]
    return train, val, test
