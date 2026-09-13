"""Load and split the committee extraction data."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import dspy

DATA_PATH = Path(__file__).with_name("training.csv")


def load_rows(path: str | Path = DATA_PATH) -> list[dict]:
    """Drop rows with a repeated body so one email cannot appear in two splits."""
    seen_bodies: set[str] = set()
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("committee", "").strip() and r.get("body", "").strip()):
                continue
            body = r["body"].strip()
            if body in seen_bodies:
                continue
            seen_bodies.add(body)
            rows.append(r)
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
