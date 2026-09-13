"""Load and split the committee extraction data."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import dspy

DATA_PATH = Path(__file__).with_name("training.csv")


def load_rows(path: str | Path = DATA_PATH) -> list[dict]:
    with open(path, newline="") as f:
        return [
            r for r in csv.DictReader(f)
            if r.get("committee", "").strip() and r.get("body", "").strip()
        ]


def make_examples(rows: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(email_body=r["body"].strip(), committee=r["committee"].strip())
        .with_inputs("email_body")
        for r in rows
    ]


def split(
    examples: list[dspy.Example],
    sizes: tuple[int, int, int] = (600, 200, 200),
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
