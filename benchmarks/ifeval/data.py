"""Load and split the IFEval-style prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import dspy

DATA_PATH = Path(__file__).with_name("prompts.jsonl")


def load_rows(path: str | Path = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_examples(rows: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(
            prompt=r["prompt"],
            constraint=r["constraint"],
            constraint_type=r["constraint_type"],
            ground_truth=r["ground_truth"],
        ).with_inputs("prompt")
        for r in rows
    ]


def split(
    examples: list[dspy.Example],
    sizes: tuple[int, int, int] = (900, 50, 200),
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
