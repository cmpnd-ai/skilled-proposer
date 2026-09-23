"""Deterministic compaction of long agent trajectories in GEPA reflective examples.

DSPy turns each predictor input into a string before the proposer sees it.
The readers here recognize the strings that dspy.ReAct, dspy.ReActV2, and
dspy.RLM produce, keep every decision in full, and shorten tool results.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

_DEDUPE_MIN_CHARS = 200
_COMPACTED_KEYS = ("Inputs", "Generated Outputs")


@dataclass(frozen=True)
class Compaction:
    """Options for compacting reflective examples.

    Args:
        observation_chars: Characters kept from the start of each tool result.
        max_field_chars: Cap for any other long string field.
        examples_token_budget: Optional limit, in tokens, for the rendered
            examples block of one proposal call.
    """

    observation_chars: int = 500
    max_field_chars: int = 2000
    examples_token_budget: int | None = None

    def __post_init__(self):
        if self.observation_chars <= 0:
            raise ValueError("observation_chars must be positive")
        if self.max_field_chars <= 0:
            raise ValueError("max_field_chars must be positive")
        if self.examples_token_budget is not None and self.examples_token_budget <= 0:
            raise ValueError("examples_token_budget must be positive or None")


@dataclass
class CompactionStats:
    """What one compaction pass kept and cut."""

    examples_kept: int
    examples_dropped: int
    chars_before: int
    chars_after: int
    tokens: int | None = None
    over_budget: bool = False


def compact_examples(
    examples: Sequence[Mapping[str, Any]],
    config: Compaction,
    render: Callable[[Sequence[Mapping[str, Any]]], str],
    count_tokens: Callable[[str], int],
) -> tuple[list[dict[str, Any]], CompactionStats]:
    """Return compacted copies of ``examples`` and stats about the pass."""
    out = _compact_all(examples, config.observation_chars, config.max_field_chars)
    stats = CompactionStats(
        examples_kept=len(out),
        examples_dropped=0,
        chars_before=len(render(examples)),
        chars_after=len(render(out)),
    )
    return out, stats


def _compact_all(examples: Sequence[Mapping[str, Any]], obs: int, cap: int) -> list[dict[str, Any]]:
    out = [_compact_example(e, obs, cap) for e in examples]
    _dedupe_inputs(out)
    return out


def _compact_example(example: Mapping[str, Any], obs: int, cap: int) -> dict[str, Any]:
    new = dict(example)
    for key in _COMPACTED_KEYS:
        if key not in new:
            continue
        value = new[key]
        if isinstance(value, Mapping):
            new[key] = {k: _compact_value(v, obs, cap) for k, v in value.items()}
        else:
            new[key] = _compact_value(value, obs, cap)
    return new


def _compact_value(value: Any, obs: int, cap: int) -> Any:
    if not isinstance(value, str):
        return value
    return _head(value, cap)


def _dedupe_inputs(examples: list[dict[str, Any]]) -> None:
    first_seen: dict[tuple[str, str], int] = {}
    for i, example in enumerate(examples, 1):
        inputs = example.get("Inputs")
        if not isinstance(inputs, dict):
            continue
        for k, v in inputs.items():
            if not isinstance(v, str) or len(v) <= _DEDUPE_MIN_CHARS:
                continue
            seen = first_seen.setdefault((k, v), i)
            if seen != i:
                inputs[k] = f"(same as Example {seen})"


def _head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} [{len(text) - limit:,} of {len(text):,} characters cut]"
