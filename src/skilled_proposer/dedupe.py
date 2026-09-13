"""Near-duplicate screening for proposed instructions."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable

_SCREENING_SETS = ("rejected", "pool")
_DUPLICATE_POLICIES = ("return", "skip")


@dataclass(frozen=True)
class DedupeConfig:
    """Settings for the dedupe step.

    threshold: similarity at or above this is a near duplicate.
    max_retries: diversify calls per proposal. A hard cap.
    against: which sets to screen against. "rejected" is the journal's
        rejected proposals for the component. "pool" is the current
        candidate pool plus the parent instruction.
    on_duplicate: what to do when a proposal is still a duplicate after the
        cap. "return" returns it and lets GEPA judge. "skip" leaves the
        component out of the proposal.
    """

    threshold: float = 0.85
    max_retries: int = 1
    against: tuple[str, ...] = _SCREENING_SETS
    on_duplicate: str = "return"

    def __post_init__(self):
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not self.against or any(s not in _SCREENING_SETS for s in self.against):
            raise ValueError(f"against must be a non-empty subset of {_SCREENING_SETS}")
        if self.on_duplicate not in _DUPLICATE_POLICIES:
            raise ValueError(f"on_duplicate must be one of {_DUPLICATE_POLICIES}")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def similarity(a: str, b: str) -> float:
    """The larger of the character sequence ratio and the token Jaccard index."""
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sequence = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb)
    return max(sequence, jaccard)


def find_duplicates(
    text: str, candidates: Iterable[tuple[str, str]], threshold: float
) -> list[tuple[str, str, float]]:
    """Return (label, text, score) for each candidate at or above the threshold."""
    matches = []
    for label, other in candidates:
        score = similarity(text, other)
        if score >= threshold:
            matches.append((label, other, score))
    matches.sort(key=lambda m: m[2], reverse=True)
    return matches


def render_duplicates(matches: list[tuple[str, str, float]]) -> str:
    blocks = [
        f"{i}. {label} (similarity {score:.2f})\n{other}"
        for i, (label, other, score) in enumerate(matches, 1)
    ]
    return "\n\n".join(blocks)
