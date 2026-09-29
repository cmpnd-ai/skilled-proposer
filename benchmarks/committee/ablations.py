"""Named configurations the harness can run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from skilled_proposer import SkilledProposer

SKILLS = [str(Path(__file__).resolve().parents[2] / "skills" / "prompt-engineering")]


@dataclass
class Ablation:
    name: str
    proposer_kwargs: dict[str, Any] | None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    reflection_minibatch_size: int | None = None

    @property
    def uses_stock_proposer(self) -> bool:
        return self.proposer_kwargs is None

    def build_proposer(self, run_dir: Path) -> SkilledProposer | None:
        if self.proposer_kwargs is None:
            return None
        kwargs = dict(self.proposer_kwargs)
        if kwargs.get("journal"):
            kwargs["journal_path"] = run_dir / "journal.json"
        if kwargs.get("review") == "seen":
            kwargs["seen_path"] = run_dir / "seen.json"
        return SkilledProposer(skills=SKILLS, **kwargs)

    def gepa_kwargs(self, extra_callbacks: list[Any]) -> dict[str, Any]:
        return {"callbacks": list(extra_callbacks), **self.engine_kwargs}


LONG_GUIDANCE = (
    "Write a thorough instruction of roughly 500 words. Cover the decision "
    "rules, the output format, and the edge cases the examples reveal."
)

ABLATIONS: dict[str, Callable[[], Ablation]] = {
    "stock": lambda: Ablation("stock", None),
    "baseline": lambda: Ablation("baseline", {}),
    "journal": lambda: Ablation("journal", {"journal": True}),
    "baseline-long": lambda: Ablation("baseline-long", {"additional_instructions": LONG_GUIDANCE}),
    "baseline-short": lambda: Ablation("baseline-short", {"max_words": 150}),
    "journal-long": lambda: Ablation("journal-long", {"journal": True, "additional_instructions": LONG_GUIDANCE}),
    "rlm": lambda: Ablation("rlm", {"engine": "rlm", "journal": True}),
    "rlm-plain": lambda: Ablation("rlm-plain", {"engine": "rlm"}),
    "rlm-seen": lambda: Ablation(
        "rlm-seen", {"engine": "rlm", "review": "seen", "journal": True},
        reflection_minibatch_size=40,
    ),
}


def resolve(names: list[str]) -> list[Ablation]:
    if names == ["all"]:
        names = list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {unknown}. Known: {list(ABLATIONS)}")
    return [ABLATIONS[n]() for n in names]
