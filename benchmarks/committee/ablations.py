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

    @property
    def uses_stock_proposer(self) -> bool:
        return self.proposer_kwargs is None

    def build_proposer(self, run_dir: Path) -> SkilledProposer | None:
        if self.proposer_kwargs is None:
            return None
        kwargs = dict(self.proposer_kwargs)
        if kwargs.get("journal"):
            kwargs["journal_path"] = run_dir / "journal.json"
        return SkilledProposer(skills=SKILLS, **kwargs)

    def gepa_kwargs(self, proposer: SkilledProposer | None, extra_callbacks: list[Any]) -> dict[str, Any]:
        if proposer is None:
            return {"callbacks": list(extra_callbacks), **self.engine_kwargs}
        return proposer.gepa_kwargs(callbacks=list(extra_callbacks), **self.engine_kwargs)


ABLATIONS: dict[str, Callable[[], Ablation]] = {
    "stock": lambda: Ablation("stock", None),
    "baseline": lambda: Ablation("baseline", {}),
    "journal": lambda: Ablation("journal", {"journal": True}),
    "dedupe": lambda: Ablation("dedupe", {"dedupe": True}),
    "journal+dedupe": lambda: Ablation("journal+dedupe", {"journal": True, "dedupe": True}),
}


def resolve(names: list[str]) -> list[Ablation]:
    if names == ["all"]:
        names = list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {unknown}. Known: {list(ABLATIONS)}")
    return [ABLATIONS[n]() for n in names]
