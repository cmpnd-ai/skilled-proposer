"""Named configurations the IFEval harness can run."""

from __future__ import annotations

from typing import Callable

from benchmarks.committee.ablations import LONG_GUIDANCE, SKILLS, Ablation, pxn_engine

ABLATIONS: dict[str, Callable[[], Ablation]] = {
    "stock": lambda: Ablation("stock", None),
    "baseline": lambda: Ablation("baseline", {}),
    "pxn-long": lambda: Ablation("pxn-long", {"additional_instructions": LONG_GUIDANCE}, pxn_engine()),
    "batch-pxn-long": lambda: Ablation(
        "batch-pxn-long", {"candidates": 4, "additional_instructions": LONG_GUIDANCE}, pxn_engine()
    ),
}


def resolve(names: list[str]) -> list[Ablation]:
    if names == ["all"]:
        names = list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {unknown}. Known: {list(ABLATIONS)}")
    return [ABLATIONS[n]() for n in names]


__all__ = ["ABLATIONS", "Ablation", "LONG_GUIDANCE", "SKILLS", "resolve"]
