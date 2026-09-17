"""Summarize benchmark results across ablations and seeds.

    uv run python -m benchmarks.committee.report --results benchmarks/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

COLUMNS = [
    ("test_score", "test"),
    ("best_valset_score", "best val"),
    ("metric_calls_to_best", "calls to best"),
    ("accept_rate", "accept"),
    ("reflection_calls", "reflect calls"),
    ("auc_valset_vs_calls", "auc"),
    ("best_instruction_words", "words"),
]


def saved_instruction_words(program_path: Path) -> int | None:
    """Word count of the instructions in a saved program file, or None when absent."""
    if not program_path.exists():
        return None
    program = json.loads(program_path.read_text())
    total = 0
    found = False
    for value in program.values():
        if isinstance(value, dict) and isinstance(value.get("signature"), dict):
            total += len(value["signature"].get("instructions", "").split())
            found = True
    return total if found else None


def load_summaries(root: str | Path) -> list[dict]:
    """Read every summary, filling the instruction word count from the saved program when missing."""
    summaries = []
    for p in sorted(Path(root).glob("*/*/summary.json")):
        summary = json.loads(p.read_text())
        if "best_instruction_words" not in summary:
            summary["best_instruction_words"] = saved_instruction_words(p.with_name("program.json"))
        summaries.append(summary)
    return summaries


def _cell(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    whole = all(isinstance(v, int) for v in values)
    if len(values) == 1:
        return f"{values[0]:d}" if whole else f"{values[0]:.3f}"
    if whole:
        return f"{fmean(values):.0f} ± {pstdev(values):.0f}"
    return f"{fmean(values):.3f} ± {pstdev(values):.3f}"


def build_table(summaries: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        grouped[s["ablation"]].append(s)
    header = ["ablation", "n"] + [label for _, label in COLUMNS]
    rows = [header]
    for name, runs in grouped.items():
        rows.append([name, str(len(runs))] + [_cell([r.get(key) for r in runs]) for key, _ in COLUMNS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)


def write_curves(root: str | Path, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "seed", "iteration", "metric_calls_used", "best_valset_score"])
        for traj in sorted(Path(root).glob("*/*/trajectory.jsonl")):
            ablation, seed = traj.parent.parent.name, traj.parent.name
            for line in traj.read_text().splitlines():
                row = json.loads(line)
                w.writerow([ablation, seed, row["iteration"], row["metric_calls_used"], row["best_valset_score"]])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="benchmarks/results")
    args = p.parse_args(argv)
    summaries = load_summaries(args.results)
    if not summaries:
        raise SystemExit(f"No summary.json files under {args.results}")
    print(build_table(summaries))
    out = Path(args.results) / "curves.csv"
    write_curves(args.results, out)
    print(f"\nCurves written to {out}")


if __name__ == "__main__":
    main()
