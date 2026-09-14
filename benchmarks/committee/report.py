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
    ("duplicates", "dups"),
    ("auc_valset_vs_calls", "auc"),
]


def load_summaries(root: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/*/summary.json"))]


def _cell(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
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
