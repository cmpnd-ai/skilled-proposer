"""Records how a GEPA run progresses, one row per iteration."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


class TrajectoryRecorder:
    """A GEPACallback that does not depend on which proposer is running.

    Like the journal, this recorder pairs a verdict to a proposal by
    position within the iteration and assumes one proposal per iteration,
    GEPA's default sampling.
    """

    def __init__(self):
        self.rows: list[dict[str, Any]] = []
        self.reflection_calls = 0
        self.metric_calls_used = 0
        self.best_valset_score: float | None = None
        self.metric_calls_to_best: int | None = None
        self._current: dict[str, Any] = {}

    # -- Callbacks ----------------------------------------------------------

    def on_budget_updated(self, event: Mapping[str, Any]) -> None:
        self.metric_calls_used = event["metric_calls_used"]

    def on_proposal_start(self, event: Mapping[str, Any]) -> None:
        self.reflection_calls += 1

    def on_valset_evaluated(self, event: Mapping[str, Any]) -> None:
        score = event["average_score"]
        if self.best_valset_score is None or score > self.best_valset_score:
            self.best_valset_score = score
            calls = self.metric_calls_used if self.metric_calls_used > 0 else event.get("num_examples_evaluated", 0)
            self.metric_calls_to_best = calls
            self.metric_calls_used = calls

    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        self._current["verdict"] = "accepted"

    def on_candidate_rejected(self, event: Mapping[str, Any]) -> None:
        self._current["verdict"] = "rejected"
        self._current["reason"] = event.get("reason", "")

    def on_merge_accepted(self, event: Mapping[str, Any]) -> None:
        self._current["verdict"] = "merge_accepted"
        self._current["is_merge"] = True

    def on_merge_rejected(self, event: Mapping[str, Any]) -> None:
        self._current["verdict"] = "merge_rejected"
        self._current["is_merge"] = True

    def on_iteration_end(self, event: Mapping[str, Any]) -> None:
        state = event["state"]
        trace = state.full_program_trace[-1] if getattr(state, "full_program_trace", None) else {}
        before, after = _minibatch_means(trace)
        row = {
            "iteration": event["iteration"],
            "metric_calls_used": self.metric_calls_used,
            "reflection_calls": self.reflection_calls,
            "best_valset_score": self.best_valset_score,
            "verdict": self._current.get("verdict", "skipped"),
            "reason": self._current.get("reason", ""),
            "is_merge": self._current.get("is_merge", False),
            "minibatch_before": before,
            "minibatch_after": after,
            "num_candidates": len(getattr(state, "program_candidates", [])),
        }
        self.rows.append(row)
        self._current = {}

    # -- Output -------------------------------------------------------------

    def write_trajectory(self, path: str | Path) -> None:
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def summary(self) -> dict[str, Any]:
        proposals = [r for r in self.rows if r["verdict"] in ("accepted", "rejected")]
        accepted = sum(1 for r in proposals if r["verdict"] == "accepted")
        return {
            "iterations": len(self.rows),
            "best_valset_score": self.best_valset_score,
            "metric_calls_to_best": self.metric_calls_to_best,
            "accept_rate": (accepted / len(proposals)) if proposals else 0.0,
            "reflection_calls": self.reflection_calls,
            "metric_calls_used": self.metric_calls_used,
            "auc_valset_vs_calls": _auc(self.rows),
        }


def _minibatch_means(trace: Mapping[str, Any]) -> tuple[float | None, float | None]:
    tasks = [t for t in trace.get("tasks", []) if "new_subsample_scores" in t]
    if not tasks:
        return None, None
    before = [s for t in tasks for s in t.get("subsample_scores", [])]
    after = [s for t in tasks for s in t.get("new_subsample_scores", [])]
    return (fmean(before) if before else None, fmean(after) if after else None)


def _auc(rows: list[dict[str, Any]]) -> float:
    """Area under best valset score against metric calls, by the trapezoid rule."""
    points = [(r["metric_calls_used"], r["best_valset_score"]) for r in rows if r["best_valset_score"] is not None]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area
