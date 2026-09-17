"""Proposal journal: what the proposer proposed and what GEPA did with it."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_WORD_RE = re.compile(r"\S+")


@dataclass
class JournalEntry:
    iteration: int
    task: int
    component: str
    parent_text: str
    proposed_text: str
    parent_idx: int | None = None
    change_summary: str = ""
    minibatch_before: float | None = None
    minibatch_after: float | None = None
    valset_average: float | None = None
    accepted: bool | None = None
    reason: str = ""
    closed: bool = False

    @property
    def verdict(self) -> str:
        if self.accepted is True:
            return "accepted"
        if self.accepted is False:
            return "rejected"
        return "not evaluated"


class Journal:
    """Ordered entries plus the lessons distilled from them.

    ``source`` says where verdicts come from. "callbacks" means GEPA reported
    them. "lineage" means the proposer inferred them from which proposals came
    back as parents, so an entry without a verdict was simply never chosen.
    """

    def __init__(
        self,
        entries: list[JournalEntry] | None = None,
        lessons: str = "",
        closed_since_distill: int = 0,
        source: str = "callbacks",
    ):
        self.entries: list[JournalEntry] = list(entries or [])
        self.lessons = lessons
        self.closed_since_distill = closed_since_distill
        self.source = source

    # -- Entries ------------------------------------------------------------

    def open(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    def open_entries(self, iteration: int) -> list[JournalEntry]:
        return [e for e in self.entries if e.iteration == iteration and not e.closed]

    def open_tasks(self, iteration: int) -> list[int]:
        seen: list[int] = []
        for e in self.open_entries(iteration):
            if e.task not in seen:
                seen.append(e.task)
        return seen

    def entries_for_task(self, iteration: int, task: int) -> list[JournalEntry]:
        return [e for e in self.open_entries(iteration) if e.task == task]

    def close(self, iteration: int) -> int:
        closed = 0
        for e in self.open_entries(iteration):
            e.closed = True
            closed += 1
        self.closed_since_distill += closed
        return closed

    # -- Rendering ----------------------------------------------------------

    def render(self, limit: int | None = None) -> str:
        lessons = self.lessons.strip()
        entries = self.entries[-limit:] if limit else self.entries
        if not lessons and not entries:
            return "No proposals have been recorded yet."
        parts = []
        if lessons:
            parts.append("## Lessons\n\n" + lessons)
        if entries:
            blocks = "\n\n".join(_render_entry(e, self.source) for e in entries)
            parts.append("## Recent proposals\n\n" + blocks)
        return "\n\n".join(parts)

    # -- Persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "lessons": self.lessons,
                "closed_since_distill": self.closed_since_distill,
                "source": self.source,
                "entries": [asdict(e) for e in self.entries],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Journal":
        data = json.loads(text)
        return cls(
            entries=[JournalEntry(**e) for e in data.get("entries", [])],
            lessons=data.get("lessons", ""),
            closed_since_distill=data.get("closed_since_distill", 0),
            source=data.get("source", "callbacks"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_json(p.read_text())


def _render_entry(e: JournalEntry, source: str = "callbacks") -> str:
    verdict = e.verdict
    if e.accepted is None and source == "lineage":
        verdict = "not chosen as a parent so far"
    lines = [f"### Iteration {e.iteration}, component `{e.component}`, {verdict}"]
    if e.accepted is not None and e.minibatch_before is not None and e.minibatch_after is not None:
        score = f"Minibatch {e.minibatch_before:.2f} -> {e.minibatch_after:.2f}."
        if e.valset_average is not None:
            score += f" Valset average {e.valset_average:.2f}."
        else:
            score += " Not on the valset."
        lines.append(score)
    lines.append("Change: " + (e.change_summary.strip() or "No change summary was given."))
    lines.append(_size_line(e))
    if e.reason:
        lines.append(f"Reason: {e.reason}")
    return "\n".join(lines)


def _size_line(e: JournalEntry) -> str:
    words = len(_WORD_RE.findall(e.proposed_text))
    delta = words - len(_WORD_RE.findall(e.parent_text))
    if delta > 0:
        tail = f"{delta} more than the parent."
    elif delta < 0:
        tail = f"{-delta} fewer than the parent."
    else:
        tail = "the same as the parent."
    return f"Size: {words} words, {tail}"
