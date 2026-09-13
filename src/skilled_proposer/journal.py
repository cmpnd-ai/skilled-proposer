"""Proposal journal: what the proposer proposed and what GEPA did with it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    near_duplicates: list[str] = field(default_factory=list)
    duplicate_after_retry: bool = False
    closed: bool = False

    @property
    def verdict(self) -> str:
        if self.accepted is True:
            return "accepted"
        if self.accepted is False:
            return "rejected"
        return "not evaluated"


class Journal:
    """Ordered entries plus the lessons distilled from them."""

    def __init__(
        self,
        entries: list[JournalEntry] | None = None,
        lessons: str = "",
        closed_since_distill: int = 0,
    ):
        self.entries: list[JournalEntry] = list(entries or [])
        self.lessons = lessons
        self.closed_since_distill = closed_since_distill

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

    def rejected_texts(self, component: str) -> list[str]:
        return [
            e.proposed_text
            for e in self.entries
            if e.closed and e.accepted is False and e.component == component
        ]

    # -- Persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "lessons": self.lessons,
                "closed_since_distill": self.closed_since_distill,
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
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_json(p.read_text())
