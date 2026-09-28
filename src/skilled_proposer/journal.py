"""Proposal journal: what the proposer proposed and which proposals GEPA kept."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

_WORD_RE = re.compile(r"\S+")

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    iteration: int
    component: str
    parent_text: str
    proposed_text: str
    change_summary: str = ""
    accepted: bool | None = None
    reason: str = ""
    closed: bool = False
    analysis: str = ""

    @property
    def entry_id(self) -> str:
        return f"{self.iteration}:{self.component}"

    @property
    def verdict(self) -> str:
        return "accepted" if self.accepted else "not chosen as a parent so far"


_ENTRY_FIELDS = {f.name for f in fields(JournalEntry)}


class Journal:
    """Ordered entries plus the lessons distilled from them.

    An entry is accepted once GEPA hands its proposal back as a parent to
    improve. An entry that never comes back stays unchosen.
    """

    def __init__(
        self,
        entries: list[JournalEntry] | None = None,
        lessons: str = "",
        closed_since_distill: int = 0,
        hypotheses: list[str] | None = None,
    ):
        self.entries: list[JournalEntry] = list(entries or [])
        self.lessons = lessons
        self.closed_since_distill = closed_since_distill
        self.hypotheses: list[str] = list(hypotheses or [])

    # -- Entries ------------------------------------------------------------

    def open(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    def open_entries(self, iteration: int) -> list[JournalEntry]:
        return [e for e in self.entries if e.iteration == iteration and not e.closed]

    def close(self, iteration: int) -> int:
        closed = 0
        for e in self.open_entries(iteration):
            e.closed = True
            closed += 1
        self.closed_since_distill += closed
        return closed

    # -- RLM notes ----------------------------------------------------------

    def to_data(self) -> dict:
        """The journal as JSON data for the RLM sandbox."""
        return {
            "lessons": [line for line in self.lessons.splitlines() if line.strip()],
            "hypotheses": list(self.hypotheses),
            "entries": [
                {
                    "id": e.entry_id,
                    "iteration": e.iteration,
                    "component": e.component,
                    "verdict": e.verdict,
                    "reason": e.reason,
                    "change_summary": e.change_summary,
                    "analysis": e.analysis,
                    "parent_text": e.parent_text,
                    "proposed_text": e.proposed_text,
                }
                for e in self.entries
            ],
        }

    def apply_notes(
        self, lessons: list[str], hypotheses: list[str], entry_analysis: dict[str, str]
    ) -> None:
        """Merge the RLM's notes. The facts about each entry never change."""
        self.lessons = "\n".join(line.strip() for line in lessons if line.strip())
        self.hypotheses = [h.strip() for h in hypotheses if h.strip()]
        by_id = {e.entry_id: e for e in self.entries}
        for entry_id, analysis in entry_analysis.items():
            entry = by_id.get(entry_id)
            if entry is None:
                logger.warning("Journal notes named an unknown entry %r; dropping its analysis.", entry_id)
                continue
            entry.analysis = analysis.strip()

    # -- Rendering ----------------------------------------------------------

    def render(self, limit: int | None = None) -> str:
        lessons = self.lessons.strip()
        entries = self.entries[-limit:] if limit else self.entries
        if not lessons and not self.hypotheses and not entries:
            return "No proposals have been recorded yet."
        parts = []
        if lessons:
            parts.append("## Lessons\n\n" + lessons)
        if self.hypotheses:
            parts.append("## Hypotheses\n\n" + "\n".join(self.hypotheses))
        if entries:
            blocks = "\n\n".join(_render_entry(e) for e in entries)
            parts.append("## Recent proposals\n\n" + blocks)
        return "\n\n".join(parts)

    # -- Persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "lessons": self.lessons,
                "closed_since_distill": self.closed_since_distill,
                "hypotheses": self.hypotheses,
                "entries": [asdict(e) for e in self.entries],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Journal":
        data = json.loads(text)
        entries = [
            JournalEntry(**{k: v for k, v in e.items() if k in _ENTRY_FIELDS})
            for e in data.get("entries", [])
        ]
        return cls(
            entries=entries,
            lessons=data.get("lessons", ""),
            closed_since_distill=data.get("closed_since_distill", 0),
            hypotheses=data.get("hypotheses", []),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_json(p.read_text())


def _render_entry(e: JournalEntry) -> str:
    lines = [f"### Iteration {e.iteration}, component `{e.component}`, {e.verdict}"]
    lines.append("Change: " + (e.change_summary.strip() or "No change summary was given."))
    lines.append(_size_line(e))
    if e.reason:
        lines.append(f"Reason: {e.reason}")
    if e.analysis:
        lines.append(f"Analysis: {e.analysis}")
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
