"""Every reflective record the proposer has seen, tagged by the instruction
that produced it, plus the lineage between those instructions."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from skilled_proposer.sandbox import to_jsonable


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def tag_records(
    records: Sequence[Mapping[str, Any]], *, component: str, iteration: int, instruction: str
) -> list[dict]:
    """JSON copies of GEPA's reflective records with provenance tags."""
    tags = {"iteration": iteration, "component": component, "instruction": text_hash(instruction)}
    return [{**to_jsonable(r), "tags": dict(tags)} for r in records]


def _relation(h: str, edits_back: Mapping[str, int]) -> tuple[str, int | None]:
    if edits_back.get(h) == 0:
        return "current", 0
    if h in edits_back:
        return "ancestor", edits_back[h]
    return "other_branch", None


class SeenStore:
    """Records per component, the instruction texts that produced them, and
    the lineage between those texts.

    Lineage comes from the proposals the proposer returns: each one adds an
    edge from the proposed text to its parent. A text with no edge (the seed,
    or text a GEPA merge produced) is a root.
    """

    def __init__(
        self,
        records: dict[str, list[dict]] | None = None,
        texts: dict[str, dict[str, str]] | None = None,
        edges: dict[str, dict[str, str]] | None = None,
    ):
        self.records: dict[str, list[dict]] = records or {}
        self.texts: dict[str, dict[str, str]] = texts or {}
        self.edges: dict[str, dict[str, str]] = edges or {}

    def add(self, component: str, instruction: str, records: Sequence[dict]) -> None:
        """Keep tagged records that ``instruction`` produced."""
        self.texts.setdefault(component, {})[text_hash(instruction)] = instruction
        self.records.setdefault(component, []).extend(records)

    def link(self, component: str, parent: str, child: str) -> None:
        """Record that ``child`` was proposed from ``parent``. The first parent wins."""
        p, c = text_hash(parent), text_hash(child)
        if p == c:
            return
        texts = self.texts.setdefault(component, {})
        texts.setdefault(p, parent)
        texts.setdefault(c, child)
        self.edges.setdefault(component, {}).setdefault(c, p)

    def view(self, component: str, current: str, iteration: int, seed: int) -> dict:
        """Records from iterations before ``iteration``, tagged by how the
        instruction that produced them relates to ``current``, shuffled."""
        edits_back = self._edits_back(component, text_hash(current))
        texts = self.texts.get(component, {})
        records = []
        for r in self.records.get(component, []):
            if r["tags"]["iteration"] >= iteration:
                continue
            relation, n = _relation(r["tags"]["instruction"], edits_back)
            records.append({**r, "tags": {**r["tags"], "relation": relation, "edits_back": n}})
        random.Random(seed * 1_000_003 + iteration).shuffle(records)
        instructions: dict[str, dict] = {}
        for r in records:
            h = r["tags"]["instruction"]
            if h not in instructions:
                instructions[h] = {
                    "text": texts.get(h, ""),
                    "relation": r["tags"]["relation"],
                    "edits_back": r["tags"]["edits_back"],
                }
        return {"instructions": instructions, "records": records}

    def _edits_back(self, component: str, start: str) -> dict[str, int]:
        """Hash -> distance for the current text and each of its ancestors."""
        edges = self.edges.get(component, {})
        out = {start: 0}
        node, n = start, 0
        while node in edges and edges[node] not in out:
            node, n = edges[node], n + 1
            out[node] = n
        return out

    # -- Persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"records": self.records, "texts": self.texts, "edges": self.edges})

    @classmethod
    def from_json(cls, text: str) -> "SeenStore":
        data = json.loads(text)
        return cls(data.get("records"), data.get("texts"), data.get("edges"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "SeenStore":
        p = Path(path)
        return cls.from_json(p.read_text()) if p.exists() else cls()
