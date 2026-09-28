"""JSON values the RLM engine loads into its sandbox."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any, Mapping

import dspy

_RELATIONS = ("current", "ancestor", "other_branch")


class SandboxJSON(dspy.SandboxSerializable):
    """JSON data loaded once into the RLM sandbox.

    RLM re-injects plain values before every code block, which resets any
    change the model makes and re-sends the data each step. A
    SandboxSerializable is injected once, so the model can filter and
    reassign it. The preview is a summary of the data's shape, never a
    record's content, so the model does not anchor on the first record.
    """

    def __init__(self, data: Any, summary: str):
        self.data = data
        self.summary = summary

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return self.summary

    def __str__(self) -> str:
        return self.summary


def to_jsonable(value: Any) -> Any:
    """Convert a reflective record to JSON types. Anything else becomes str()."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def summarize_records(name: str, data: Mapping[str, Any]) -> str:
    """Counts, fields, and field sizes for a records value. No record content."""
    records = data.get("records", [])
    lines = [
        f"`{name}` is a dict loaded from JSON with keys {sorted(data)}.",
        f"`{name}['records']` is a list of {len(records)} record dicts.",
    ]
    fields = [k for k in dict.fromkeys(k for r in records for k in r) if k != "tags"]
    if fields:
        lines.append("Record fields: " + ", ".join(fields) + ", plus `tags` (provenance).")
        for f in fields:
            sizes = [len(json.dumps(r[f])) for r in records if f in r]
            lines.append(
                f"- {f}: {min(sizes)} to {max(sizes)} characters, "
                f"median {int(statistics.median(sizes))}."
            )
    if "instructions" in data:
        lines.append(
            f"`{name}['instructions']` maps {len(data['instructions'])} instruction "
            "hashes to their text, relation, and edits_back."
        )
        counts = Counter(r["tags"]["relation"] for r in records)
        if counts:
            lines.append(
                "Records by relation: "
                + ", ".join(f"{k} {counts[k]}" for k in _RELATIONS if counts[k])
                + "."
            )
    return "\n".join(lines)


def summarize_journal(data: Mapping[str, Any]) -> str:
    """Counts for the journal value. No entry content."""
    entries = data.get("entries", [])
    accepted = sum(1 for e in entries if e["verdict"] == "accepted")
    lines = [
        f"`journal` is a dict loaded from JSON with keys {sorted(data)}.",
        f"{len(data.get('lessons', []))} lessons and "
        f"{len(data.get('hypotheses', []))} hypotheses.",
        f"`journal['entries']` is a list of {len(entries)} proposals: "
        f"{accepted} accepted, {len(entries) - accepted} not chosen as a parent so far.",
    ]
    if entries:
        iterations = [e["iteration"] for e in entries]
        lines.append(
            f"Iterations {min(iterations)} to {max(iterations)}. "
            "Each entry has: " + ", ".join(entries[0]) + "."
        )
    return "\n".join(lines)
