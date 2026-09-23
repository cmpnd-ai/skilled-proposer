"""Deterministic compaction of long agent trajectories in GEPA reflective examples.

DSPy turns each predictor input into a string before the proposer sees it.
The readers here recognize the strings that dspy.ReAct, dspy.ReActV2, and
dspy.RLM produce, keep every decision in full, and shorten tool results.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

_DEDUPE_MIN_CHARS = 200
_COMPACTED_KEYS = ("Inputs", "Generated Outputs")


@dataclass(frozen=True)
class Compaction:
    """Options for compacting reflective examples.

    Args:
        observation_chars: Characters kept from the start of each tool result.
        max_field_chars: Cap for any other long string field.
        examples_token_budget: Optional limit, in tokens, for the rendered
            examples block of one proposal call.
    """

    observation_chars: int = 500
    max_field_chars: int = 2000
    examples_token_budget: int | None = None

    def __post_init__(self):
        if self.observation_chars <= 0:
            raise ValueError("observation_chars must be positive")
        if self.max_field_chars <= 0:
            raise ValueError("max_field_chars must be positive")
        if self.examples_token_budget is not None and self.examples_token_budget <= 0:
            raise ValueError("examples_token_budget must be positive or None")


@dataclass
class CompactionStats:
    """What one compaction pass kept and cut."""

    examples_kept: int
    examples_dropped: int
    chars_before: int
    chars_after: int
    tokens: int | None = None
    over_budget: bool = False


def compact_examples(
    examples: Sequence[Mapping[str, Any]],
    config: Compaction,
    render: Callable[[Sequence[Mapping[str, Any]]], str],
    count_tokens: Callable[[str], int],
) -> tuple[list[dict[str, Any]], CompactionStats]:
    """Return compacted copies of ``examples`` and stats about the pass."""
    out = _compact_all(examples, config.observation_chars, config.max_field_chars)
    stats = CompactionStats(
        examples_kept=len(out),
        examples_dropped=0,
        chars_before=len(render(examples)),
        chars_after=len(render(out)),
    )
    return out, stats


def _compact_all(examples: Sequence[Mapping[str, Any]], obs: int, cap: int) -> list[dict[str, Any]]:
    out = [_compact_example(e, obs, cap) for e in examples]
    _dedupe_inputs(out)
    return out


def _compact_example(example: Mapping[str, Any], obs: int, cap: int) -> dict[str, Any]:
    new = dict(example)
    for key in _COMPACTED_KEYS:
        if key not in new:
            continue
        value = new[key]
        if isinstance(value, Mapping):
            new[key] = {k: _compact_value(v, obs, cap) for k, v in value.items()}
        else:
            new[key] = _compact_value(value, obs, cap)
    return new


def _compact_value(value: Any, obs: int, cap: int) -> Any:
    if not isinstance(value, str):
        return value
    for reader in (_read_react,):
        rendered = reader(value, obs, cap)
        if rendered is not None:
            return rendered
    return _head(value, cap)


def _dedupe_inputs(examples: list[dict[str, Any]]) -> None:
    first_seen: dict[tuple[str, str], int] = {}
    for i, example in enumerate(examples, 1):
        inputs = example.get("Inputs")
        if not isinstance(inputs, dict):
            continue
        for k, v in inputs.items():
            if not isinstance(v, str) or len(v) <= _DEDUPE_MIN_CHARS:
                continue
            seen = first_seen.setdefault((k, v), i)
            if seen != i:
                inputs[k] = f"(same as Example {seen})"


def _head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} [{len(text) - limit:,} of {len(text):,} characters cut]"


# --------------------------------------------------------------------------- #
# Steps and rendering
# --------------------------------------------------------------------------- #

@dataclass
class _Call:
    name: str
    args: Any
    result: str | None = None
    is_error: bool = False


@dataclass
class _Step:
    number: int
    thought: str = ""
    calls: list[_Call] | None = None
    fields: dict[str, str] | None = None


def _render_steps(steps: list[_Step], obs: int, cap: int) -> str:
    cut = 0
    blocks = []
    for step in steps:
        lines = [f"Step {step.number}"]
        for k, v in (step.fields or {}).items():
            lines.append(_line(k, _head(v, cap)))
        if step.thought:
            lines.append(_line("thought", step.thought))
        for call in step.calls or []:
            lines.append(_line("call", _format_call(call.name, call.args)))
            if call.result is not None:
                cut += max(0, len(call.result) - obs)
                label = "error" if call.is_error else "result"
                lines.append(_line(label, _head(call.result, obs)))
        blocks.append("\n".join(lines))
    noun = "step" if len(steps) == 1 else "steps"
    header = f"Trajectory: {len(steps)} {noun}, {cut:,} characters cut from tool results."
    return "\n\n".join([header] + blocks)


def _line(label: str, text: str) -> str:
    return f"  {label}: " + text.replace("\n", "\n    ")


def _format_call(name: str, args: Any) -> str:
    if isinstance(args, Mapping):
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in args.items())
        return f"{name}({inner})"
    return f"{name}({args})" if args not in (None, "") else f"{name}()"


# --------------------------------------------------------------------------- #
# dspy.ReAct: the adapter's user message format
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Style:
    first: re.Pattern
    opener: str
    closer: str = ""


_STYLES = (
    _Style(re.compile(r"\[\[ ## thought_(\d+) ## \]\]\n"), "[[ ## {} ## ]]\n"),
    _Style(re.compile(r"<thought_(\d+)>\n"), "<{}>\n", "\n</{}>"),
    _Style(re.compile(r"thought_(\d+): "), "{}: "),
)

_REACT_ORDER = ("thought", "tool_name", "tool_args", "observation")


def _react_next(key: str) -> list[str]:
    part, _, index = key.rpartition("_")
    n = int(index)
    if part == "tool_args":
        return [f"observation_{n}", f"thought_{n + 1}"]
    if part == "observation":
        return [f"thought_{n + 1}"]
    return [f"{_REACT_ORDER[_REACT_ORDER.index(part) + 1]}_{n}"]


def _read_react(text: str, obs: int, cap: int) -> str | None:
    for style in _STYLES:
        match = style.first.match(text)
        if match:
            break
    else:
        return None

    values: dict[str, str] = {}
    key = f"thought_{match.group(1)}"
    pos = match.end()
    while True:
        end, next_key = len(text), None
        for candidate in _react_next(key):
            i = text.find("\n\n" + style.opener.format(candidate), pos)
            if i != -1:
                end, next_key = i, candidate
                break
        value = text[pos:end]
        closer = style.closer.format(key)
        if closer and value.endswith(closer):
            value = value[: -len(closer)]
        values[key] = value
        if next_key is None:
            break
        pos = end + 2 + len(style.opener.format(next_key))
        key = next_key

    steps: dict[int, _Step] = {}
    for key, value in values.items():
        part, _, index = key.rpartition("_")
        n = int(index)
        step = steps.setdefault(n, _Step(number=n + 1, calls=[]))
        if part == "thought":
            step.thought = value
        elif part == "tool_name":
            step.calls.append(_Call(name=value, args=None))
        elif part == "tool_args" and step.calls:
            step.calls[-1].args = _parse_args(value)
        elif part == "observation" and step.calls:
            step.calls[-1].result = value
    return _render_steps(list(steps.values()), obs, cap)


def _parse_args(text: str) -> Any:
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(text)
        except (ValueError, SyntaxError, TypeError):
            continue
    return text
