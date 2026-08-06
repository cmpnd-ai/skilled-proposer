"""Skill loading: named blocks of reference material for the reflection LM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Skill:
    """A named block of reference material for the reflection LM."""

    name: str
    content: str
    description: str | None = None

    @classmethod
    def load(cls, source: "Skill | str | Path") -> "Skill":
        """Load a skill from a Skill, a path, or an inline string.

        - Directory path  -> reads `SKILL.md` inside it (Agent Skills layout).
        - File path       -> reads the file.
        - Any other string-> treated as inline skill content.

        A leading YAML frontmatter block is parsed for `name` and
        `description` and stripped from the content.
        """
        if isinstance(source, Skill):
            return source

        try:
            path = Path(source)
            exists = path.exists()
        except OSError:  # e.g. inline string too long to be a valid path
            exists = False

        if exists:
            if path.is_dir():
                skill_md = path / "SKILL.md"
                if not skill_md.exists():
                    raise FileNotFoundError(
                        f"Skill directory {path} has no SKILL.md"
                    )
                text = skill_md.read_text(encoding="utf-8")
                fallback_name = path.name
            else:
                text = path.read_text(encoding="utf-8")
                fallback_name = path.stem
        else:
            text = str(source).strip()
            if not text:
                raise ValueError("Empty skill content")
            fallback_name = None

        meta, content = _parse_frontmatter(text)
        content = content.strip()

        if fallback_name is None:
            # First non-empty line doubles as a display name for inline skills.
            first_line = content.splitlines()[0].lstrip("# ").strip() if content else ""
            fallback_name = (
                (first_line[:60] + "…") if len(first_line) > 60 else first_line
            ) or "inline-skill"

        return cls(
            name=meta.get("name") or fallback_name,
            content=content,
            description=meta.get("description"),
        )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading `--- ... ---` block into (metadata, body).

    Only flat, unindented `key: value` lines are read; nested YAML is
    ignored. Returns ({}, text) when there is no well-formed block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            meta: dict[str, str] = {}
            for raw in lines[1:end]:
                if raw.startswith((" ", "\t")) or ":" not in raw:
                    continue
                key, _, value = raw.partition(":")
                meta[key.strip()] = value.strip().strip("'\"")
            return meta, "\n".join(lines[end + 1 :])
    return {}, text


def render_skills(skills: Sequence[Skill]) -> str:
    """Render skills as <skill> blocks for a reflection prompt, or 'None'."""
    if not skills:
        return "None"
    parts = []
    for skill in skills:
        attrs = f"name={skill.name!r}"
        if skill.description:
            attrs += f" description={skill.description!r}"
        parts.append(f"<skill {attrs}>\n{skill.content.strip()}\n</skill>")
    return "\n\n".join(parts)
