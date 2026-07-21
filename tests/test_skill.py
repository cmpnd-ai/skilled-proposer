from pathlib import Path

import pytest

from skilled_proposer.skill import Skill

FIXTURE = Path(__file__).parent.parent / "skills" / "prompt-engineering"


def test_skill_passthrough():
    s = Skill(name="x", content="y")
    assert Skill.load(s) is s


def test_inline_skill_name_from_first_line():
    s = Skill.load("# My Guide\nDo things well.")
    assert s.name == "My Guide"
    assert "Do things well." in s.content
    assert s.description is None


def test_inline_skill_long_name_truncated():
    s = Skill.load("x" * 100 + "\nbody")
    assert len(s.name) == 61  # 60 chars + ellipsis
    assert s.name.endswith("…")


def test_empty_inline_skill_raises():
    with pytest.raises(ValueError):
        Skill.load("   ")


def test_file_skill(tmp_path):
    f = tmp_path / "my-skill.md"
    f.write_text("Some guidance.", encoding="utf-8")
    s = Skill.load(f)
    assert s.name == "my-skill"
    assert s.content == "Some guidance."


def test_directory_skill(tmp_path):
    d = tmp_path / "my-dir-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("Dir guidance.", encoding="utf-8")
    s = Skill.load(d)
    assert s.name == "my-dir-skill"
    assert s.content == "Dir guidance."


def test_directory_without_skill_md_raises(tmp_path):
    d = tmp_path / "empty-dir"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        Skill.load(d)


def test_frontmatter_parsed_and_stripped(tmp_path):
    f = tmp_path / "fm.md"
    f.write_text(
        "---\nname: custom-name\ndescription: What it does.\n---\n\n# Body\ntext",
        encoding="utf-8",
    )
    s = Skill.load(f)
    assert s.name == "custom-name"
    assert s.description == "What it does."
    assert s.content.startswith("# Body")
    assert "---" not in s.content


def test_frontmatter_inline_string():
    s = Skill.load("---\nname: inline-fm\n---\nbody text")
    assert s.name == "inline-fm"
    assert s.content == "body text"


def test_frontmatter_without_closing_fence_is_content():
    s = Skill.load("--- not frontmatter\nbody")
    assert "--- not frontmatter" in s.content


def test_frontmatter_ignores_indented_lines(tmp_path):
    f = tmp_path / "nested.md"
    f.write_text(
        "---\nname: n\nmetadata:\n  type: user\n---\nbody",
        encoding="utf-8",
    )
    s = Skill.load(f)
    assert s.name == "n"
    assert s.content == "body"


def test_prompt_engineering_fixture():
    s = Skill.load(FIXTURE)
    assert s.name == "prompt-engineering"
    assert s.description is not None
    assert s.description.startswith("Use when optimizing")
    assert not s.content.startswith("---")
    assert "Prompt Engineering" in s.content
