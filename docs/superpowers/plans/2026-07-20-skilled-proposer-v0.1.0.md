# skilled-proposer v0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `reference_code.py` into a shippable PyPI package named `skilled-proposer`, with tests, CI, and the GitHub remote wired up.

**Architecture:** A src-layout package `skilled_proposer` split into three modules. `skill.py` holds the `Skill` dataclass and loader with frontmatter parsing. `signatures.py` holds the two DSPy signatures and the `InstructionProposalModule`. `proposer.py` holds the `SkilledProposer` class plus rendering and length helpers. The reference code at `reference_code.py` is the source of truth for ported behavior and is deleted at the end.

**Tech Stack:** Python >=3.10, dspy>=3.0, hatchling build backend, uv for env management, pytest with `dspy.utils.dummies.DummyLM` (no API calls), GitHub Actions.

## Global Constraints

- PyPI name `skilled-proposer`, import name `skilled_proposer`, version `0.1.0`.
- `requires-python = ">=3.10"`. Dependencies are exactly `["dspy>=3.0"]`. Dev dependency is pytest only.
- License is MIT.
- No PyYAML dependency. Frontmatter parsing is written by hand.
- litellm is never imported at module top level. It stays an optional import inside the token counter.
- Tests must make no network or API calls. Use `from dspy.utils.dummies import DummyLM`.
- The `skills/prompt-engineering/` directory is a repo-only example and test fixture. It is not packaged into the wheel.
- Run all commands from the repo root `/Users/dbreunig/Development/cmpnd/skilled_proposer`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Repo scaffolding and packaging metadata

**Files:**
- Create: `.gitignore`, `LICENSE`, `pyproject.toml`, `src/skilled_proposer/__init__.py`
- Move: `prompt-engineering/` → `skills/prompt-engineering/` (git mv is not possible, the directory is untracked; use plain `mv`)
- Delete: `skills/prompt-engineering/.DS_Store` (after the move)

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package skeleton. `import skilled_proposer` works and `skilled_proposer.__version__ == "0.1.0"`. Later tasks add modules under `src/skilled_proposer/` and tests under `tests/`.

- [ ] **Step 1: Move the example skill and remove .DS_Store**

```bash
mkdir -p skills
mv prompt-engineering skills/prompt-engineering
rm -f skills/prompt-engineering/.DS_Store
```

- [ ] **Step 2: Write .gitignore**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
.pytest_cache/
.ruff_cache/
.DS_Store
```

- [ ] **Step 3: Write LICENSE**

Create `LICENSE` with the standard MIT text:

```
MIT License

Copyright (c) 2026 Drew Breunig

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Write pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "skilled-proposer"
version = "0.1.0"
description = "A GEPA instruction proposer for DSPy that writes generalizable, skill-informed instructions"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.10"
authors = [{ name = "Drew Breunig", email = "dbreunig@gmail.com" }]
dependencies = ["dspy>=3.0"]
keywords = ["dspy", "gepa", "prompt-optimization", "llm"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]

[project.urls]
Homepage = "https://github.com/cmpnd-ai/skilled-proposer"
Repository = "https://github.com/cmpnd-ai/skilled-proposer"

[tool.uv]
dev-dependencies = ["pytest>=8.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/skilled_proposer"]
```

- [ ] **Step 5: Write a stub README**

`uv sync` builds an editable install, and hatchling fails if the declared `readme = "README.md"` file is missing. Create a one-line stub now. Task 6 replaces it with the full README.

Create `README.md`:

```markdown
# skilled-proposer

A GEPA instruction proposer for DSPy that writes generalizable, skill-informed instructions. Full README lands with v0.1.0.
```

- [ ] **Step 6: Write the package __init__**

Create `src/skilled_proposer/__init__.py`:

```python
"""skilled-proposer: a GEPA instruction proposer for DSPy."""

__version__ = "0.1.0"
```

(Public exports are added in Task 6, once the modules exist.)

- [ ] **Step 7: Sync the environment and verify the import**

```bash
uv sync
uv run python -c "import skilled_proposer; print(skilled_proposer.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add .gitignore LICENSE pyproject.toml README.md src/ skills/ uv.lock
git commit -m "Scaffold skilled-proposer package skeleton

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: skill.py — Skill dataclass with frontmatter parsing

**Files:**
- Create: `src/skilled_proposer/skill.py`
- Test: `tests/test_skill.py`

**Interfaces:**
- Consumes: nothing from other tasks. The reference behavior is `Skill` in `reference_code.py:79-119`.
- Produces: `Skill` frozen dataclass with fields `name: str`, `content: str`, `description: str | None = None`, and classmethod `Skill.load(source: Skill | str | Path) -> Skill`. Task 4 imports it as `from skilled_proposer.skill import Skill`.

Behavior contract for `Skill.load`:
- A `Skill` instance passes through unchanged.
- A path to a directory reads `SKILL.md` inside it, or raises `FileNotFoundError` if missing. The fallback name is the directory name.
- A path to a file reads the file. The fallback name is the file stem.
- Any other string is inline content. An empty inline string raises `ValueError`. The fallback name is the first non-empty line, stripped of `#` and truncated to 60 chars.
- After loading text from any source, a leading YAML frontmatter block is parsed. A frontmatter block is a first line of exactly `---`, then flat `key: value` lines, then a closing `---` line. `name` and `description` from the block override the fallbacks. The block is stripped from `content`. Indented lines inside the block are ignored (they belong to nested YAML, which we do not parse). If there is no closing `---`, the text is treated as having no frontmatter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skill.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_skill.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'skilled_proposer.skill'`

- [ ] **Step 3: Write the implementation**

Create `src/skilled_proposer/skill.py`:

```python
"""Skill loading: named blocks of reference material for the reflection LM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_skill.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skilled_proposer/skill.py tests/test_skill.py
git commit -m "Add Skill loader with frontmatter parsing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: signatures.py — reflection signatures and proposal module

**Files:**
- Create: `src/skilled_proposer/signatures.py`
- Test: `tests/test_signatures.py`

**Interfaces:**
- Consumes: nothing from other tasks. The reference behavior is `reference_code.py:126-232` and ports verbatim.
- Produces: `ProposeGeneralizableInstruction` (dspy.Signature with input fields `current_instruction`, `examples_with_feedback`, `reference_skills`, `additional_guidance`, `length_limit` and output field `new_instruction`), `CompressInstruction` (inputs `instruction`, `length_limit`; output `shortened_instruction`), and `InstructionProposalModule(base_instructions: str | None = None)` with predictors `self.propose` and `self.compress` and a keyword-only `forward`. Task 4 imports `InstructionProposalModule` from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signatures.py`:

```python
from skilled_proposer.signatures import (
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
)


def test_propose_signature_fields():
    assert set(ProposeGeneralizableInstruction.input_fields) == {
        "current_instruction",
        "examples_with_feedback",
        "reference_skills",
        "additional_guidance",
        "length_limit",
    }
    assert set(ProposeGeneralizableInstruction.output_fields) == {"new_instruction"}


def test_compress_signature_fields():
    assert set(CompressInstruction.input_fields) == {"instruction", "length_limit"}
    assert set(CompressInstruction.output_fields) == {"shortened_instruction"}


def test_module_default_meta_prompt_has_anti_overfitting_rules():
    m = InstructionProposalModule()
    assert "do not overfit" in m.propose.signature.instructions.lower()


def test_module_base_instructions_override():
    m = InstructionProposalModule("Custom meta prompt.")
    assert m.propose.signature.instructions == "Custom meta prompt."
    # Fields are unchanged by the override.
    assert set(m.propose.signature.output_fields) == {"new_instruction"}


def test_module_named_predictors():
    names = {name for name, _ in InstructionProposalModule().named_predictors()}
    assert names == {"propose", "compress"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signatures.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'skilled_proposer.signatures'`

- [ ] **Step 3: Write the implementation**

Create `src/skilled_proposer/signatures.py`. This is a verbatim port of `reference_code.py:126-232` with an import header and a module docstring:

```python
"""Reflection signatures and the dspy.Module that houses them."""

from __future__ import annotations

import dspy


class ProposeGeneralizableInstruction(dspy.Signature):
    """You are improving the instruction given to an AI assistant that performs
    one component of a larger task pipeline. You are shown the current
    instruction, plus examples of the assistant's inputs, outputs, and
    evaluator feedback. Write a new, better instruction.

    ## How to use the examples
    The examples are evidence of *weaknesses in the instruction*, not content
    for the instruction. From them, extract only things that transfer to
    unseen inputs:
    - The task's input format and what a correct output looks like (described
      abstractly, e.g. "answer with a single ISO-8601 date", never via a
      specific answer from an example).
    - Generalizable strategies and decision rules the assistant should follow.
    - Recurring failure modes, and guidance that prevents each one.
    - Genuinely domain-general knowledge (conventions, definitions, procedures
      that apply to the whole task domain).

    ## Hard constraints — do not overfit
    - NEVER copy example-specific content into the instruction: no verbatim
      inputs or outputs, no answers, no named entities, quantities, dates, or
      facts that belong to individual examples.
    - Do not enumerate the examples or reference them ("as in Example 2").
    - Litmus test: every sentence of the new instruction must be equally
      useful on inputs you have never seen. If a sentence would only help
      when a particular training example reappears, delete it.
    - Prefer a small number of high-leverage rules over exhaustive case lists.
    - Preserve whatever the current instruction already does well.

    ## Reference material
    If reference skills are provided, treat them as authoritative guidance on
    how to write an effective instruction for this assistant and task domain;
    apply what is relevant. Follow any additional guidance from the user.
    Obey the length limit exactly if one is given.

    Output only the new instruction text, ready to be used verbatim.
    """

    current_instruction: str = dspy.InputField(
        desc="The instruction currently given to the assistant."
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Inputs, assistant outputs, and evaluator feedback. Evidence of "
        "weaknesses only — never a source of content to copy."
    )
    reference_skills: str = dspy.InputField(
        desc="Reference material (skills) to inform the instruction. May be 'None'."
    )
    additional_guidance: str = dspy.InputField(
        desc="Extra requirements from the user for the new instruction. May be 'None'."
    )
    length_limit: str = dspy.InputField(
        desc="Length limit for the new instruction, or 'None'."
    )
    new_instruction: str = dspy.OutputField(
        desc="The improved, generalizable instruction. Instruction text only."
    )


class CompressInstruction(dspy.Signature):
    """Shorten the instruction to fit within the stated limit. Preserve every
    strategy, rule, and constraint; cut redundancy and verbosity. Do not add
    new content. Output only the shortened instruction."""

    instruction: str = dspy.InputField()
    length_limit: str = dspy.InputField()
    shortened_instruction: str = dspy.OutputField()


class InstructionProposalModule(dspy.Module):
    """dspy.Module housing the proposal and compression predictors.

    Mirrors DSPy's own custom-proposer pattern (see
    SingleComponentMultiModalProposer in dspy/teleprompt/gepa/
    instruction_proposal.py): keeping the predictors on a Module makes them
    discoverable via named_predictors(), lets their state be saved/loaded,
    and routes calls through the standard Module path (callbacks, history).
    """

    def __init__(self, base_instructions: str | None = None):
        super().__init__()
        signature = ProposeGeneralizableInstruction
        if base_instructions:
            signature = signature.with_instructions(base_instructions)
        self.propose = dspy.Predict(signature)
        self.compress = dspy.Predict(CompressInstruction)

    def forward(
        self,
        *,
        current_instruction: str,
        examples_with_feedback: str,
        reference_skills: str,
        additional_guidance: str,
        length_limit: str,
    ) -> dspy.Prediction:
        return self.propose(
            current_instruction=current_instruction,
            examples_with_feedback=examples_with_feedback,
            reference_skills=reference_skills,
            additional_guidance=additional_guidance,
            length_limit=length_limit,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signatures.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skilled_proposer/signatures.py tests/test_signatures.py
git commit -m "Add reflection signatures and proposal module

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: proposer.py — SkilledProposer with on_error

**Files:**
- Create: `src/skilled_proposer/proposer.py`
- Test: `tests/test_proposer.py`

**Interfaces:**
- Consumes: `from skilled_proposer.skill import Skill` (Task 2) and `from skilled_proposer.signatures import InstructionProposalModule` (Task 3). The reference behavior is `reference_code.py:239-463`.
- Produces: `SkilledProposer` with the init signature shown below, callable per GEPA's ProposalFn protocol, plus module-level helpers `_render_examples(examples) -> str`, `_count_words(text) -> int`, `_count_tokens(text, model) -> int`. Task 5 tests length enforcement on this same class. Task 6 exports `SkilledProposer` from the package root.

New init signature (two changes from the reference: `on_error` is added, and skills rendering includes descriptions):

```python
SkilledProposer(
    skills=None, additional_instructions=None, base_instructions=None,
    max_tokens=None, max_words=None, prompt_model=None, max_examples=None,
    on_error="keep",
)
```

`on_error="keep"` logs a per-component proposal failure and keeps the current instruction. `on_error="raise"` re-raises. Any other value raises `ValueError` at init.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_proposer.py`:

```python
import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer.proposer import SkilledProposer, _render_examples
from skilled_proposer.skill import Skill


def test_init_validation():
    with pytest.raises(ValueError):
        SkilledProposer(max_tokens=0)
    with pytest.raises(ValueError):
        SkilledProposer(max_words=-1)
    with pytest.raises(ValueError):
        SkilledProposer(on_error="explode")


def test_call_proposes_for_each_component():
    lm = DummyLM([{"new_instruction": "New A."}, {"new_instruction": "New B."}])
    proposer = SkilledProposer(prompt_model=lm)
    out = proposer(
        candidate={"a": "old a", "b": "old b"},
        reflective_dataset={"a": [{"Inputs": "x", "Feedback": "wrong"}], "b": []},
        components_to_update=["a", "b"],
    )
    assert out == {"a": "New A.", "b": "New B."}


def test_on_error_keep_returns_current_instruction():
    proposer = SkilledProposer()

    def boom(**kwargs):
        raise RuntimeError("boom")

    proposer.module = boom
    out = proposer(
        candidate={"a": "old a"},
        reflective_dataset={"a": []},
        components_to_update=["a"],
    )
    assert out == {"a": "old a"}


def test_on_error_raise_propagates():
    proposer = SkilledProposer(on_error="raise")

    def boom(**kwargs):
        raise RuntimeError("boom")

    proposer.module = boom
    with pytest.raises(RuntimeError, match="boom"):
        proposer(
            candidate={"a": "old a"},
            reflective_dataset={"a": []},
            components_to_update=["a"],
        )


def test_render_skills_includes_name_and_description():
    proposer = SkilledProposer(
        skills=[Skill(name="guide", content="Be terse.", description="A guide.")]
    )
    rendered = proposer._render_skills()
    assert "<skill name='guide' description='A guide.'>" in rendered
    assert "Be terse." in rendered


def test_render_skills_without_description():
    proposer = SkilledProposer(skills=["# Inline\nBody."])
    rendered = proposer._render_skills()
    assert "<skill name='Inline'>" in rendered


def test_render_skills_none():
    assert SkilledProposer()._render_skills() == "None"


def test_max_examples_caps_rendered_examples():
    captured = {}

    class Result:
        new_instruction = "ok"

    def capture(**kwargs):
        captured.update(kwargs)
        return Result()

    proposer = SkilledProposer(max_examples=1)
    proposer.module = capture
    proposer(
        candidate={"a": "old"},
        reflective_dataset={"a": [{"Inputs": "one"}, {"Inputs": "two"}]},
        components_to_update=["a"],
    )
    assert "# Example 1" in captured["examples_with_feedback"]
    assert "# Example 2" not in captured["examples_with_feedback"]


def test_render_examples_empty():
    assert _render_examples([]) == "No examples were provided."


def test_render_examples_nested():
    out = _render_examples(
        [{"Inputs": {"question": "q1"}, "Feedback": "too vague"}]
    )
    assert "# Example 1" in out
    assert "## Inputs" in out
    assert "### question" in out
    assert "q1" in out
    assert "too vague" in out


def test_render_examples_list_items():
    out = _render_examples([{"Outputs": ["first", "second"]}])
    assert "### Item 1" in out
    assert "### Item 2" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proposer.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'skilled_proposer.proposer'`

- [ ] **Step 3: Write the implementation**

Create `src/skilled_proposer/proposer.py`. This ports `reference_code.py:239-463` with three edits: the `on_error` parameter (init + `__call__`), the description-aware `<skill>` tag in `_render_skills`, and imports from the new modules. Full file:

```python
"""SkilledProposer: a custom GEPA instruction proposer for DSPy.

Implements GEPA's `ProposalFn` protocol (gepa.core.adapter.ProposalFn), so it
plugs directly into `dspy.GEPA(instruction_proposer=...)`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import dspy

from skilled_proposer.signatures import InstructionProposalModule
from skilled_proposer.skill import Skill

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


class SkilledProposer:
    """GEPA `ProposalFn` that proposes generalizable instructions, optionally
    informed by skills, extra guidance, and a length budget.

    Args:
        skills: Iterable of `Skill` objects, paths to SKILL.md files or skill
            directories, or inline strings of skill content.
        additional_instructions: Ad-hoc guidance for the reflection LM, applied
            to every proposal.
        base_instructions: Optional replacement for the built-in meta-prompt
            (the docstring of `ProposeGeneralizableInstruction`). The
            input/output fields are unchanged; only the prompt text is
            swapped, via `Signature.with_instructions`. Note this replaces
            the anti-overfitting contract too — if you still want it,
            include equivalent rules in your replacement text.
        max_tokens: Optional cap on the proposed instruction length in tokens
            (counted with litellm's tokenizer when available, else ~4 chars
            per token).
        max_words: Optional cap on the proposed instruction length in words.
        prompt_model: Optional dspy.LM to run proposals with. Not needed under
            `dspy.GEPA`, which already wraps calls in the reflection LM's
            context; useful with the standalone `gepa` package.
        max_examples: Cap on reflective examples rendered per component, to
            keep the meta-prompt bounded. None = no cap.
        on_error: What to do when a proposal fails for one component.
            "keep" (default) logs the error and keeps the current
            instruction, so long GEPA runs survive flaky proposals.
            "raise" propagates the error, so failures surface during
            development.
    """

    def __init__(
        self,
        skills: Sequence[Skill | str | Path] | None = None,
        additional_instructions: str | None = None,
        base_instructions: str | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        prompt_model: "dspy.LM | None" = None,
        max_examples: int | None = None,
        on_error: str = "keep",
    ):
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_words is not None and max_words <= 0:
            raise ValueError("max_words must be positive")
        if on_error not in ("keep", "raise"):
            raise ValueError('on_error must be "keep" or "raise"')

        self.skills = [Skill.load(s) for s in (skills or [])]
        self.additional_instructions = (additional_instructions or "").strip()
        self.base_instructions = (base_instructions or "").strip() or None
        self.max_tokens = max_tokens
        self.max_words = max_words
        self.prompt_model = prompt_model
        self.max_examples = max_examples
        self.on_error = on_error

        # All LM-facing predictors live on a dspy.Module.
        self.module = InstructionProposalModule(self.base_instructions)

    # -- ProposalFn ---------------------------------------------------------

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for name in components_to_update:
            current = candidate[name]
            examples = list(reflective_dataset.get(name, []))
            if self.max_examples is not None:
                examples = examples[: self.max_examples]
            try:
                results[name] = self._propose_one(current, examples)
            except Exception:
                if self.on_error == "raise":
                    raise
                logger.exception(
                    "SkilledProposer failed for component %r; keeping current text.",
                    name,
                )
                results[name] = current
        return results

    # -- Internals ----------------------------------------------------------

    def _propose_one(
        self, current_instruction: str, examples: Sequence[Mapping[str, Any]]
    ) -> str:
        kwargs = dict(
            current_instruction=current_instruction,
            examples_with_feedback=_render_examples(examples),
            reference_skills=self._render_skills(),
            additional_guidance=self.additional_instructions or "None",
            length_limit=self._length_limit_text(),
        )
        if self.prompt_model is not None:
            with dspy.context(lm=self.prompt_model):
                new_text = self.module(**kwargs).new_instruction
        else:
            new_text = self.module(**kwargs).new_instruction
        return self._enforce_length(new_text.strip())

    def _render_skills(self) -> str:
        if not self.skills:
            return "None"
        parts = []
        for skill in self.skills:
            attrs = f"name={skill.name!r}"
            if skill.description:
                attrs += f" description={skill.description!r}"
            parts.append(f"<skill {attrs}>\n{skill.content.strip()}\n</skill>")
        return "\n\n".join(parts)

    # -- Length budget ------------------------------------------------------

    def _length_limit_text(self) -> str:
        parts = []
        if self.max_words is not None:
            parts.append(f"at most {self.max_words} words")
        if self.max_tokens is not None:
            parts.append(f"at most {self.max_tokens} tokens")
        return (
            "The new instruction must be " + " and ".join(parts) + "."
            if parts
            else "None"
        )

    def _enforce_length(self, text: str) -> str:
        if self._within_budget(text):
            return text
        # One compression attempt with the LM.
        try:
            kwargs = dict(instruction=text, length_limit=self._length_limit_text())
            if self.prompt_model is not None:
                with dspy.context(lm=self.prompt_model):
                    shortened = self.module.compress(**kwargs).shortened_instruction.strip()
            else:
                shortened = self.module.compress(**kwargs).shortened_instruction.strip()
            if shortened and self._within_budget(shortened):
                return shortened
            text = shortened or text
        except Exception:
            logger.exception("Compression pass failed; falling back to truncation.")
        return self._truncate(text)

    def _within_budget(self, text: str) -> bool:
        if self.max_words is not None and _count_words(text) > self.max_words:
            return False
        if self.max_tokens is not None and _count_tokens(text, self._model_name()) > self.max_tokens:
            return False
        return True

    def _truncate(self, text: str) -> str:
        words = _WORD_RE.findall(text)
        limit = len(words)
        if self.max_words is not None:
            limit = min(limit, self.max_words)
        if self.max_tokens is not None:
            model = self._model_name()
            # Binary search the largest word-prefix that fits the token budget.
            lo, hi = 0, limit
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _count_tokens(" ".join(words[:mid]), model) <= self.max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            limit = lo
        truncated = " ".join(words[:limit])
        logger.warning(
            "Proposed instruction exceeded the length budget even after "
            "compression; hard-truncated to %d words.", limit,
        )
        return truncated

    def _model_name(self) -> str | None:
        lm = self.prompt_model or getattr(dspy.settings, "lm", None)
        return getattr(lm, "model", None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_tokens(text: str, model: str | None) -> int:
    if model:
        try:
            import litellm

            return litellm.token_counter(model=model, text=text)
        except Exception:
            pass
    return max(1, round(len(text) / 4))  # rough fallback: ~4 chars/token


def _render_examples(examples: Sequence[Mapping[str, Any]]) -> str:
    """Render GEPA's reflective dataset entries as markdown.

    Entries typically look like {"Inputs": ..., "Generated Outputs": ...,
    "Feedback": ...}, but arbitrary keys and nesting are handled.

    TODO(v0.2): rich types (e.g. dspy.Image) are stringified here. For
    vision programs, pass them through as multimodal content instead — see
    DSPy's MultiModalInstructionProposer for the pattern.
    """
    if not examples:
        return "No examples were provided."

    def render_value(value: Any, level: int = 3) -> str:
        if isinstance(value, Mapping):
            out = ""
            for k, v in value.items():
                out += f"{'#' * level} {k}\n{render_value(v, min(level + 1, 6))}"
            return out or "\n"
        if isinstance(value, (list, tuple)):
            out = ""
            for i, item in enumerate(value, 1):
                out += f"{'#' * level} Item {i}\n{render_value(item, min(level + 1, 6))}"
            return out or "\n"
        return f"{str(value).strip()}\n\n"

    blocks = []
    for i, example in enumerate(examples, 1):
        block = f"# Example {i}\n"
        for key, value in example.items():
            block += f"## {key}\n{render_value(value)}"
        blocks.append(block)
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_proposer.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skilled_proposer/proposer.py tests/test_proposer.py
git commit -m "Add SkilledProposer with configurable on_error

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Length-budget tests

**Files:**
- Test: `tests/test_length.py`

**Interfaces:**
- Consumes: `SkilledProposer` from Task 4 (already implemented; this task adds test coverage for its length machinery). `DummyLM` returns queued dicts in call order: the propose call consumes the first dict (key `new_instruction`), the compress call consumes the second (key `shortened_instruction`).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the tests**

Create `tests/test_length.py`:

```python
from dspy.utils.dummies import DummyLM

from skilled_proposer.proposer import SkilledProposer, _count_tokens, _count_words


def _propose(proposer):
    return proposer(
        candidate={"a": "old"},
        reflective_dataset={"a": []},
        components_to_update=["a"],
    )["a"]


def test_no_limit_passes_through():
    lm = DummyLM([{"new_instruction": "A long enough instruction."}])
    assert _propose(SkilledProposer(prompt_model=lm)) == "A long enough instruction."


def test_within_word_budget_no_compression():
    lm = DummyLM([{"new_instruction": "Three word answer."}])
    assert _propose(SkilledProposer(prompt_model=lm, max_words=5)) == "Three word answer."


def test_compression_pass_used_when_over_budget():
    long_text = " ".join(["word"] * 20)
    lm = DummyLM(
        [
            {"new_instruction": long_text},
            {"shortened_instruction": "Short now."},
        ]
    )
    assert _propose(SkilledProposer(prompt_model=lm, max_words=5)) == "Short now."


def test_truncation_when_compression_still_over_budget():
    long_text = " ".join(f"w{i}" for i in range(20))
    still_long = " ".join(f"s{i}" for i in range(10))
    lm = DummyLM(
        [
            {"new_instruction": long_text},
            {"shortened_instruction": still_long},
        ]
    )
    result = _propose(SkilledProposer(prompt_model=lm, max_words=5))
    assert result == "s0 s1 s2 s3 s4"


def test_length_limit_text():
    assert SkilledProposer()._length_limit_text() == "None"
    assert (
        SkilledProposer(max_words=10)._length_limit_text()
        == "The new instruction must be at most 10 words."
    )
    assert (
        SkilledProposer(max_words=10, max_tokens=40)._length_limit_text()
        == "The new instruction must be at most 10 words and at most 40 tokens."
    )


def test_truncate_by_tokens_fallback_counter():
    # With no model name, _count_tokens is ~len/4. The truncated prefix
    # must fit the token budget; the full text must not.
    proposer = SkilledProposer(max_tokens=10)
    text = " ".join(["abcde"] * 40)
    truncated = proposer._truncate(text)
    assert _count_tokens(truncated, None) <= 10
    assert _count_words(truncated) < 40
    assert text.startswith(truncated)


def test_count_words():
    assert _count_words("one two  three\nfour") == 4
    assert _count_words("") == 0


def test_count_tokens_fallback():
    assert _count_tokens("x" * 40, None) == 10
    assert _count_tokens("", None) == 1
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_length.py -v`
Expected: all 8 tests PASS. If `test_truncation_when_compression_still_over_budget` fails because DummyLM raises when its queue is exhausted, the failure is in the test setup, not the code. Check `uv run python -c "import inspect; from dspy.utils.dummies import DummyLM; print(inspect.signature(DummyLM.__init__))"` and adjust the queued responses so exactly two LM calls are made.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests across the four test files PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_length.py
git commit -m "Add length-budget test coverage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Public API, README, delete reference code

**Files:**
- Modify: `src/skilled_proposer/__init__.py`
- Create: `README.md`
- Delete: `reference_code.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: `SkilledProposer` (Task 4), `Skill` (Task 2), signatures (Task 3).
- Produces: `from skilled_proposer import SkilledProposer, Skill` as the documented public API.

- [ ] **Step 1: Write the failing test**

Create `tests/test_public_api.py`:

```python
import skilled_proposer


def test_public_exports():
    assert skilled_proposer.__version__ == "0.1.0"
    assert skilled_proposer.SkilledProposer is not None
    assert skilled_proposer.Skill is not None
    assert set(skilled_proposer.__all__) == {
        "SkilledProposer",
        "Skill",
        "InstructionProposalModule",
        "ProposeGeneralizableInstruction",
        "CompressInstruction",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: FAIL with `AttributeError: module 'skilled_proposer' has no attribute 'SkilledProposer'`

- [ ] **Step 3: Update the package __init__**

Replace the contents of `src/skilled_proposer/__init__.py` with:

```python
"""skilled-proposer: a GEPA instruction proposer for DSPy."""

from skilled_proposer.proposer import SkilledProposer
from skilled_proposer.signatures import (
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
)
from skilled_proposer.skill import Skill

__version__ = "0.1.0"

__all__ = [
    "SkilledProposer",
    "Skill",
    "InstructionProposalModule",
    "ProposeGeneralizableInstruction",
    "CompressInstruction",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: PASS

- [ ] **Step 5: Write README.md**

Replace the Task 1 stub `README.md` with:

````markdown
# skilled-proposer

A custom instruction proposer for [GEPA](https://dspy.ai/api/optimizers/GEPA/), the reflective prompt optimizer in [DSPy](https://dspy.ai). Drop it into `dspy.GEPA(instruction_proposer=...)` to get instructions that generalize instead of memorizing your training set, informed by reference skills you provide.

## Why

GEPA improves a program by asking a reflection model to rewrite each component's instruction based on execution traces and evaluator feedback. The stock proposer tells the reflection model to include "niche and domain specific factual information" from those traces in the new instruction. That helps on benchmarks, but it copies entities, numbers, and answers from your training examples into the prompt, and the prompt then does worse on inputs it has never seen.

`SkilledProposer` uses a different meta-prompt. It treats the examples as evidence of weaknesses in the instruction, extracts strategies and decision rules that transfer, and forbids copying example-specific content. It also adds three practical controls:

- Skills. Pass SKILL.md files, skill directories, or inline strings. The reflection model gets them as reference material, e.g., a prompting guide for your student model.
- Extra guidance. A plain string applied to every proposal.
- Length budgets. Cap the proposed instruction by words or tokens. The cap is enforced by a prompt constraint, then a compression pass, then truncation.

## Install

```bash
pip install skilled-proposer
```

Requires Python 3.10 or newer and dspy 3.0 or newer.

## Quickstart

```python
import dspy
from skilled_proposer import SkilledProposer

proposer = SkilledProposer(
    skills=["./skills/prompt-engineering"],
    additional_instructions="Write instructions in imperative voice.",
    max_words=300,
)

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
    instruction_proposer=proposer,
    auto="medium",
)

optimized = optimizer.compile(program, trainset=train, valset=val)
```

## Skills

A skill is reference material for the reflection model. Each entry in `skills` can be:

- a path to a directory that contains a `SKILL.md` (the [Agent Skills](https://agentskills.io) layout)
- a path to a markdown file
- an inline string
- a `Skill(name=..., content=..., description=...)` object

If the file starts with YAML frontmatter, `name` and `description` are read from it and the block is stripped from the content. This repo ships an example at [`skills/prompt-engineering`](skills/prompt-engineering), a prompt optimization guide the reflection model can apply when rewriting instructions. Only the `SKILL.md` file is read in v0.1. Files under `references/` are ignored.

## Options

```python
SkilledProposer(
    skills=None,                   # skills, paths, or inline strings
    additional_instructions=None,  # guidance applied to every proposal
    base_instructions=None,        # replace the built-in meta-prompt
    max_words=None,                # word cap on proposed instructions
    max_tokens=None,               # token cap on proposed instructions
    prompt_model=None,             # LM for the standalone gepa package
    max_examples=None,             # cap reflective examples per component
    on_error="keep",               # "keep" or "raise"
)
```

- `base_instructions` replaces the whole meta-prompt, including the anti-overfitting rules. If you still want those rules, include equivalent text in your replacement.
- `on_error="keep"` logs a failed proposal and keeps the current instruction, so a long GEPA run survives a flaky call. Use `on_error="raise"` during development so failures surface.
- `max_tokens` counts tokens with litellm's tokenizer when it can resolve your model name, and falls back to about 4 characters per token.

## Using the standalone gepa package

`dspy.GEPA` runs the proposer inside the reflection model's context, so you do not pass a model. The standalone [gepa](https://github.com/gepa-ai/gepa) package does not set a DSPy context, so pass the model yourself:

```python
proposer = SkilledProposer(
    skills=["./skills/prompt-engineering"],
    prompt_model=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
)
```

Then pass `proposer` wherever gepa accepts a `ProposalFn`.

## Limits

- v0.1 is text only. Rich values such as `dspy.Image` are stringified in the reflective examples, so the reflection model cannot see them. Multimodal support is planned for v0.2.

## License

MIT
````

- [ ] **Step 6: Delete the reference code and verify the build**

```bash
rm reference_code.py
uv run pytest -v
uv build
```

Expected: all tests PASS, and `uv build` produces `dist/skilled_proposer-0.1.0.tar.gz` and `dist/skilled_proposer-0.1.0-py3-none-any.whl`.

Check the wheel does not contain the example skill:

```bash
unzip -l dist/skilled_proposer-0.1.0-py3-none-any.whl
```

Expected: only `skilled_proposer/*` files and dist-info. No `skills/` entries.

- [ ] **Step 7: Commit**

```bash
git add src/skilled_proposer/__init__.py README.md tests/test_public_api.py
git rm --cached reference_code.py 2>/dev/null; git add -A
git commit -m "Add public API and README; remove reference code

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(`reference_code.py` was never committed, so `rm` plus `git add -A` is enough. The `git rm --cached` is a no-op guard.)

---

### Task 7: CI workflow, remote, push

**Files:**
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: the full test suite and `pyproject.toml` from earlier tasks.
- Produces: CI on push and pull request, and `origin` pointing at `https://github.com/cmpnd-ai/skilled-proposer` with `main` pushed.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/test.yml`:

```yaml
name: test

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync
      - run: uv run pytest -v
```

- [ ] **Step 2: Verify the suite passes locally one more time**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "Add GitHub Actions test workflow

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Add the remote and push**

```bash
git remote add origin https://github.com/cmpnd-ai/skilled-proposer
gh repo view cmpnd-ai/skilled-proposer >/dev/null 2>&1 || gh repo create cmpnd-ai/skilled-proposer --public --description "A GEPA instruction proposer for DSPy that writes generalizable, skill-informed instructions"
git push -u origin main
```

Expected: push succeeds and `git branch -vv` shows `main` tracking `origin/main`.

- [ ] **Step 5: Confirm CI**

```bash
gh run watch --repo cmpnd-ai/skilled-proposer --exit-status || gh run list --repo cmpnd-ai/skilled-proposer --limit 1
```

Expected: the `test` workflow runs on all four Python versions and passes. If `gh run watch` cannot find a run yet, wait briefly and use `gh run list` to check status.
