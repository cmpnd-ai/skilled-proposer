# skilled-proposer v0.1.0 design

Date: 2026-07-20
Status: approved pending user review

## What this is

`skilled-proposer` is a Python package that provides a custom instruction
proposer for GEPA, the prompt optimizer in DSPy. The proposer is called
`SkilledProposer`. It implements GEPA's `ProposalFn` protocol, so a user can
pass it to `dspy.GEPA(instruction_proposer=...)`.

It differs from GEPA's stock proposer in four ways.

- The default meta-prompt asks the reflection model for transferable
  strategies and forbids copying facts from training examples into the
  instruction. The stock proposer does the opposite, which can leak training
  data into the prompt and hurt generalization.
- The user can pass skills, meaning SKILL.md files, skill directories, or
  inline strings. The proposer gives their content to the reflection model as
  reference material.
- The user can pass extra guidance as a plain string, applied to every
  proposal.
- The user can set a word or token limit on the proposed instruction. The
  proposer enforces the limit with a prompt constraint, then a compression
  pass, then truncation.

## Decisions from the design discussion

- The anti-overfitting meta-prompt is the default, not a locked contract.
  `base_instructions` stays a supported way to replace it. The README will
  state that replacing it also removes the anti-overfitting rules.
- DSPy is the primary host. The standalone `gepa` package is a secondary
  documented path through the `prompt_model` argument. Both appear in the
  README. dspy gets full test coverage and gepa gets a documented section.
- `Skill.load` parses YAML frontmatter. It reads `name` and `description`
  from a leading `---` block and strips the block from the content. Files in
  a skill's `references/` directory are still ignored in v0.1.
- A new `on_error` argument controls what happens when a proposal fails for
  one component. The value `"keep"` logs the error and keeps the current
  instruction. The value `"raise"` propagates the error. The default is
  `"keep"` because GEPA runs are long and expensive.
- v0.1 is text only. Values such as `dspy.Image` are stringified. The README
  states this limit, and multimodal support is planned for v0.2 following
  the pattern in DSPy's `MultiModalInstructionProposer`.
- The `prompt-engineering` skill directory already in the repo is the shipped
  example. Tests also load it as a fixture. It stays in the repo but is not
  packaged into the wheel.

## Package layout

PyPI name `skilled-proposer`, import name `skilled_proposer`, src layout.

```
src/skilled_proposer/
  __init__.py       exports SkilledProposer, Skill, and __version__
  skill.py          Skill dataclass and loader, frontmatter parsing
  signatures.py     ProposeGeneralizableInstruction, CompressInstruction,
                    InstructionProposalModule
  proposer.py       SkilledProposer, example rendering, length enforcement
skills/prompt-engineering/   example skill, moved from the repo root
tests/
.github/workflows/test.yml
pyproject.toml
README.md
LICENSE
.gitignore
```

`reference_code.py` is deleted once its content is ported into the package.

## Code changes from the reference code

1. Frontmatter parsing in `Skill.load`. When loaded content starts with a
   `---` line, the loader reads flat `key: value` lines up to the closing
   `---`. It uses `name` and `description` from that block and falls back to
   the directory or file name when they are absent. The parser is written by
   hand, so the package does not depend on PyYAML. The `Skill` dataclass
   gains an optional `description` field. When a description is present, the
   rendered `<skill>` tag includes it.
2. The `on_error` argument described above.
3. No other behavior changes. The rest of the reference code ships as is.

## Packaging

- Build backend is hatchling.
- Dependencies are `dspy>=3.0`. GEPA is part of dspy 3.x. litellm arrives
  through dspy, and the token counter keeps treating it as an optional
  import.
- `requires-python = ">=3.10"`, which matches the union type syntax already
  in the code.
- License is MIT.
- Version starts at 0.1.0.

## Tests and CI

Tests use pytest and `dspy.utils.DummyLM`, so they make no API calls. They
cover:

- skill loading from an inline string, a file, and a directory
- frontmatter parsing, using `skills/prompt-engineering` as a fixture
- error cases in skill loading, e.g., a directory without SKILL.md
- example rendering, including nested values and the empty case
- length enforcement, including the word cap, the token cap with the
  fallback counter, and the path where compression fails and the text is
  truncated
- both `on_error` modes
- the `__call__` contract, meaning each component in `components_to_update`
  maps to a proposed instruction

CI is a GitHub Actions workflow that runs pytest on push and pull request
across Python 3.10 to 3.13, using uv.

## Repo formalization

- Add a `.gitignore` for Python plus `.DS_Store`. The skill directory
  currently contains a `.DS_Store` file.
- Make the initial commit.
- Add the remote `https://github.com/cmpnd-ai/skilled-proposer` as `origin`
  and push `main`.

## README outline

- Why this proposer exists, compared with the stock proposer.
- Quickstart with `dspy.GEPA`.
- A section on the standalone `gepa` package and `prompt_model`.
- The skill format, shown with the `prompt-engineering` example.
- Length budgets, `on_error`, and `base_instructions`, with the warning that
  replacing the meta-prompt removes the anti-overfitting rules.
- Limits, meaning text only in v0.1.
