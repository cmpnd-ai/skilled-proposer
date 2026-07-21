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

If the file starts with YAML frontmatter, `name` and `description` are read from it and the block is stripped from the content. This repo ships an example at [`skills/prompt-engineering`](skills/prompt-engineering), a prompt optimization guide the reflection model can apply when rewriting instructions.

### Skills with subfolders

Loading a skill directory reads only its `SKILL.md`. Files in subfolders such as `models/` or `references/` are not loaded. This is deliberate. An agent browsing a skill can open those files when it needs them, but GEPA calls the proposer in a plain LM call with no filesystem, many times per run. What the reflection model should see is also known before the run starts, e.g., you know which student model you are optimizing. So you, the developer, pick the extra files and pass them alongside the parent skill:

```python
proposer = SkilledProposer(
    skills=[
        "./skills/prompt-engineering",                    # reads SKILL.md
        "./skills/prompt-engineering/models/openai.md",   # guidance for the student model
    ],
)
```

Each entry becomes its own `<skill>` block in the reflection prompt. Pass only the files that apply to your run. Inlining a whole skill folder would grow every proposal call for no benefit.

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
