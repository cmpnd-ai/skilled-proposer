# skilled-proposer

A custom instruction proposer for [GEPA](https://dspy.ai/api/optimizers/GEPA/), the reflective prompt optimizer in [DSPy](https://dspy.ai). Drop it into `dspy.GEPA(instruction_proposer=...)` to get instructions that generalize instead of memorizing your training set, informed by reference skills you provide.

## Why

GEPA improves a program by asking a reflection model to rewrite each component's instruction based on execution traces and evaluator feedback. The stock proposer tells the reflection model to include "niche and domain specific factual information" from those traces in the new instruction. This helps with some tasks, but it can copy entities, numbers, and answers from your training examples into the prompt, and the prompt might perform worse on inputs it has never seen.

`SkilledProposer` uses a different meta-prompt. It gives the reflection model a three-step procedure. First, infer the task from the examples, because the assistant will only ever see the instruction. Second, diagnose why each failure happened and find the general rule that would have prevented it. Third, write the replacement instruction from those rules. The prompt then states one principle against overfitting. The proposer also adds four practical controls:

- **Skills:** Pass SKILL.md files, skill directories, or inline strings. The reflection model gets them as reference material, e.g., a prompting guide for your student model.
- **Extra guidance:** A plain string applied to every proposal.
- **Length budgets:** Cap the proposed instruction by words or tokens. The cap is enforced by a prompt constraint, then a compression pass, then truncation.
- **A proposal journal:** The reflection model sees what it proposed earlier in the run, which of those proposals GEPA kept, and lessons distilled from that record.

## Install

```bash
pip install skilled-proposer
```

Requires Python 3.10 or newer and DSPy 3.3.1 or newer.

## Quickstart

```python
import dspy
from skilled_proposer import SkilledProposer

proposer = SkilledProposer(
    skills=[
        "./skills/prompt-engineering",                   # reads SKILL.md
        "./skills/prompt-engineering/models/openai.md",  # guidance for the student model
    ],
    additional_instructions="Write instructions in imperative voice.",
    max_words=300,
)

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.6-luna"),
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
    prompt_model=None,             # (standalone GEPA only)
    max_examples=None,             # cap reflective examples per component
    retries=1,                     # extra attempts per component on failure
    on_error="skip",               # "skip" or "raise"
    journal=False,                 # record proposals and their fate, see below
    journal_path=None,             # JSON file for the journal
    journal_entries=12,            # entries shown to the reflection model
    distill_every=5,               # entries between lesson distillations
)
```

- `base_instructions` replaces the whole meta-prompt, including the anti-overfitting rules. If you still want those rules, include equivalent text in your replacement.
- `retries` gives each component that many extra attempts when its proposal fails or comes back unusable. The default is one retry.
- `on_error="skip"` logs a component whose proposal still fails after retries and leaves it out of the returned dict. GEPA then keeps the parent's text for that component, and when no component survives it skips the proposal without spending minibatch evaluations on a child identical to its parent. `"keep"` is accepted as an alias. Use `on_error="raise"` during development so the first failure surfaces with no retries. Either way, LM/provider errors (`LMError`) always propagate, so a dead API key fails the run instead of silently keeping unchanged text for the whole run.
- `max_tokens` counts tokens with litellm's tokenizer when it can resolve your model name, and falls back to about 4 characters per token.

## Proposal journal

GEPA calls the proposer many times in one run, and each call sees only the current instruction and a fresh set of examples. The reflection model has no memory of what it proposed before or whether GEPA kept it, so it can circle back to an approach that already failed or drift away from one that worked. The journal gives it that memory.

```python
proposer = SkilledProposer(
    skills=["./skills/prompt-engineering"],
    journal=True,
    journal_path="runs/committee/journal.json",
)
```

With `journal=True` the proposer records every proposal it makes: the text, the reflection model's own one or two sentence summary of what it changed, and the size of the change. Every later call shows the reflection model the newest `journal_entries` entries under a `proposal_journal` field. An entry looks like this in the prompt:

```
### Iteration 4, component `extract`, accepted
Change: Added a rule to copy the committee name from the disclaimer and stop at the address.
Size: 231 words, 48 more than the parent.
Reason: Chosen as the parent in iteration 5.
```

Every `distill_every` closed entries, the proposer asks the reflection model to condense the whole journal into a short list of lessons about which kinds of changes this task rewards. The lessons lead every later prompt, above the entries. A failed distillation keeps the prior lessons. Set `distill_every=None` to turn distillation off.

Set `journal_path` to keep the journal in a JSON file. The proposer writes it after every proposal and loads it when the file exists, so a run resumed from GEPA's `log_dir` keeps its record. The file is also the easiest way to read what the reflection model tried and what it learned.

The journal is text only. It reads the instruction text GEPA hands back, so it works with any GEPA sampling strategy and with the standalone `gepa` package.

## Batch proposals

Experimental. `candidates=n` makes one reflection call produce n candidate instructions for a component. The candidates condition on each other, and the prompt asks each one to fix the diagnosed failures by a materially different route. The proposer returns the first candidate, keeps the rest, and serves them in order on the following calls that carry the same parent text. When the kept candidates run out, or the parent changes, the next call runs a fresh batch.

Batching pays off only when GEPA asks for several proposals from one parent in one iteration and keeps every one that improves. Pass these engine settings through `gepa_kwargs`, with the sampling strategy's n equal to `candidates`:

```python
from gepa.strategies.proposal_sampling import SameParentSampling
from gepa.strategies.proposal_selection import AllImprovements

proposer = SkilledProposer(skills=["./skills/prompt-engineering"], candidates=4)

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.6-luna"),
    instruction_proposer=proposer,
    reflection_minibatch_size=15,
    gepa_kwargs={
        "sampling_strategy": SameParentSampling(4),
        "selection_strategy": AllImprovements(),
    },
    auto="medium",
)
```

With the default sampling strategy GEPA evaluates one candidate per iteration and the rest are discarded when the parent changes. With `BestImprovement` GEPA keeps one candidate per iteration and discards the diversity the batch produced. Keep n small so the whole batch fits the reflection model's output limit.

The batch prompt is the standard meta-prompt plus one section that asks for distinct routes, so `base_instructions`, `additional_instructions`, skills, the journal, and the length budget all apply to every candidate.

## Recommended engine settings

These are GEPA engine settings, passed through `gepa_kwargs`, that pair well with this proposer:

- A larger `reflection_minibatch_size` gives the reflection model more failures to diagnose per call. DSPy's default is 3. Use 10 to 20 so each proposal sees enough of the task's variety.
- `acceptance_criterion=ImprovementOrEqualAcceptance()` from `gepa.strategies.acceptance` lets a proposal that ties its parent on the minibatch through to the valset, which helps when the minibatch is small.
- `candidate_selection_strategy="epsilon_greedy"` on `dspy.GEPA` explores parents off the Pareto front some of the time.

```python
from gepa.strategies.acceptance import ImprovementOrEqualAcceptance

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.6-luna"),
    instruction_proposer=proposer,
    reflection_minibatch_size=15,
    candidate_selection_strategy="epsilon_greedy",
    gepa_kwargs={"acceptance_criterion": ImprovementOrEqualAcceptance()},
    auto="medium",
)
```

## Using the standalone gepa package

`dspy.GEPA` runs the proposer inside the reflection model's context, so you do not pass a model. The standalone [gepa](https://github.com/gepa-ai/gepa) package does not set a DSPy context, so pass the model yourself:

```python
proposer = SkilledProposer(
    skills=[
        "./skills/prompt-engineering",
        "./skills/prompt-engineering/models/openai.md",
    ],
    prompt_model=dspy.LM("openai/gpt-5.6-luna"),
)
```

Then pass `proposer` wherever gepa accepts a `ProposalFn`.

## Using with Flex

dspy 3.3 added [`dspy.Flex`](https://dspy.ai/diving-deeper/flex/), a module that holds its whole implementation as Python source, which GEPA rewrites during optimization. GEPA sends Flex components to a built-in code proposer, and a custom `instruction_proposer` never sees them. The built-in prompt does not warn the reflection model against memorizing the training set, and with code the risk is worse than with instructions. The model can write a branch that matches one training input and returns its answer.

`SkilledCodeProposer` applies this package's approach to Flex source. The reflection model gets the same three step procedure, a rule against overfitting written for code, your reference skills, and your extra guidance. Every proposal is checked before it is used. It must parse and define a class with a `forward` method, or the current source is kept.

dspy has no `code_proposer` hook yet (coming soon!), so this package patches the built-in proposer for the duration of a `compile` call:

```python
import dspy
from skilled_proposer import SkilledCodeProposer, SkilledProposer, use_code_proposer

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.6-luna"),
    instruction_proposer=SkilledProposer(skills=["./skills/prompt-engineering"]),
    auto="medium",
)

code_proposer = SkilledCodeProposer(
    skills=["./skills/prompt-engineering"],
    additional_instructions="Prefer few predictors and plain Python.",
)

with use_code_proposer(code_proposer):
    optimized = optimizer.compile(program, trainset=train, valset=val)
```

Notes:

- dspy logs a warning that a custom `instruction_proposer` skips code components. Under the patch the warning is expected and harmless, because the patched proposer handles them.
- The patch is a bridge. We are proposing a `code_proposer` parameter for `dspy.GEPA`; once it lands, pass `SkilledCodeProposer` there and drop the patch.
- `SkilledCodeProposer` takes `skills`, `additional_instructions`, `base_instructions`, `prompt_model`, `max_examples`, `retries`, and `on_error`, with the same meanings as `SkilledProposer`. A proposal that does not parse, defines no class, or has no `forward` method counts as a failure. There is no length budget for code.

## Limits

- The proposer is text only. Rich values such as `dspy.Image` are stringified in the reflective examples, so the reflection model cannot see them. Multimodal support is planned for a later release.

## License

MIT
