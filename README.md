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
    max_words=1000,                # word cap on proposed instructions
    max_tokens=None,               # token cap on proposed instructions
    prompt_model=None,             # (standalone GEPA only)
    max_examples=None,             # cap reflective examples per component
    retries=1,                     # extra attempts per component on failure
    on_error="skip",               # "skip" or "raise"
    journal=False,                 # record proposals and their fate, see below
    journal_path=None,             # JSON file for the journal
    journal_entries=12,            # entries shown to the reflection model
    distill_every=5,               # entries between lesson distillations
    engine="predict",              # "predict", "rlm", or "auto", see below
    review="minibatch",            # "minibatch" or "seen" (RLM only)
    seen_path=None,                # JSON file for the seen store
    rlm_threshold=30_000,          # auto: prompt tokens that select RLM
    sub_lm=None,                   # LM for the RLM's llm_query calls
    max_iters=20,                  # RLM REPL iterations per proposal
    max_llm_calls=50,              # RLM sub-LM calls per proposal
    interpreter_factory=None,      # custom sandbox; None uses Deno
    seed=0,                        # order of seen records
    compaction=False,              # shorten agent histories, see below
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

## Agent programs

Experimental. An agent built on `dspy.ReAct`, `dspy.ReActV2`, or `dspy.RLM` collects a long history on each run. DSPy puts that history into the reflective examples, so most of the proposal prompt can be the text of pages the agent fetched. Set `compaction=True` to shorten the examples before the proposer renders them.

```python
from skilled_proposer import Compaction, SkilledProposer

proposer = SkilledProposer(compaction=True)
proposer = SkilledProposer(
    compaction=Compaction(observation_chars=1000, examples_token_budget=60_000)
)
```

With compaction on, the proposer renders each history as numbered steps.

- Thoughts, tool names, tool arguments, and code are kept in full.
- Each tool result keeps its first `observation_chars` characters, 500 by default. A marker says how many characters were cut.
- RLM outputs keep the head and tail view that the agent saw.
- Any other string field longer than `max_field_chars`, 2,000 by default, keeps its head.
- A long input that repeats across examples, e.g., the ReActV2 tool list, is written once.
- Feedback is never cut.

`examples_token_budget` limits the tokens in the rendered examples for one proposal call. The skills, the journal, and the current instruction are outside the budget. When the examples are over the budget, the proposer halves the caps, then drops examples from the end until they fit. The proposer logs what it kept and dropped.

For `dspy.ReAct` and `dspy.RLM`, DSPy picks one step of each run at random for the reflective example, and the example holds the history up to that step. The reflection model can therefore miss the step where the agent went wrong. Compaction cannot add steps that DSPy did not pass.

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

## RLM engine

With `engine="rlm"`, each proposal runs as a [`dspy.RLM`](https://dspy.ai/api/modules/RLM/). The reflection model gets the reflective records as data in a Python sandbox and analyzes them with code and sub-LM calls before it writes. It counts failures across the batch, labels their root causes, and reads the records that the counts point to. Long inputs and agent histories no longer have to fit in one prompt.

```bash
pip install "skilled-proposer[rlm]"   # adds Deno for dspy's sandbox
```

```python
proposer = SkilledProposer(
    skills=["./skills/prompt-engineering"],
    engine="rlm",
    review="seen",
    journal=True,
    journal_path="runs/committee/journal.json",
    seen_path="runs/committee/seen.json",
)

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.6-luna"),
    instruction_proposer=proposer,
    reflection_minibatch_size=40,
    auto="medium",
)
```

- **Engine choice.** `engine="auto"` uses RLM when `review="seen"`, or when the first proposal's prompt is at least `rlm_threshold` tokens, and Predict otherwise. The choice is made once and kept for the run.
- **Seen review.** `review="seen"` also shows the model every record GEPA gave the proposer earlier in the run. Each record is tagged `current`, `ancestor` (with how many edits back) or `other_branch`, depending on how the instruction that produced it relates to the one being improved, and the store keeps those instructions' text. The model chooses what to fix from this round's records and uses the older ones to measure how common a failure is. The store reaches the whole trainset only after one epoch, `train size / reflection_minibatch_size` iterations, so pair it with a large `reflection_minibatch_size`. The relation is tracked per component, so for a multi-predictor program a `current` record may come from a time when other components differed.
- **Journal.** With `journal=True`, the model reads the journal as data and writes back lessons, open hypotheses and its analysis of past entries each round. It cannot change the record of what was proposed or kept. The separate distillation step does not run under RLM.
- **Skills.** Directory skills expose their other files, such as `models/openai.md`, to the model as `skill_files`, so it can open them when a diagnosis calls for them.
- **Failures.** An RLM run that errors, runs out of `max_iters` or returns an empty instruction counts as a failure. `retries` runs the RLM again and then `on_error` applies. There is no fallback to the Predict engine.
- **No extra rollouts.** The RLM analyzes records GEPA already produced. It never runs the student, so metric budgets are unchanged.

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
