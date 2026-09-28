"""Reflection signatures and the dspy.Module that houses them."""

from __future__ import annotations

import dspy
import pydantic

from skilled_proposer.sandbox import SandboxJSON


class ProposeGeneralizableInstruction(dspy.Signature):
    """You are improving the instruction given to an AI assistant that performs
    a task. You are shown the current instruction, plus examples of the
    assistant's inputs, outputs, and evaluator feedback. Write a new, better
    instruction.

    ## Procedure

    1. Infer the task. From the inputs, outputs, and current instruction,
       work out what the task is: the input format, what a correct output
       looks like, and the conventions of the domain. The assistant will see
       only your instruction — never these examples — so the instruction
       must teach the task completely on its own.

    2. Diagnose the failures. For each example with negative feedback,
       determine why the assistant went wrong. Then find the general rule,
       strategy, or piece of domain knowledge that would have prevented the
       failure — one that helps on any input where the same mistake could
       recur, not just on this example.

    3. Write the replacement. State the task, then the strategies and
       decision rules the assistant should follow. Keep whatever the current
       instruction already does well, and fix what your diagnosis showed to
       be broken. Prefer a few high-leverage rules over exhaustive case
       lists.

    ## Generalize — do not overfit

    The instruction will be used on inputs unlike these examples. Every
    sentence must be equally useful on inputs you have never seen: include
    knowledge, definitions, and procedures that apply across the whole task
    domain, and express anything you learn from a specific example in its
    general form. The specific entities, quantities, dates, and answers in
    these examples belong to the examples, not the task — a description of
    correct output like "a single ISO-8601 date" transfers; the date itself
    does not.

    ## Reference material

    Reference skills are trusted material the user chose to provide as
    reference for this task. Review them and draw on them as needed when
    crafting the new instruction. Follow any additional guidance from the
    user.

    ## Proposal journal

    A proposal journal, if given, lists earlier proposals for this
    instruction, what the optimizer did with each one, and lessons
    distilled from them. Do not repeat an approach the journal shows was
    rejected. Build on what was accepted. When you write a change summary,
    name what you changed and why in one or two sentences.

    Do not exceed the length limit if one is given.

    Output only the new instruction text, ready to be used verbatim.
    """

    current_instruction: str = dspy.InputField(
        desc="The instruction currently given to the assistant."
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Inputs, assistant outputs, and evaluator feedback. Use these to "
        "infer the task and diagnose where the instruction fails."
    )
    proposal_journal: str = dspy.InputField(
        desc="Earlier proposals for this instruction, their outcomes, and "
        "distilled lessons. May be 'None'."
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


class DistillLessons(dspy.Signature):
    """Read a journal of instruction proposals and their outcomes. Write a
    short list of lessons about which kinds of instruction changes the
    optimizer accepted and which it rejected on this task. Each lesson must
    apply to future proposals, not to one entry. Do not mention specific
    examples, entities, or numbers from the task. Keep lessons that still
    hold from the prior list, drop ones the journal now contradicts, and
    add new ones. Output only the lessons, one per line."""

    journal: str = dspy.InputField(desc="The rendered proposal journal.")
    prior_lessons: str = dspy.InputField(desc="The current lessons, or 'None'.")
    lessons: str = dspy.OutputField(desc="The updated lessons, one per line.")


def change_summary_field():
    """A fresh output field, because a FieldInfo must not be shared across signatures."""
    return dspy.OutputField(desc="One or two sentences naming what you changed and why.")


class InstructionProposalModule(dspy.Module):
    """dspy.Module housing the proposal, compression, and distillation
    predictors.

    Keeping the predictors on a Module makes them discoverable via
    named_predictors(), lets their state be saved/loaded, and routes calls
    through the standard Module path (callbacks, history).
    """

    def __init__(self, base_instructions: str | None = None, with_change_summary: bool = False):
        super().__init__()
        signature = ProposeGeneralizableInstruction
        if base_instructions:
            signature = signature.with_instructions(base_instructions)
        if with_change_summary:
            signature = signature.append("change_summary", change_summary_field(), str)
        self.propose = dspy.Predict(signature)
        self.compress = dspy.Predict(CompressInstruction)
        self.distill = dspy.Predict(DistillLessons)

    def forward(
        self,
        *,
        current_instruction: str,
        examples_with_feedback: str,
        proposal_journal: str,
        reference_skills: str,
        additional_guidance: str,
        length_limit: str,
    ) -> dspy.Prediction:
        return self.propose(
            current_instruction=current_instruction,
            examples_with_feedback=examples_with_feedback,
            proposal_journal=proposal_journal,
            reference_skills=reference_skills,
            additional_guidance=additional_guidance,
            length_limit=length_limit,
        )


# --------------------------------------------------------------------------- #
# RLM engine
# --------------------------------------------------------------------------- #

class JournalNotes(pydantic.BaseModel):
    """The RLM's notes for the proposal journal."""

    lessons: list[str]
    hypotheses: list[str]
    entry_analysis: dict[str, str]


_RLM_PROMPT = """\
You are improving the instruction for one component of an AI program. The
assistant that runs it will see only your instruction, never these records,
so the instruction must teach the task on its own and work on inputs unlike
any you see here.

## Your evidence
- `examples`: this round's records (inputs, outputs, evaluator feedback). They
  were sampled at random for this round. Choose what to fix from these.
{seen}{journal}- `skill` and `skill_files`: reference material. Open files when a diagnosis
  calls for them.

## Measure before you read
Before reading any single record closely, compute an overview with code: how
many records failed (judge from the feedback), output lengths, recurring
phrases in feedback. Label every failure's root cause with
llm_query_batched and count the labels in Python. Never estimate a count by
eye. Check for absences too: things a correct output needs that no output
contains. Then read a few records from the largest failure group, and from
passing records that look similar, to find the rule that separates them.

## Choose a target
Fix the largest failure group that an instruction can fix, and at most one
more. Later rounds will handle the rest. If a group is really an output-format
or contract failure, fix that first.{target}

## Edit, don't rewrite
Treat `current_instruction` as a variable and change only the section your
diagnosis covers. Keep what already works. Rewrite from scratch only when
failures are widespread{rewrite}.

## Write rules that transfer
State each rule as the concept behind it, in one or two sentences. Where a
rule's boundary is subtle, show one case where it applies and one where it
does not, written in your own words with invented details. Never copy
entities, numbers, phrases or answers from the records. Never refer to "the
examples", "the data" or "this dataset" in the instruction.
Before submitting, check that the change would not flip a sample of the
passing records in {flip} to wrong: ask llm_query whether the new rule
changes the correct output.

## Reference material and guidance
Follow any additional guidance from the user. Do not exceed the length limit
if one is given.

## Submit
Check the length limit in code. Then SUBMIT:
- new_instruction.
- change_summary: what you changed, with the counts that justify it
  (e.g. "7 of 15 failures: ...").
{notes}"""

_SEEN = """\
- `seen`: records from earlier rounds. Only records tagged `current` were
  produced by the instruction you are improving. `ancestor` and
  `other_branch` records were produced by older or different instructions,
  and `seen["instructions"]` holds their text. Before counting an old failure
  as a current problem, check whether the current instruction already
  addresses it. A failure that still appears in `current` records after an
  edit meant to fix it is persistent. One that stops appearing was likely
  fixed. Use `seen` to measure prevalence and past attempts; choose what to
  fix from `examples`.
"""

_SEEN_EMPTY = "- `seen` is empty this run.\n"

_JOURNAL = """\
- `journal`: earlier proposals, whether the optimizer kept them, and your own
  past lessons and hypotheses. Read the lessons and hypotheses first.
"""

_NOTES = """\
- journal_notes:
  - lessons: general lessons about what this task rewards.
  - hypotheses: untested ideas, each with the evidence that would confirm it.
  - entry_analysis: for past entries that today's evidence explains, why they
    were kept or not kept, keyed by entry id.
"""


def rlm_instructions(*, journal: bool, seen: bool) -> str:
    """The RLM engine's prompt, without the parts for inputs this run lacks."""
    target = ""
    if seen:
        target += " Check the target's prevalence in `seen`."
    if journal:
        target += (
            " If the journal shows a similar change was rejected, try a"
            " different fix unless you have new evidence."
        )
    return _RLM_PROMPT.format(
        seen=_SEEN if seen else _SEEN_EMPTY,
        journal=_JOURNAL if journal else "",
        target=target,
        rewrite=", or when the journal shows several rounds without an accepted change" if journal else "",
        flip="`examples` and `seen`" if seen else "`examples`",
        notes=_NOTES if journal else "",
    )


class ProposeWithAnalysis(dspy.Signature):
    """Propose an instruction by analyzing records in a Python sandbox.
    `rlm_signature` replaces this text with the RLM engine's prompt."""

    current_instruction: str = dspy.InputField(
        desc="The instruction currently given to the assistant."
    )
    examples: SandboxJSON = dspy.InputField(
        desc="This round's reflective records, sampled at random."
    )
    seen: SandboxJSON = dspy.InputField(
        desc="Records from earlier rounds and the instructions that produced them."
    )
    journal: SandboxJSON = dspy.InputField(
        desc="Earlier proposals, their outcomes, lessons, and hypotheses."
    )
    skill: str = dspy.InputField(desc="Reference skills. May be 'None'.")
    skill_files: dict[str, str] = dspy.InputField(
        desc="Other files from the reference skills, keyed by path."
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
    change_summary: str = dspy.OutputField(
        desc="What you changed and the counts that justify it."
    )
    journal_notes: JournalNotes = dspy.OutputField(
        desc="Updated lessons, hypotheses, and analysis of past entries."
    )


def rlm_signature(
    base_instructions: str | None = None, *, journal: bool, seen: bool
) -> type[dspy.Signature]:
    """The RLM engine's signature for this run's settings."""
    sig = ProposeWithAnalysis.with_instructions(
        base_instructions or rlm_instructions(journal=journal, seen=seen)
    )
    if not journal:
        sig = sig.delete("journal").delete("journal_notes")
    return sig


class ProposeGeneralizableModuleSource(dspy.Signature):
    """You are improving the source code of a dspy.Flex submodule. A Flex
    holds its whole implementation as one dspy.Module subclass, and an
    optimizer rewrites that source to improve task performance. You are
    shown the submodule's task (its signature), the available context
    (tools and style notes), the catalog of allowed primitives, the
    current source, and examples of the program's inputs, outputs, and
    evaluator feedback. Write a revised source that fixes the observed
    failures.

    ## Output contract

    The source is ONE dspy.Module subclass with two coupled methods, and
    you must output the entire, internally consistent class:

    1. `def __init__(self):` calls `super().__init__()` and assigns the
       predictors the module needs. Pick the simplest primitive that fits
       each step: `dspy.Predict("...")` for a direct call (the common
       default), `dspy.ReAct` when a step must call tools, and 
       `dspy.RLM("...")` when a step requires reasoning over a large or 
       structured input. Assign no predictors at all if the task needs no
       LM call.
    2. `def forward(self, **inputs):` calls those predictors as
       `self.<name>` and returns `dspy.Prediction(<output fields>=...)`.

    Because `forward` calls predictors by name, never rename a predictor
    in one place without updating the other. Follow the primitives
    catalog and use only what it allows.

    Tools are optional. Tools listed in the available context are in
    scope by name — reference them by those exact names, never import or
    redefine them, and only those tools may be wired into `dspy.RLM` /
    `dspy.ReAct` via `tools=[...]`. If the available context says no
    tools were provided, do not reference any. You may also write plain
    helper functions inside `forward` for logic the tools do not cover;
    helpers live in this source and are optimized with it, but they
    cannot be passed to a sub-predictor as tools.

    The natural-language instructions of the predictors you construct
    live in this source, via
    `dspy.Signature("inputs -> outputs", "instructions")`, and this
    source is the ONLY place those instructions get optimized. Treat
    them as first-class: state the task, the domain knowledge the model
    lacked, the required output format, and the rules that prevent the
    observed errors.

    ## Procedure

    1. Infer the task. From the task description, the current source,
       and the examples, work out what the program must do: the input
       format, what a correct output looks like, and the conventions of
       the domain. The module will see only its inputs at run time —
       never these examples — so the source must solve the task
       completely on its own.

    2. Diagnose the failures. For each example with negative feedback,
       determine why the current module went wrong, then find the
       general fix that would have prevented it — one that helps on any
       input where the same mistake could recur. Fix a predictor's
       instructions when the failure is about WHAT the model should do
       or know; change the structure when it is about HOW the steps are
       wired together.

    3. Write the replacement. Keep whatever the current source already
       does well, apply your fixes, and prefer the simplest structure
       that addresses the diagnosis — do not add predictors, tools, or
       steps speculatively.

    ## Generalize — do not overfit

    The revised module will run on inputs unlike these examples. The
    memorization table is the failure mode to avoid: a branch that
    matches a specific training input — or an entity, phrase, or number
    from one — and returns that example's answer scores well during
    optimization and does nothing on inputs the module has never seen.
    Do not write branches keyed to specific inputs, literal answer
    lookups, or instruction text that embeds entities, quantities,
    dates, or answers from these examples. Every branch, rule, and
    instruction sentence must be one you would defend for an input you
    have never seen; express anything you learn from a specific example
    in its general form.

    ## Reference material

    Reference skills are trusted material the user chose to provide as
    reference for this task. Review them and draw on them as needed when
    revising the source and the instructions inside it. Follow any
    additional guidance from the user.

    Output only the revised Python source, ready to be used verbatim —
    no code fences, no commentary.
    """

    task_description: str = dspy.InputField(
        desc="The submodule's task: its signature name, objective, and "
        "input and output fields."
    )
    available_context: str = dspy.InputField(
        desc="Tools (in scope by name) and style notes available to the "
        "module. May be '(no extra context)'."
    )
    primitives_catalog: str = dspy.InputField(
        desc="Catalog of allowed primitives and conventions the revised "
        "source must follow."
    )
    current_source: str = dspy.InputField(
        desc="The module's current full source: one dspy.Module subclass."
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Whole-program inputs, outputs, and evaluator feedback. Use "
        "these to infer the task and diagnose failures."
    )
    reference_skills: str = dspy.InputField(
        desc="Reference material (skills) to inform the revision. May be 'None'."
    )
    additional_guidance: str = dspy.InputField(
        desc="Extra requirements from the user for the revised source. May be 'None'."
    )
    revised_source: str = dspy.OutputField(
        desc="The full revised module source: one dspy.Module subclass. "
        "Python source only, no code fences."
    )


class CodeProposalModule(dspy.Module):
    """dspy.Module housing the code proposal predictor.

    Mirrors InstructionProposalModule: keeping the predictor on a Module
    makes it discoverable via named_predictors(), lets its state be
    saved/loaded, and routes calls through the standard Module path.
    """

    def __init__(self, base_instructions: str | None = None):
        super().__init__()
        signature = ProposeGeneralizableModuleSource
        if base_instructions:
            signature = signature.with_instructions(base_instructions)
        self.propose = dspy.Predict(signature)

    def forward(
        self,
        *,
        task_description: str,
        available_context: str,
        primitives_catalog: str,
        current_source: str,
        examples_with_feedback: str,
        reference_skills: str,
        additional_guidance: str,
    ) -> dspy.Prediction:
        return self.propose(
            task_description=task_description,
            available_context=available_context,
            primitives_catalog=primitives_catalog,
            current_source=current_source,
            examples_with_feedback=examples_with_feedback,
            reference_skills=reference_skills,
            additional_guidance=additional_guidance,
        )
