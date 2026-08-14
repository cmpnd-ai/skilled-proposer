"""Reflection signatures and the dspy.Module that houses them."""

from __future__ import annotations

import dspy


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
       default), `dspy.ChainOfThought("...")` when explicit reasoning
       helps, and `dspy.RLM` / `dspy.ReAct` when a step must call tools
       or explore a large or structured input. Assign no predictors at
       all if the task needs no LM call.
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
