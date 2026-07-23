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
