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
