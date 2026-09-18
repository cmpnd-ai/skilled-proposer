import dspy
from dspy.utils.dummies import DummyLM

from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    DistillLessons,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
    ProposeGeneralizableModuleSource,
    batch_signature,
)


def test_propose_signature_fields():
    assert set(ProposeGeneralizableInstruction.input_fields) == {
        "current_instruction",
        "examples_with_feedback",
        "proposal_journal",
        "reference_skills",
        "additional_guidance",
        "length_limit",
    }
    assert set(ProposeGeneralizableInstruction.output_fields) == {"new_instruction"}


def test_propose_prompt_mentions_journal():
    text = ProposeGeneralizableInstruction.instructions.lower()
    assert "journal" in text
    assert "rejected" in text


def test_distill_signature_fields():
    assert set(DistillLessons.input_fields) == {"journal", "prior_lessons"}
    assert set(DistillLessons.output_fields) == {"lessons"}



def test_compress_signature_fields():
    assert set(CompressInstruction.input_fields) == {"instruction", "length_limit"}
    assert set(CompressInstruction.output_fields) == {"shortened_instruction"}


def test_module_default_meta_prompt_has_anti_overfitting_rules():
    m = InstructionProposalModule()
    assert "do not overfit" in m.propose.signature.instructions.lower()


def test_module_base_instructions_override():
    m = InstructionProposalModule("Custom meta prompt.")
    assert m.propose.signature.instructions == "Custom meta prompt."
    assert set(m.propose.signature.output_fields) == {"new_instruction"}


def test_module_change_summary_off_by_default():
    m = InstructionProposalModule()
    assert set(m.propose.signature.output_fields) == {"new_instruction"}


def test_module_change_summary_on():
    m = InstructionProposalModule(with_change_summary=True)
    assert list(m.propose.signature.output_fields) == ["new_instruction", "change_summary"]


def test_module_change_summary_with_base_instructions():
    m = InstructionProposalModule("Custom.", with_change_summary=True)
    assert m.propose.signature.instructions == "Custom."
    assert "change_summary" in m.propose.signature.output_fields


def test_module_named_predictors():
    names = {name for name, _ in InstructionProposalModule().named_predictors()}
    assert names == {"propose", "compress", "distill"}


def test_code_signature_fields():
    assert set(ProposeGeneralizableModuleSource.input_fields) == {
        "task_description",
        "available_context",
        "primitives_catalog",
        "current_source",
        "examples_with_feedback",
        "reference_skills",
        "additional_guidance",
    }
    assert set(ProposeGeneralizableModuleSource.output_fields) == {"revised_source"}


def test_code_module_default_meta_prompt_has_anti_overfitting_rules():
    m = CodeProposalModule()
    instructions = m.propose.signature.instructions.lower()
    assert "do not overfit" in instructions
    assert "memorization table" in instructions


def test_code_module_base_instructions_override():
    m = CodeProposalModule("Custom code meta prompt.")
    assert m.propose.signature.instructions == "Custom code meta prompt."
    assert set(m.propose.signature.output_fields) == {"revised_source"}


def test_code_module_named_predictors():
    names = {name for name, _ in CodeProposalModule().named_predictors()}
    assert names == {"propose"}


def test_batch_signature_fields():
    sig = batch_signature()
    assert set(sig.input_fields) == set(ProposeGeneralizableInstruction.input_fields) | {"candidate_count"}
    assert list(sig.output_fields) == ["new_instructions", "change_summaries"]
    assert sig.instructions.startswith(ProposeGeneralizableInstruction.instructions.rstrip())
    assert "## Candidates" in sig.instructions


def test_batch_signature_base_instructions_replace_core_text():
    sig = batch_signature("Custom base.")
    assert sig.instructions.startswith("Custom base.")
    assert "## Candidates" in sig.instructions
    assert "Diagnose the failures" not in sig.instructions


def test_core_signature_untouched_by_batch_builder():
    batch_signature()
    assert set(ProposeGeneralizableInstruction.output_fields) == {"new_instruction"}
    assert "## Candidates" not in ProposeGeneralizableInstruction.instructions


def test_module_builds_batch_predictor_only_when_candidates_set():
    assert InstructionProposalModule().propose_many is None
    module = InstructionProposalModule(candidates=3)
    assert isinstance(module.propose_many, dspy.Predict)
    lm = DummyLM([{"new_instructions": ["one", "two"], "change_summaries": ["r1", "r2"]}])
    with dspy.context(lm=lm):
        pred = module.forward_many(
            current_instruction="c",
            examples_with_feedback="e",
            proposal_journal="None",
            reference_skills="None",
            additional_guidance="None",
            length_limit="None",
            candidate_count=2,
        )
    assert pred.new_instructions == ["one", "two"]
    assert pred.change_summaries == ["r1", "r2"]
