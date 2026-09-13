from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    DistillLessons,
    DiversifyInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
    ProposeGeneralizableModuleSource,
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


def test_diversify_signature_fields():
    assert set(DiversifyInstruction.input_fields) == {
        "current_instruction",
        "proposal",
        "near_duplicates",
        "examples_with_feedback",
        "reference_skills",
        "additional_guidance",
        "length_limit",
    }
    assert set(DiversifyInstruction.output_fields) == {"new_instruction"}
    assert "do not overfit" in DiversifyInstruction.instructions.lower()


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
    assert set(m.diversify.signature.output_fields) == {"new_instruction"}


def test_module_change_summary_on():
    m = InstructionProposalModule(with_change_summary=True)
    assert list(m.propose.signature.output_fields) == ["new_instruction", "change_summary"]
    assert list(m.diversify.signature.output_fields) == ["new_instruction", "change_summary"]


def test_module_change_summary_with_base_instructions():
    m = InstructionProposalModule("Custom.", with_change_summary=True)
    assert m.propose.signature.instructions == "Custom."
    assert "change_summary" in m.propose.signature.output_fields


def test_module_named_predictors():
    names = {name for name, _ in InstructionProposalModule().named_predictors()}
    assert names == {"propose", "diversify", "compress", "distill"}


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
