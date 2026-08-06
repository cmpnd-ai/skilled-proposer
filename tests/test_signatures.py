from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
    ProposeGeneralizableModuleSource,
)


def test_propose_signature_fields():
    assert set(ProposeGeneralizableInstruction.input_fields) == {
        "current_instruction",
        "examples_with_feedback",
        "reference_skills",
        "additional_guidance",
        "length_limit",
    }
    assert set(ProposeGeneralizableInstruction.output_fields) == {"new_instruction"}


def test_compress_signature_fields():
    assert set(CompressInstruction.input_fields) == {"instruction", "length_limit"}
    assert set(CompressInstruction.output_fields) == {"shortened_instruction"}


def test_module_default_meta_prompt_has_anti_overfitting_rules():
    m = InstructionProposalModule()
    assert "do not overfit" in m.propose.signature.instructions.lower()


def test_module_base_instructions_override():
    m = InstructionProposalModule("Custom meta prompt.")
    assert m.propose.signature.instructions == "Custom meta prompt."
    # Fields are unchanged by the override.
    assert set(m.propose.signature.output_fields) == {"new_instruction"}


def test_module_named_predictors():
    names = {name for name, _ in InstructionProposalModule().named_predictors()}
    assert names == {"propose", "compress"}


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
