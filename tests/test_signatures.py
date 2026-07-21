from skilled_proposer.signatures import (
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
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
