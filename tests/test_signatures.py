from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    DistillLessons,
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


from skilled_proposer.signatures import JournalNotes, rlm_instructions, rlm_signature


def test_rlm_instructions_with_everything():
    text = rlm_instructions(journal=True, seen=True)
    assert "Only records tagged `current`" in text
    assert "seen[\"instructions\"]" in text
    assert "`journal`:" in text and "journal_notes" in text
    assert "Check the target's prevalence in `seen`." in text
    assert "Measure before you read" in text


def test_rlm_instructions_without_journal_or_seen():
    text = rlm_instructions(journal=False, seen=False)
    assert "`seen` is empty this run." in text
    assert "journal" not in text.lower()
    assert "prevalence in `seen`" not in text


def test_rlm_signature_fields():
    sig = rlm_signature(journal=True, seen=True)
    assert list(sig.input_fields) == [
        "current_instruction", "examples", "seen", "journal", "skill", "skill_files",
        "additional_guidance", "length_limit",
    ]
    assert list(sig.output_fields) == ["new_instruction", "change_summary", "journal_notes"]
    assert sig.output_fields["journal_notes"].annotation is JournalNotes
    bare = rlm_signature(journal=False, seen=False)
    assert "journal" not in bare.input_fields
    assert "journal_notes" not in bare.output_fields


def test_rlm_signature_base_instructions_replace_the_prompt():
    assert rlm_signature("Custom.", journal=False, seen=False).instructions == "Custom."
