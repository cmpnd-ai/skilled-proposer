import skilled_proposer


def test_public_exports():
    assert skilled_proposer.__version__ == "0.1.2"
    assert skilled_proposer.SkilledProposer is not None
    assert skilled_proposer.SkilledCodeProposer is not None
    assert skilled_proposer.Skill is not None
    assert set(skilled_proposer.__all__) == {
        "SkilledProposer",
        "SkilledCodeProposer",
        "Skill",
        "InstructionProposalModule",
        "ProposeGeneralizableInstruction",
        "CompressInstruction",
        "CodeProposalModule",
        "ProposeGeneralizableModuleSource",
        "use_code_proposer",
        "install_code_proposer",
        "uninstall_code_proposer",
    }
