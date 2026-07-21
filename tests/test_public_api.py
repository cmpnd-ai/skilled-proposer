import skilled_proposer


def test_public_exports():
    assert skilled_proposer.__version__ == "0.1.0"
    assert skilled_proposer.SkilledProposer is not None
    assert skilled_proposer.Skill is not None
    assert set(skilled_proposer.__all__) == {
        "SkilledProposer",
        "Skill",
        "InstructionProposalModule",
        "ProposeGeneralizableInstruction",
        "CompressInstruction",
    }
