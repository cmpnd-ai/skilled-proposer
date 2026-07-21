"""skilled-proposer: a GEPA instruction proposer for DSPy."""

from skilled_proposer.proposer import SkilledProposer
from skilled_proposer.signatures import (
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
)
from skilled_proposer.skill import Skill

__version__ = "0.1.0"

__all__ = [
    "SkilledProposer",
    "Skill",
    "InstructionProposalModule",
    "ProposeGeneralizableInstruction",
    "CompressInstruction",
]
