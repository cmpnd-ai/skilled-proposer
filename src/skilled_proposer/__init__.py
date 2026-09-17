"""skilled-proposer: GEPA instruction and Flex code proposers for DSPy."""

from skilled_proposer.code_proposer import SkilledCodeProposer
from skilled_proposer.journal import Journal, JournalEntry
from skilled_proposer.patch import (
    install_code_proposer,
    uninstall_code_proposer,
    use_code_proposer,
)
from skilled_proposer.proposer import SkilledProposer
from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    DistillLessons,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
    ProposeGeneralizableModuleSource,
)
from skilled_proposer.skill import Skill

__version__ = "0.2.0"

__all__ = [
    "SkilledProposer",
    "SkilledCodeProposer",
    "Skill",
    "Journal",
    "JournalEntry",
    "InstructionProposalModule",
    "ProposeGeneralizableInstruction",
    "CompressInstruction",
    "DistillLessons",
    "CodeProposalModule",
    "ProposeGeneralizableModuleSource",
    "use_code_proposer",
    "install_code_proposer",
    "uninstall_code_proposer",
]
