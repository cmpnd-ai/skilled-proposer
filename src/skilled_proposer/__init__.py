"""skilled-proposer: GEPA instruction and Flex code proposers for DSPy."""

from skilled_proposer.code_proposer import SkilledCodeProposer
from skilled_proposer.patch import (
    install_code_proposer,
    uninstall_code_proposer,
    use_code_proposer,
)
from skilled_proposer.proposer import SkilledProposer
from skilled_proposer.signatures import (
    CodeProposalModule,
    CompressInstruction,
    InstructionProposalModule,
    ProposeGeneralizableInstruction,
    ProposeGeneralizableModuleSource,
)
from skilled_proposer.skill import Skill

__version__ = "0.1.1"

__all__ = [
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
]
