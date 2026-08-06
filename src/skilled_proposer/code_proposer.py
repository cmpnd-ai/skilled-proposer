"""SkilledCodeProposer: a code proposer for dspy.Flex components under GEPA.

Implements the CodeProposalFn contract this package proposes for DSPy:
``__call__(candidate, reflective_dataset, components_to_update,
task_descriptions, context_blurbs) -> dict[str, str]``, called under the
reflection LM's context (or with an explicit ``prompt_model``). Until
``dspy.GEPA`` grows a ``code_proposer`` hook, wire it in with
``skilled_proposer.patch.use_code_proposer``.
"""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"\A```[^\n]*\n(.*)\n```\s*\Z", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip one enclosing markdown code fence, if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _validate_module_source(source: str) -> str | None:
    """Return an error message when ``source`` is not plausible module code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"proposed source does not parse: {e}"
    if not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
        return "proposed source defines no class"
    return None


def _primitives_catalog() -> str:
    """dspy's catalog of primitives allowed inside Flex source (lazy import)."""
    try:
        from dspy.predict.flex.primitives_doc import PRIMITIVES_CATALOG
    except ImportError as e:
        raise ImportError(
            "SkilledCodeProposer needs dspy.Flex, which requires dspy>=3.3; "
            f"the installed dspy has no dspy.predict.flex ({e})."
        ) from e
    return PRIMITIVES_CATALOG
