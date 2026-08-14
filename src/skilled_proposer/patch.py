"""Wire a CodeProposalFn into dspy.GEPA until the upstream hook exists.

``dspy.GEPA`` routes ``dspy.Flex`` code components to the module-level
function ``propose_code`` in ``dspy.teleprompt.gepa.gepa_utils``; there
is no ``code_proposer`` parameter yet. This module swaps that function
for a wrapper that calls a user-supplied proposer with the
CodeProposalFn contract. Remove once dspy.GEPA accepts a
``code_proposer`` directly.

Note: dspy still logs a warning that a custom instruction_proposer
"skips code components" — under this patch that warning is expected and
harmless, since the patched proposer handles them.

This patch is process-global state: it replaces a module-level function
for as long as it is installed, affecting every GEPA compile running in
the process, so it is meant to be installed once around a single
``compile`` call (e.g. via ``use_code_proposer``), not left installed
across concurrent compiles.
"""

from __future__ import annotations

import contextlib
import inspect

_EXPECTED_PARAMS = [
    "code_keys",
    "candidate",
    "reflective_dataset",
    "task_descriptions",
    "context_blurbs",
    "reflection_lm",
]

_original_propose_code = None


def _gepa_utils():
    try:
        from dspy.teleprompt.gepa import gepa_utils
    except ImportError as e:
        raise ImportError(
            "Patching GEPA's code proposer requires dspy>=3.3 "
            f"(no dspy.teleprompt.gepa.gepa_utils in the installed dspy: {e})."
        ) from e
    if not hasattr(gepa_utils, "propose_code"):
        raise RuntimeError(
            "The installed dspy has no gepa_utils.propose_code; this "
            "feature requires dspy>=3.3 and this skilled-proposer version "
            "cannot patch it."
        )
    return gepa_utils


def install_code_proposer(proposer) -> None:
    """Replace dspy's built-in Flex code proposer with ``proposer``.

    ``proposer`` is any CodeProposalFn: ``(candidate, reflective_dataset,
    components_to_update, task_descriptions, context_blurbs) ->
    dict[str, str]``. The wrapper runs it inside the reflection LM's
    context when GEPA supplies one.
    """
    global _original_propose_code
    if _original_propose_code is not None:
        raise RuntimeError("A code proposer is already installed.")

    gepa_utils = _gepa_utils()
    params = list(inspect.signature(gepa_utils.propose_code).parameters)
    if params != _EXPECTED_PARAMS:
        import dspy

        raise RuntimeError(
            f"dspy {getattr(dspy, '__version__', '?')}'s propose_code has "
            f"parameters {params}, expected {_EXPECTED_PARAMS}; this "
            "skilled-proposer version cannot patch it. (This feature "
            "requires dspy>=3.3; a newer dspy may have changed internals.)"
        )

    def _patched(code_keys, candidate, reflective_dataset,
                 task_descriptions, context_blurbs, reflection_lm):
        import dspy

        if reflection_lm is not None:
            with dspy.context(lm=reflection_lm):
                return proposer(
                    candidate=candidate,
                    reflective_dataset=reflective_dataset,
                    components_to_update=list(code_keys),
                    task_descriptions=task_descriptions,
                    context_blurbs=context_blurbs,
                )
        return proposer(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=list(code_keys),
            task_descriptions=task_descriptions,
            context_blurbs=context_blurbs,
        )

    _original_propose_code = gepa_utils.propose_code
    gepa_utils.propose_code = _patched


def uninstall_code_proposer() -> None:
    """Restore dspy's built-in code proposer. Safe to call when not installed."""
    global _original_propose_code
    if _original_propose_code is None:
        return
    _gepa_utils().propose_code = _original_propose_code
    _original_propose_code = None


@contextlib.contextmanager
def use_code_proposer(proposer):
    """Context manager: install ``proposer`` for the duration of the block.

    Usage::

        with use_code_proposer(SkilledCodeProposer(skills=[...])):
            optimized = optimizer.compile(program, trainset=train, valset=val)
    """
    install_code_proposer(proposer)
    try:
        yield proposer
    finally:
        uninstall_code_proposer()
