"""Deprecated: wire a CodeProposalFn into dspy.GEPA by patching dspy.

dspy 3.4 added a ``code_proposer`` parameter to ``dspy.GEPA``, so pass
the proposer there instead::

    optimizer = dspy.GEPA(..., code_proposer=SkilledCodeProposer(...))

This module swaps ``propose_code`` in ``dspy.teleprompt.gepa.gepa_utils``
for a wrapper that calls a user-supplied proposer with the
CodeProposalFn contract. It is kept for one release so existing callers
keep working, and will be removed in the next.

When ``code_proposer`` is also passed to ``dspy.GEPA``, GEPA calls it
and never reaches the patched function.

This patch is process-global state: it replaces a module-level function
for as long as it is installed, affecting every GEPA compile running in
the process, so it is meant to be installed once around a single
``compile`` call (e.g. via ``use_code_proposer``), not left installed
across concurrent compiles.
"""

from __future__ import annotations

import contextlib
import inspect
import warnings

_EXPECTED_PARAMS = [
    "code_keys",
    "candidate",
    "reflective_dataset",
    "task_descriptions",
    "context_blurbs",
    "reflection_lm",
]

_original_propose_code = None


_DEPRECATION = (
    "{name} is deprecated and will be removed in the next release; pass "
    "the proposer to dspy.GEPA(code_proposer=...) instead."
)


def _gepa_utils():
    from dspy.teleprompt.gepa import gepa_utils

    return gepa_utils


def install_code_proposer(proposer) -> None:
    """Deprecated: pass ``proposer`` to ``dspy.GEPA(code_proposer=...)``.

    Replaces dspy's built-in Flex code proposer with ``proposer``, any
    CodeProposalFn: ``(candidate, reflective_dataset,
    components_to_update, task_descriptions, context_blurbs) ->
    dict[str, str]``. The wrapper runs it inside the reflection LM's
    context when GEPA supplies one.
    """
    warnings.warn(
        _DEPRECATION.format(name="install_code_proposer"),
        DeprecationWarning,
        stacklevel=2,
    )
    _install(proposer)


def _install(proposer) -> None:
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
            "skilled-proposer version cannot patch it. Pass the proposer "
            "to dspy.GEPA(code_proposer=...) instead."
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
    """Restore dspy's built-in code proposer. Safe to call when not installed.

    Not deprecated on its own, so cleanup after a deprecated install
    does not warn twice.
    """
    global _original_propose_code
    if _original_propose_code is None:
        return
    _gepa_utils().propose_code = _original_propose_code
    _original_propose_code = None


@contextlib.contextmanager
def use_code_proposer(proposer):
    """Deprecated: pass ``proposer`` to ``dspy.GEPA(code_proposer=...)``.

    Context manager: install ``proposer`` for the duration of the block.
    """
    warnings.warn(
        _DEPRECATION.format(name="use_code_proposer"),
        DeprecationWarning,
        stacklevel=3,
    )
    _install(proposer)
    try:
        yield proposer
    finally:
        uninstall_code_proposer()
