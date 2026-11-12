"""SkilledCodeProposer: a code proposer for dspy.Flex components under GEPA.

Implements dspy's CodeProposalFn protocol
(``dspy.teleprompt.gepa.gepa_utils.CodeProposalFn``):
``__call__(candidate, reflective_dataset, components_to_update,
task_descriptions, context_blurbs) -> dict[str, str]``. Pass it to
``dspy.GEPA(code_proposer=...)``, which calls it under the reflection
LM's context (or set an explicit ``prompt_model``).
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Mapping, Sequence
from pathlib import Path

import dspy

from skilled_proposer.proposer import _attempt, _render_examples
from skilled_proposer.signatures import CodeProposalModule
from skilled_proposer.skill import Skill, render_skills

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
    class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if not class_defs:
        return "proposed source defines no class"
    has_forward = any(
        isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and member.name == "forward"
        for class_def in class_defs
        for member in ast.walk(class_def)
    )
    if not has_forward:
        return "proposed source defines no class with a forward method"
    return None


def _primitives_catalog() -> str:
    """dspy's catalog of primitives allowed inside Flex source."""
    from dspy.predict.flex.primitives_doc import PRIMITIVES_CATALOG

    return PRIMITIVES_CATALOG


class SkilledCodeProposer:
    """Proposes revised source for ``dspy.Flex`` components, optionally
    informed by skills and extra guidance.

    Implements dspy's CodeProposalFn protocol; pass it to
    ``dspy.GEPA(code_proposer=...)``. GEPA calls it once per reflection
    round with the code components to update; it returns a full
    replacement ``dspy.Module`` subclass source per component.

    Args:
        skills: Iterable of ``Skill`` objects, paths to SKILL.md files or
            skill directories, or inline strings of skill content.
        additional_instructions: Ad-hoc guidance for the reflection LM,
            applied to every proposal.
        base_instructions: Optional replacement for the built-in code
            meta-prompt (the docstring of
            ``ProposeGeneralizableModuleSource``). Replaces the
            anti-overfitting contract too — include equivalent rules in
            your replacement if you still want them.
        prompt_model: Optional dspy.LM to run proposals with. Not needed
            under ``dspy.GEPA(code_proposer=...)``, which calls the
            proposer in the reflection LM's context.
        max_examples: Cap on reflective examples rendered per component.
            None = no cap.
        retries: Extra attempts per component when a proposal fails or
            does not validate as module source. Ignored when
            on_error="raise".
        on_error: What to do when a component's proposal still fails after
            retries. "skip" (default) logs it and leaves that component
            out of the returned dict, so GEPA keeps the parent's source
            and does not spend minibatch evaluations on an unchanged
            child. "keep" is an alias for "skip". "raise" propagates the
            first error with no retries. Either way, an LM/provider error
            (LMError) always propagates.
    """

    def __init__(
        self,
        skills: Sequence[Skill | str | Path] | None = None,
        additional_instructions: str | None = None,
        base_instructions: str | None = None,
        prompt_model: "dspy.LM | None" = None,
        max_examples: int | None = None,
        retries: int = 1,
        on_error: str = "skip",
    ):
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if on_error not in ("skip", "keep", "raise"):
            raise ValueError('on_error must be "skip" or "raise"')

        self.skills = [Skill.load(s) for s in (skills or [])]
        self.additional_instructions = (additional_instructions or "").strip()
        self.base_instructions = (base_instructions or "").strip() or None
        self.prompt_model = prompt_model
        self.max_examples = max_examples
        self.retries = retries
        self.on_error = "skip" if on_error == "keep" else on_error

        self.module = CodeProposalModule(self.base_instructions)
        # Indirection so tests (and callers) can stub the catalog fetch.
        self._catalog = _primitives_catalog

    # -- CodeProposalFn -----------------------------------------------------

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        task_descriptions: Mapping[str, str],
        context_blurbs: Mapping[str, str],
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for name in components_to_update:
            current = candidate[name]
            examples = list(reflective_dataset.get(name, []))
            if self.max_examples is not None:
                examples = examples[: self.max_examples]
            proposed = _attempt(
                lambda: self._propose_one(
                    current,
                    examples,
                    task_descriptions.get(name, name),
                    context_blurbs.get(name, "(no extra context)"),
                ),
                name=name,
                retries=self.retries,
                on_error=self.on_error,
                label="SkilledCodeProposer",
            )
            if proposed is not None:
                results[name] = proposed
        return results

    # -- Internals ----------------------------------------------------------

    def _propose_one(
        self,
        current_source: str,
        examples: Sequence[Mapping[str, Any]],
        task_description: str,
        context_blurb: str,
    ) -> str:
        kwargs = dict(
            task_description=task_description,
            available_context=context_blurb,
            primitives_catalog=self._catalog(),
            current_source=current_source,
            examples_with_feedback=_render_examples(examples),
            reference_skills=render_skills(self.skills),
            additional_guidance=self.additional_instructions or "None",
        )
        if self.prompt_model is not None:
            with dspy.context(lm=self.prompt_model):
                raw = self.module(**kwargs).revised_source
        else:
            raw = self.module(**kwargs).revised_source
        source = _strip_code_fences(raw.strip()).strip()
        error = _validate_module_source(source)
        if error:
            raise ValueError(error)
        return source
