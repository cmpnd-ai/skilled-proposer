"""SkilledProposer: a custom GEPA instruction proposer for DSPy.

Implements GEPA's `ProposalFn` protocol (gepa.core.adapter.ProposalFn), so it
plugs directly into `dspy.GEPA(instruction_proposer=...)`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import dspy

try:
    from dspy.utils.exceptions import LMError
except ImportError:  # dspy < 3.x fallback: nothing raises it
    class LMError(Exception):
        pass

from skilled_proposer.dedupe import DedupeConfig, find_duplicates, render_duplicates
from skilled_proposer.journal import Journal, JournalEntry
from skilled_proposer.signatures import InstructionProposalModule
from skilled_proposer.skill import Skill, render_skills

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


@dataclass
class Proposal:
    """One component's proposal and what the proposer learned about it."""

    text: str
    change_summary: str = ""
    near_duplicates: list[str] = field(default_factory=list)
    duplicate_after_retry: bool = False


def _attempt(propose, *, name: str, retries: int, on_error: str, label: str):
    """Run ``propose`` for one component with the shared failure policy.

    Returns the proposal, or None when the component should be left out
    of the result. LMError always propagates; on_error="raise" propagates
    the first failure of any kind.
    """
    attempts = 1 if on_error == "raise" else retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return propose()
        except LMError:
            raise
        except Exception:
            if on_error == "raise":
                raise
            if attempt < attempts:
                logger.warning(
                    "%s failed for component %r (attempt %d of %d); retrying.",
                    label, name, attempt, attempts, exc_info=True,
                )
                continue
            logger.exception(
                "%s failed for component %r after %d attempt(s); "
                "leaving it out of the proposal.",
                label, name, attempts,
            )
    return None


class SkilledProposer:
    """GEPA `ProposalFn` that proposes generalizable instructions, optionally
    informed by skills, extra guidance, and a length budget.

    Args:
        skills: Iterable of `Skill` objects, paths to SKILL.md files or skill
            directories, or inline strings of skill content.
        additional_instructions: Ad-hoc guidance for the reflection LM, applied
            to every proposal.
        base_instructions: Optional replacement for the built-in meta-prompt
            (the docstring of `ProposeGeneralizableInstruction`). The
            input/output fields are unchanged; only the prompt text is
            swapped, via `Signature.with_instructions`. Note this replaces
            the anti-overfitting contract too — if you still want it,
            include equivalent rules in your replacement text.
        max_tokens: Optional cap on the proposed instruction length in tokens
            (counted with litellm's tokenizer when available, else ~4 chars
            per token).
        max_words: Optional cap on the proposed instruction length in words.
        prompt_model: Optional dspy.LM to run proposals with. Not needed under
            `dspy.GEPA`, which already wraps calls in the reflection LM's
            context; useful with the standalone `gepa` package.
        max_examples: Cap on reflective examples rendered per component, to
            keep the meta-prompt bounded. None = no cap.
        retries: Extra attempts per component when a proposal fails or
            comes back unusable, before giving up on that component.
            Ignored when on_error="raise".
        on_error: What to do when a component's proposal still fails after
            retries. "skip" (default) logs the error and leaves that
            component out of the returned dict. GEPA then keeps the
            parent's text for it, and when no component survives it
            skips the proposal entirely instead of spending minibatch
            evaluations on a child identical to its parent. "keep" is an
            alias for "skip". "raise" propagates the first error with no
            retries, so failures surface during development. Either way,
            an LM/provider error (LMError) always propagates.
        journal: Experimental. Record every proposal and what GEPA did with
            it, and show that record to the reflection model on each call.
            Register the proposer's callbacks with
            ``dspy.GEPA(gepa_kwargs=proposer.gepa_kwargs())``; without
            that, the journal records proposals but never their verdicts.
        journal_path: Optional JSON file for the journal. Loaded when it
            exists and written after every iteration, so a resumed run keeps
            its record. Requires journal=True.
        journal_entries: How many recent entries render into the prompt.
        distill_every: Closed entries between lesson distillations. None
            turns distillation off. Ignored when journal is off.
        dedupe: Experimental. Screen each proposal against earlier rejected
            proposals and the current candidate pool. On a near duplicate,
            ask the reflection model once for a materially different
            approach. True uses DedupeConfig(); pass a DedupeConfig to
            change the threshold, the retry cap, the sets, or what happens
            to a proposal that is still a duplicate.
    """

    def __init__(
        self,
        skills: Sequence[Skill | str | Path] | None = None,
        additional_instructions: str | None = None,
        base_instructions: str | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        prompt_model: "dspy.LM | None" = None,
        max_examples: int | None = None,
        retries: int = 1,
        on_error: str = "skip",
        *,
        journal: bool = False,
        journal_path: str | Path | None = None,
        journal_entries: int = 12,
        distill_every: int | None = 5,
        dedupe: bool | DedupeConfig = False,
    ):
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_words is not None and max_words <= 0:
            raise ValueError("max_words must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if on_error not in ("skip", "keep", "raise"):
            raise ValueError('on_error must be "skip" or "raise"')
        if journal_path is not None and not journal:
            raise ValueError("journal_path requires journal=True")
        if journal_entries <= 0:
            raise ValueError("journal_entries must be positive")
        if distill_every is not None and distill_every <= 0:
            raise ValueError("distill_every must be positive or None")
        if isinstance(dedupe, DedupeConfig):
            self.dedupe: DedupeConfig | None = dedupe
        elif dedupe is True:
            self.dedupe = DedupeConfig()
        elif dedupe is False:
            self.dedupe = None
        else:
            raise ValueError("dedupe must be a bool or a DedupeConfig")

        self.skills = [Skill.load(s) for s in (skills or [])]
        self.additional_instructions = (additional_instructions or "").strip()
        self.base_instructions = (base_instructions or "").strip() or None
        self.max_tokens = max_tokens
        self.max_words = max_words
        self.prompt_model = prompt_model
        self.max_examples = max_examples
        self.retries = retries
        self.on_error = "skip" if on_error == "keep" else on_error

        self.journal_enabled = journal
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.journal_entries = journal_entries
        self.distill_every = distill_every if journal else None
        self.journal = Journal.load(self.journal_path) if self.journal_path else Journal()
        self.stats = {"duplicates": 0, "duplicates_after_retry": 0}

        # Callback bookkeeping for the current iteration.
        self._iteration = 0
        self._callbacks_seen = False
        self._warned_unregistered = False
        self._pending_parents: list[tuple[int, dict[str, str]]] = []
        self._calls_this_iteration = 0
        self._pool: list[dict[str, str]] = []

        # All LM-facing predictors live on a dspy.Module.
        self.module = InstructionProposalModule(
            self.base_instructions, with_change_summary=journal
        )

    # -- ProposalFn ---------------------------------------------------------

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        **kwargs: Any,
    ) -> dict[str, str]:
        parent_idx = self._begin_call()
        task = self._calls_this_iteration
        self._calls_this_iteration += 1

        results: dict[str, str] = {}
        for name in components_to_update:
            current = candidate[name]
            examples = list(reflective_dataset.get(name, []))
            if self.max_examples is not None:
                examples = examples[: self.max_examples]
            proposal = _attempt(
                lambda: self._propose_one(name, current, examples),
                name=name,
                retries=self.retries,
                on_error=self.on_error,
                label="SkilledProposer",
            )
            if proposal is None:
                continue
            results[name] = proposal.text
            if self.journal_enabled:
                self.journal.open(
                    JournalEntry(
                        iteration=self._iteration,
                        task=task,
                        component=name,
                        parent_idx=parent_idx,
                        parent_text=current,
                        proposed_text=proposal.text,
                        change_summary=proposal.change_summary,
                        near_duplicates=list(proposal.near_duplicates),
                        duplicate_after_retry=proposal.duplicate_after_retry,
                    )
                )
        return results

    def _begin_call(self) -> int | None:
        """Pick the parent for this call and warn once if no callback ever fired."""
        if not self._callbacks_seen:
            self._iteration += 1
            self._calls_this_iteration = 0
            if self.journal_enabled and not self._warned_unregistered:
                self._warned_unregistered = True
                logger.warning(
                    "SkilledProposer callbacks are not registered, so the journal "
                    "will record proposals without verdicts. Pass "
                    "gepa_kwargs=proposer.gepa_kwargs() to dspy.GEPA."
                )
        if self._pending_parents:
            parent_idx, _ = self._pending_parents.pop(0)
            return parent_idx
        return None

    def gepa_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Keyword arguments for dspy.GEPA(gepa_kwargs=...) that register this
        proposer as a callback. Extra keywords pass through."""
        callbacks = [self, *overrides.pop("callbacks", [])]
        return {"callbacks": callbacks, **overrides}

    # -- GEPACallback -------------------------------------------------------

    def on_candidate_selected(self, event: Mapping[str, Any]) -> None:
        self._callbacks_seen = True
        iteration = event["iteration"]
        if iteration != self._iteration:
            self._iteration = iteration
            self._pending_parents = []
            self._calls_this_iteration = 0
        self._pending_parents.append((event["candidate_idx"], dict(event["candidate"])))

    def on_valset_evaluated(self, event: Mapping[str, Any]) -> None:
        # GEPA rejects before it evaluates on the valset, so a task with a
        # verdict already is never the one being evaluated here.
        iteration = event["iteration"]
        for task in self.journal.open_tasks(iteration):
            entries = self.journal.entries_for_task(iteration, task)
            if all(e.accepted is None and e.valset_average is None for e in entries):
                for e in entries:
                    e.valset_average = event["average_score"]
                return

    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        self._attach_verdict(event["iteration"], accepted=True, reason="")

    def on_candidate_rejected(self, event: Mapping[str, Any]) -> None:
        self._attach_verdict(event["iteration"], accepted=False, reason=event.get("reason", ""))

    def _attach_verdict(self, iteration: int, *, accepted: bool, reason: str) -> None:
        open_tasks = [
            (task, self.journal.entries_for_task(iteration, task))
            for task in self.journal.open_tasks(iteration)
        ]
        unverdicted = [(t, es) for t, es in open_tasks if all(e.accepted is None for e in es)]
        if accepted:
            # An accepted task already carries its valset score.
            with_valset = [(t, es) for t, es in unverdicted if es[0].valset_average is not None]
            unverdicted = with_valset or unverdicted
        if not unverdicted:
            return
        _, entries = unverdicted[0]
        for e in entries:
            e.accepted = accepted
            e.reason = reason

    def on_iteration_end(self, event: Mapping[str, Any]) -> None:
        iteration = event["iteration"]
        state = event["state"]
        self._attach_minibatch_scores(iteration, state)
        self.journal.close(iteration)
        self._pool = [dict(c) for c in getattr(state, "program_candidates", [])]
        if self.journal_path is not None:
            self.journal.save(self.journal_path)
        if self.distill_every and self.journal.closed_since_distill >= self.distill_every:
            self._distill()

    def _attach_minibatch_scores(self, iteration: int, state: Any) -> None:
        trace = state.full_program_trace[-1] if getattr(state, "full_program_trace", None) else {}
        evaluated = [t for t in trace.get("tasks", []) if "new_subsample_scores" in t]
        for task, record in zip(self.journal.open_tasks(iteration), evaluated):
            before = record.get("subsample_scores") or []
            after = record.get("new_subsample_scores") or []
            for e in self.journal.entries_for_task(iteration, task):
                e.minibatch_before = fmean(before) if before else None
                e.minibatch_after = fmean(after) if after else None

    def _distill(self) -> None:
        """Refresh the lessons from the whole journal. Never raises into GEPA."""
        try:
            pred = self._run(
                self.module.distill,
                journal=self.journal.render(limit=None),
                prior_lessons=self.journal.lessons.strip() or "None",
            )
            lessons = (pred.lessons or "").strip()
            if lessons:
                self.journal.lessons = lessons
        except Exception:
            logger.warning("Lesson distillation failed; keeping the prior lessons.", exc_info=True)
        self.journal.closed_since_distill = 0
        if self.journal_path is not None:
            self.journal.save(self.journal_path)

    # -- Internals ----------------------------------------------------------

    def _run(self, predictor, **kwargs) -> dspy.Prediction:
        if self.prompt_model is not None:
            with dspy.context(lm=self.prompt_model):
                return predictor(**kwargs)
        return predictor(**kwargs)

    def _propose_one(
        self, name: str, current_instruction: str, examples: Sequence[Mapping[str, Any]]
    ) -> Proposal | None:
        kwargs = dict(
            current_instruction=current_instruction,
            examples_with_feedback=_render_examples(examples),
            proposal_journal=self._render_journal(),
            reference_skills=self._render_skills(),
            additional_guidance=self.additional_instructions or "None",
            length_limit=self._length_limit_text(),
        )
        pred = self._run(self.module, **kwargs)
        proposal = _proposal_from(pred)

        if self.dedupe is not None:
            proposal = self._screen(name, current_instruction, proposal, kwargs)
            if proposal is None:
                return None

        proposal.text = self._enforce_length(proposal.text)
        return proposal

    def _screen(
        self, name: str, current_instruction: str, proposal: Proposal, kwargs: dict[str, Any]
    ) -> Proposal | None:
        """Diversify a near-duplicate proposal at most max_retries times."""
        config = self.dedupe
        screening = self._screening_set(name, current_instruction)
        matches = find_duplicates(proposal.text, screening, config.threshold)
        if not matches:
            return proposal
        self.stats["duplicates"] += 1
        first_labels = [label for label, _, _ in matches]

        calls = 0
        while matches and calls < config.max_retries:
            calls += 1
            pred = self._run(
                self.module.diversify,
                current_instruction=current_instruction,
                proposal=proposal.text,
                near_duplicates=render_duplicates(matches),
                examples_with_feedback=kwargs["examples_with_feedback"],
                reference_skills=kwargs["reference_skills"],
                additional_guidance=kwargs["additional_guidance"],
                length_limit=kwargs["length_limit"],
            )
            proposal = _proposal_from(pred)
            matches = find_duplicates(proposal.text, screening, config.threshold)

        proposal.near_duplicates = first_labels
        if matches:
            proposal.duplicate_after_retry = True
            self.stats["duplicates_after_retry"] += 1
            if config.on_duplicate == "skip":
                logger.info(
                    "Proposal for component %r is still a near duplicate after %d "
                    "diversify call(s); leaving it out.", name, calls,
                )
                return None
        return proposal

    def _screening_set(self, name: str, parent_text: str) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        if "rejected" in self.dedupe.against:
            for i, text in enumerate(self.journal.rejected_texts(name), 1):
                items.append((f"rejected entry {i}", text))
        if "pool" in self.dedupe.against:
            items.append(("the current instruction", parent_text))
            for i, candidate in enumerate(self._pool, 1):
                text = candidate.get(name)
                if text is not None and text != parent_text:
                    items.append((f"candidate {i}", text))
        return items

    def _render_journal(self) -> str:
        if not self.journal_enabled:
            return "None"
        return self.journal.render(limit=self.journal_entries)

    def _render_skills(self) -> str:
        return render_skills(self.skills)

    # -- Length budget ------------------------------------------------------

    def _length_limit_text(self) -> str:
        parts = []
        if self.max_words is not None:
            parts.append(f"at most {self.max_words} words")
        if self.max_tokens is not None:
            parts.append(f"at most {self.max_tokens} tokens")
        return (
            "The new instruction must be " + " and ".join(parts) + "."
            if parts
            else "None"
        )

    def _enforce_length(self, text: str) -> str:
        if self._within_budget(text):
            return text
        # One compression attempt with the LM.
        try:
            kwargs = dict(instruction=text, length_limit=self._length_limit_text())
            shortened = self._run(self.module.compress, **kwargs).shortened_instruction.strip()
            if shortened and self._within_budget(shortened):
                return shortened
            text = shortened or text
        except Exception:
            logger.exception("Compression pass failed; falling back to truncation.")
        return self._truncate(text)

    def _within_budget(self, text: str) -> bool:
        if self.max_words is not None and _count_words(text) > self.max_words:
            return False
        if self.max_tokens is not None and _count_tokens(text, self._model_name()) > self.max_tokens:
            return False
        return True

    def _truncate(self, text: str) -> str:
        words = _WORD_RE.findall(text)
        limit = len(words)
        if self.max_words is not None:
            limit = min(limit, self.max_words)
        if self.max_tokens is not None:
            model = self._model_name()
            # Binary search the largest word-prefix that fits the token budget.
            lo, hi = 0, limit
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _count_tokens(" ".join(words[:mid]), model) <= self.max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            limit = lo
        truncated = " ".join(words[:limit])
        logger.warning(
            "Proposed instruction exceeded the length budget even after "
            "compression; hard-truncated to %d words.", limit,
        )
        return truncated

    def _model_name(self) -> str | None:
        lm = self.prompt_model or getattr(dspy.settings, "lm", None)
        return getattr(lm, "model", None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _proposal_from(pred: Any) -> Proposal:
    return Proposal(
        text=pred.new_instruction.strip(),
        change_summary=(getattr(pred, "change_summary", "") or "").strip(),
    )


def _count_tokens(text: str, model: str | None) -> int:
    if model:
        try:
            import litellm

            return litellm.token_counter(model=model, text=text)
        except Exception:
            pass
    return max(1, round(len(text) / 4))  # rough fallback: ~4 chars/token


def _render_examples(examples: Sequence[Mapping[str, Any]]) -> str:
    """Render GEPA's reflective dataset entries as markdown.

    Entries typically look like {"Inputs": ..., "Generated Outputs": ...,
    "Feedback": ...}, but arbitrary keys and nesting are handled.

    TODO(v0.2): rich types (e.g. dspy.Image) are stringified here. For
    vision programs, pass them through as multimodal content instead — see
    DSPy's MultiModalInstructionProposer for the pattern.
    """
    if not examples:
        return "No examples were provided."

    def render_value(value: Any, level: int = 3) -> str:
        if isinstance(value, Mapping):
            out = ""
            for k, v in value.items():
                out += f"{'#' * level} {k}\n{render_value(v, min(level + 1, 6))}"
            return out or "\n"
        if isinstance(value, (list, tuple)):
            out = ""
            for i, item in enumerate(value, 1):
                out += f"{'#' * level} Item {i}\n{render_value(item, min(level + 1, 6))}"
            return out or "\n"
        return f"{str(value).strip()}\n\n"

    blocks = []
    for i, example in enumerate(examples, 1):
        block = f"# Example {i}\n"
        for key, value in example.items():
            block += f"## {key}\n{render_value(value)}"
        blocks.append(block)
    return "\n\n".join(blocks)
