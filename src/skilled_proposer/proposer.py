"""SkilledProposer: a custom GEPA instruction proposer for DSPy.

Implements GEPA's `ProposalFn` protocol (gepa.core.adapter.ProposalFn), so it
plugs directly into `dspy.GEPA(instruction_proposer=...)`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import dspy

try:
    from dspy.utils.exceptions import LMError
except ImportError:  # dspy < 3.x fallback: nothing raises it
    class LMError(Exception):
        pass

from skilled_proposer.journal import Journal, JournalEntry
from skilled_proposer.signatures import InstructionProposalModule, rlm_signature
from skilled_proposer.skill import Skill, render_skills
from skilled_proposer.sandbox import SandboxJSON, summarize_journal, summarize_records
from skilled_proposer.store import SeenStore, tag_records

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")

# dspy.RLM's final_reasoning when it ran out of iterations and pulled an
# answer out of the transcript instead of the model calling SUBMIT.
_EXTRACT_FALLBACK = "Extract forced final output"


@dataclass
class Proposal:
    """One component's proposal and the reflection model's summary of it."""

    text: str
    change_summary: str = ""


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


def _monty_error() -> str | None:
    """Why the default RLM sandbox (Monty) is unavailable here, or None when it's fine."""
    try:
        import dspy_monty_interpreter  # noqa: F401
    except ImportError as e:
        return str(e)
    return None


def _require_monty() -> None:
    error = _monty_error()
    if error is not None:
        raise RuntimeError(
            'The RLM engine needs dspy-monty-interpreter. Install it with '
            f'`pip install "skilled-proposer[rlm]"`, or pass interpreter_factory. ({error})'
        )


def _default_interpreter_factory():
    from dspy_monty_interpreter import MontyInterpreter

    return MontyInterpreter


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
        max_words: Cap on the proposed instruction length in words. Defaults
            to 1000; pass None to remove the cap.
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
        journal: Experimental. Record every proposal, the reflection model's
            summary of what it changed, and whether GEPA later chose it as a
            parent, and show that record to the reflection model on each
            call. A proposal that GEPA hands back as the parent to improve
            was accepted; one that never comes back is shown as not chosen.
        journal_path: Optional JSON file for the journal. Loaded when it
            exists and written after every iteration, so a resumed run keeps
            its record. Requires journal=True.
        journal_entries: How many recent entries render into the prompt.
        distill_every: Closed entries between lesson distillations. None
            turns distillation off. Ignored when journal is off.
        engine: "predict" (default) proposes with one dspy.Predict call.
            "rlm" runs each proposal as a dspy.RLM that analyzes the
            records with code in a sandbox. "auto" uses RLM when
            review="seen" or when the first call's prompt is at least
            rlm_threshold tokens, and Predict otherwise. The choice is
            fixed for the run.
        review: "minibatch" (default) shows the RLM this call's records.
            "seen" also shows every record from earlier calls, tagged by
            how the instruction that produced it relates to the current
            one. Requires engine="rlm" or "auto".
        seen_path: Optional JSON file for the seen store. Requires
            review="seen".
        rlm_threshold: Prompt size in tokens at which engine="auto"
            chooses RLM.
        sub_lm: LM for the RLM's llm_query calls. Defaults to the
            reflection LM.
        max_iters: RLM REPL iterations per proposal.
        max_llm_calls: Cap on the RLM's sub-LM calls per proposal.
        interpreter_factory: Zero-argument callable returning a dspy
            CodeInterpreter. None uses `dspy-monty-interpreter`'s Rust
            sandbox (the `rlm` extra). Pass dspy's own `PythonInterpreter`
            for the Deno/Pyodide sandbox instead (the `rlm-deno` extra).
        seed: Seeds the order of seen records.
    """

    def __init__(
        self,
        skills: Sequence[Skill | str | Path] | None = None,
        additional_instructions: str | None = None,
        base_instructions: str | None = None,
        max_tokens: int | None = None,
        max_words: int | None = 1000,
        prompt_model: "dspy.LM | None" = None,
        max_examples: int | None = None,
        retries: int = 1,
        on_error: str = "skip",
        *,
        journal: bool = False,
        journal_path: str | Path | None = None,
        journal_entries: int = 12,
        distill_every: int | None = 5,
        engine: str = "predict",
        review: str = "minibatch",
        seen_path: str | Path | None = None,
        rlm_threshold: int = 30_000,
        sub_lm: "dspy.LM | None" = None,
        max_iters: int = 20,
        max_llm_calls: int = 50,
        interpreter_factory=None,
        seed: int = 0,
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
        if engine not in ("predict", "rlm", "auto"):
            raise ValueError('engine must be "predict", "rlm", or "auto"')
        if review not in ("minibatch", "seen"):
            raise ValueError('review must be "minibatch" or "seen"')
        if review == "seen" and engine == "predict":
            raise ValueError('review="seen" needs engine="rlm" or engine="auto"')
        if seen_path is not None and review != "seen":
            raise ValueError('seen_path requires review="seen"')
        if rlm_threshold <= 0 or max_iters <= 0 or max_llm_calls <= 0:
            raise ValueError("rlm_threshold, max_iters, and max_llm_calls must be positive")

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

        # All LM-facing predictors live on a dspy.Module.
        self.module = InstructionProposalModule(
            self.base_instructions, with_change_summary=journal
        )

        self.engine = engine
        self.review = review
        self.rlm_threshold = rlm_threshold
        self.max_iters = max_iters
        self.interpreter_factory = interpreter_factory
        self.seed = seed
        self.seen_path = Path(seen_path) if seen_path is not None else None
        self.store = None
        if review == "seen":
            self.store = SeenStore.load(self.seen_path) if self.seen_path else SeenStore()
        # Each proposal call counts as one journal iteration. A resumed run
        # continues numbering after the saved journal and store, so entry ids
        # stay unique and earlier records stay visible in `seen`.
        self._iteration = max(
            [e.iteration for e in self.journal.entries]
            + [r["tags"]["iteration"] for rs in (self.store.records if self.store else {}).values() for r in rs]
            + [0]
        )

        # The engine is fixed for the whole run. "auto" without seen review
        # is decided on the first call, from the size of its prompt.
        self._engine: str | None = {"predict": "predict", "rlm": "rlm"}.get(engine)
        if engine == "auto" and review == "seen":
            self._engine = "rlm"
        if self._engine == "rlm" and interpreter_factory is None:
            _require_monty()
        if self._engine is None and interpreter_factory is None:
            # GEPA catches proposer errors, so an "auto" run that picks RLM
            # without Monty would fail every call without stopping. Say so now.
            error = _monty_error()
            if error is not None:
                logger.warning(
                    'engine="auto" may choose RLM on the first call, but dspy-monty-interpreter '
                    "is not available, so every proposal would fail. Install it with "
                    '`pip install "skilled-proposer[rlm]"` or use engine="predict". (%s)',
                    error,
                )
        if engine != "predict":
            self.module.rlm = dspy.RLM(
                rlm_signature(self.base_instructions, journal=journal, seen=review == "seen"),
                max_iters=max_iters,
                max_llm_calls=max_llm_calls,
                sub_lm=sub_lm,
                interpreter_factory=interpreter_factory or _default_interpreter_factory(),
            )

    # -- ProposalFn ---------------------------------------------------------

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        **kwargs: Any,
    ) -> dict[str, str]:
        self._iteration += 1
        if self.journal_enabled:
            self._advance_journal(candidate)
        if self._engine is None and components_to_update:
            first = components_to_update[0]
            self._resolve_engine(candidate[first], self._examples(first, reflective_dataset))

        results: dict[str, str] = {}
        seen_batches: list[tuple[str, str, list[dict]]] = []
        for name in components_to_update:
            current = candidate[name]
            examples = self._examples(name, reflective_dataset)
            if self._engine == "rlm":
                records = tag_records(
                    examples, component=name, iteration=self._iteration, instruction=current
                )
                if self.store is not None:
                    seen_batches.append((name, current, records))
                propose = lambda: self._propose_rlm(name, current, records)  # noqa: E731
            else:
                propose = lambda: self._propose_one(name, current, examples)  # noqa: E731
            proposal = _attempt(
                propose,
                name=name,
                retries=self.retries,
                on_error=self.on_error,
                label="SkilledProposer",
            )
            if proposal is None:
                continue
            results[name] = proposal.text
            if self.store is not None:
                self.store.link(name, current, proposal.text)
            if self.journal_enabled:
                self.journal.open(
                    JournalEntry(
                        iteration=self._iteration,
                        component=name,
                        parent_text=current,
                        proposed_text=proposal.text,
                        change_summary=proposal.change_summary,
                    )
                )
        # Stored after every component is proposed, so this call's records
        # show up only in `examples`, never in `seen`.
        if self.store is not None:
            for name, current, records in seen_batches:
                self.store.add(name, current, records)
            self._save_store()
        if self.journal_enabled:
            self._save_journal()
        return results

    # -- Journal ------------------------------------------------------------

    def _advance_journal(self, candidate: Mapping[str, str]) -> None:
        """Mark proposals that came back as parents accepted, close the
        previous iteration's entries, and distill when due."""
        for e in self.journal.entries:
            if e.accepted is None and candidate.get(e.component) == e.proposed_text:
                e.accepted = True
                e.reason = f"Chosen as the parent in iteration {self._iteration}."
        self.journal.close(self._iteration - 1)
        if (
            self.distill_every
            and self._engine != "rlm"
            and self.journal.closed_since_distill >= self.distill_every
        ):
            self._distill()

    def _save_journal(self) -> None:
        """Write the journal to journal_path. Logs and never raises on failure."""
        if self.journal_path is None:
            return
        try:
            self.journal.save(self.journal_path)
        except Exception:
            logger.warning("Failed to save the journal to %s.", self.journal_path, exc_info=True)

    def _save_store(self) -> None:
        """Write the seen store to seen_path. Logs and never raises on failure."""
        if self.seen_path is None:
            return
        try:
            self.store.save(self.seen_path)
        except Exception:
            logger.warning("Failed to save the seen store to %s.", self.seen_path, exc_info=True)

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

    # -- Internals ----------------------------------------------------------

    def _run(self, predictor, **kwargs) -> dspy.Prediction:
        if self.prompt_model is not None:
            with dspy.context(lm=self.prompt_model):
                return predictor(**kwargs)
        return predictor(**kwargs)

    def _examples(self, name: str, reflective_dataset) -> list:
        examples = list(reflective_dataset.get(name, []))
        return examples[: self.max_examples] if self.max_examples is not None else examples

    def _predict_inputs(self, current_instruction: str, examples) -> dict[str, str]:
        return dict(
            current_instruction=current_instruction,
            examples_with_feedback=_render_examples(examples),
            proposal_journal=self._render_journal(),
            reference_skills=self._render_skills(),
            additional_guidance=self.additional_instructions or "None",
            length_limit=self._length_limit_text(),
        )

    def _resolve_engine(self, current_instruction: str, examples) -> str:
        """Pick the engine once. Every later call reuses it."""
        if self._engine is not None:
            return self._engine
        prompt = "\n\n".join(self._predict_inputs(current_instruction, examples).values())
        size = _count_tokens(prompt, self._model_name())
        engine = "rlm" if size >= self.rlm_threshold else "predict"
        if engine == "rlm" and self.interpreter_factory is None:
            _require_monty()
        logger.info("SkilledProposer engine: %s (first prompt %d tokens).", engine, size)
        self._engine = engine
        return engine

    def _propose_one(
        self, name: str, current_instruction: str, examples: Sequence[Mapping[str, Any]]
    ) -> Proposal | None:
        pred = self._run(self.module, **self._predict_inputs(current_instruction, examples))
        proposal = _proposal_from(pred)
        proposal.text = self._enforce_length(proposal.text)
        return proposal

    def _propose_rlm(self, name: str, current_instruction: str, records: list[dict]) -> Proposal:
        """One proposal through dspy.RLM. Raises on any unusable result."""
        examples = {"records": records}
        inputs = dict(
            current_instruction=current_instruction,
            examples=SandboxJSON(examples, summarize_records("examples", examples)),
            seen=self._seen_input(name, current_instruction),
            skill=self._render_skills(),
            skill_files=self._skill_files(),
            additional_guidance=self.additional_instructions or "None",
            length_limit=self._length_limit_text(),
        )
        if self.journal_enabled:
            data = self.journal.to_data()
            inputs["journal"] = SandboxJSON(data, summarize_journal(data))
        pred = self._run(self.module.rlm, **inputs)
        if getattr(pred, "final_reasoning", "") == _EXTRACT_FALLBACK:
            raise RuntimeError(f"RLM reached max_iters={self.max_iters} without calling SUBMIT.")
        text = (pred.new_instruction or "").strip()
        if not text:
            raise ValueError("RLM returned an empty instruction.")
        proposal = Proposal(
            text=self._enforce_length(text),
            change_summary=(pred.change_summary or "").strip(),
        )
        if self.journal_enabled:
            notes = pred.journal_notes
            self.journal.apply_notes(notes.lessons, notes.hypotheses, notes.entry_analysis)
        return proposal

    def _seen_input(self, name: str, current_instruction: str) -> SandboxJSON:
        if self.store is None:
            return SandboxJSON(
                {"instructions": {}, "records": []},
                "`seen` is empty: this run reviews only the current minibatch.",
            )
        view = self.store.view(name, current_instruction, self._iteration, self.seed)
        return SandboxJSON(view, summarize_records("seen", view))

    def _skill_files(self) -> dict[str, str]:
        return {f"{s.name}/{path}": text for s in self.skills for path, text in s.files.items()}

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
