"""Runs dspy.GEPA end to end with stub LMs. No network."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledProposer


class StudentLM(DummyLM):
    """Answers correctly only when the instruction contains GOOD."""

    def __init__(self):
        super().__init__([])

    def forward(self, prompt=None, messages=None, **kwargs):
        system = messages[0]["content"] if messages else (prompt or "")
        self.answers = iter([{"answer": "gold" if "GOOD" in system else "bad"}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ok = (pred.answer or "").strip() == gold.answer
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback="ok" if ok else f"expected {gold.answer}")


@pytest.fixture
def data():
    train = [dspy.Example(question=f"q{i}", answer="gold").with_inputs("question") for i in range(4)]
    val = [dspy.Example(question=f"v{i}", answer="gold").with_inputs("question") for i in range(3)]
    return train, val


@pytest.fixture(autouse=True)
def student_lm():
    dspy.configure(lm=StudentLM())
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    yield


def run_gepa(proposer, reflection_lm, data, max_metric_calls=30):
    train, val = data
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        instruction_proposer=proposer,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=2,
        num_threads=1,
        use_merge=False,
        seed=0,
        gepa_kwargs=proposer.gepa_kwargs(),
    )
    return optimizer.compile(Program(), trainset=train, valset=val)


def test_journal_appears_from_second_iteration(data, tmp_path):
    reflection = DummyLM([
        {"new_instruction": "meh instruction", "change_summary": "Reworded the task."},
        {"new_instruction": "GOOD instruction", "change_summary": "Added the GOOD rule."},
    ] + [{"new_instruction": "GOOD again", "change_summary": "x"}] * 10)
    proposer = SkilledProposer(journal=True, journal_path=tmp_path / "journal.json", distill_every=None)

    optimized = run_gepa(proposer, reflection, data)

    assert "GOOD" in optimized.predict.signature.instructions
    entries = proposer.journal.entries
    assert len(entries) >= 2
    assert entries[0].accepted is False and "not better" in entries[0].reason
    assert entries[0].parent_idx == 0 and entries[0].minibatch_before == 0.0
    assert entries[1].accepted is True and entries[1].valset_average == 1.0
    assert all(e.closed for e in entries)

    first_prompt = reflection.history[0]["messages"][-1]["content"]
    second_prompt = reflection.history[1]["messages"][-1]["content"]
    assert "No proposals have been recorded yet." in first_prompt
    assert "### Iteration 1, component `predict`, rejected" in second_prompt
    assert "Change: Reworded the task." in second_prompt
    assert (tmp_path / "journal.json").exists()


def test_dedupe_runs_inside_gepa(data):
    reflection = DummyLM([
        {"new_instruction": "Given the fields `question`, produce the fields `answer`!"},
        {"new_instruction": "GOOD instruction"},
    ] + [{"new_instruction": "GOOD instruction"}] * 10)
    proposer = SkilledProposer(dedupe=True)

    optimized = run_gepa(proposer, reflection, data)

    assert "GOOD" in optimized.predict.signature.instructions
    assert proposer.stats["duplicates"] >= 1
    diversify_prompt = reflection.history[1]["messages"][-1]["content"]
    assert "near_duplicates" in diversify_prompt
