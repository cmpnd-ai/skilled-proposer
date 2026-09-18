import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer.proposer import SkilledProposer


def batch_lm(*batches):
    """A stub reflection model that answers each call with one batch of candidates."""
    return DummyLM([
        {"new_instructions": list(texts), "change_summaries": [f"route {t}" for t in texts]}
        for texts in batches
    ])


def call(proposer, parent="old", component="a"):
    return proposer(
        candidate={component: parent},
        reflective_dataset={component: [{"Inputs": "x", "Feedback": "wrong"}]},
        components_to_update=[component],
    )


def test_candidates_validation():
    with pytest.raises(ValueError):
        SkilledProposer(candidates=0)
    with pytest.raises(ValueError):
        SkilledProposer(candidates=-2)
    assert SkilledProposer(candidates=None).candidates is None
    assert SkilledProposer(candidates=1).candidates is None
    assert SkilledProposer(candidates=3).candidates == 3


def test_candidates_off_leaves_single_path():
    proposer = SkilledProposer()
    assert proposer.candidates is None
    assert proposer.module.propose_many is None
    assert proposer._cache == {}


def test_propose_batch_returns_proposals_in_order():
    proposer = SkilledProposer(candidates=3, prompt_model=batch_lm(["one", "two", "three"]))
    proposals = proposer._propose_batch("a", "old", [])
    assert [p.text for p in proposals] == ["one", "two", "three"]
    assert [p.change_summary for p in proposals] == ["route one", "route two", "route three"]


def test_propose_batch_passes_candidate_count():
    lm = batch_lm(["one", "two", "three"])
    SkilledProposer(candidates=3, prompt_model=lm)._propose_batch("a", "old", [])
    prompt = str(lm.history[0]["messages"])
    assert "candidate_count" in prompt
    assert "3" in prompt


def test_propose_batch_drops_empty_parent_and_duplicate_candidates():
    lm = batch_lm(["", "old", "one", " one ", "two"])
    proposals = SkilledProposer(candidates=5, prompt_model=lm)._propose_batch("a", "old", [])
    assert [p.text for p in proposals] == ["one", "two"]


def test_propose_batch_missing_summary_becomes_empty():
    lm = DummyLM([{"new_instructions": ["one", "two"], "change_summaries": ["r1"]}])
    proposals = SkilledProposer(candidates=2, prompt_model=lm)._propose_batch("a", "old", [])
    assert [p.change_summary for p in proposals] == ["r1", ""]


def test_propose_batch_raises_when_nothing_usable():
    lm = batch_lm(["", "old"])
    with pytest.raises(ValueError):
        SkilledProposer(candidates=2, prompt_model=lm)._propose_batch("a", "old", [])


def test_propose_batch_enforces_length_per_candidate():
    long_text = " ".join(["word"] * 20)
    lm = DummyLM([
        {"new_instructions": ["Short one.", long_text], "change_summaries": ["a", "b"]},
        {"shortened_instruction": "Short two."},
    ])
    proposer = SkilledProposer(candidates=2, prompt_model=lm, max_words=5)
    proposals = proposer._propose_batch("a", "old", [])
    assert [p.text for p in proposals] == ["Short one.", "Short two."]
