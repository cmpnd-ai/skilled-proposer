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
    idx = prompt.rindex("candidate_count")
    assert "3" in prompt[idx : idx + 80]


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


def test_propose_batch_drops_candidates_that_collapse_after_length_enforcement():
    lm = DummyLM([
        {
            "new_instructions": ["one two three four", "one two three five"],
            "change_summaries": ["a", "b"],
        },
        {"shortened_instruction": "one two three"},
        {"shortened_instruction": "one two three"},
    ])
    proposer = SkilledProposer(candidates=2, prompt_model=lm, max_words=3)
    proposals = proposer._propose_batch("a", "old", [])
    assert [p.text for p in proposals] == ["one two three"]


def test_batch_serves_candidates_across_calls_then_regenerates():
    lm = batch_lm(["one", "two", "three"], ["four", "five", "six"])
    proposer = SkilledProposer(candidates=3, prompt_model=lm)
    assert [call(proposer)["a"] for _ in range(4)] == ["one", "two", "three", "four"]
    assert len(lm.history) == 2


def test_parent_change_replaces_cache():
    lm = batch_lm(["one", "two", "three"], ["four", "five", "six"])
    proposer = SkilledProposer(candidates=3, prompt_model=lm)
    assert call(proposer, parent="old")["a"] == "one"
    assert call(proposer, parent="one")["a"] == "four"
    assert call(proposer, parent="one")["a"] == "five"
    assert len(lm.history) == 2
    parent, leftovers = proposer._cache["a"]
    assert parent == "one"
    assert [p.text for p in leftovers] == ["six"]


def test_short_batch_serves_what_it_has():
    lm = batch_lm(["one", "two"], ["three", "four", "five"])
    proposer = SkilledProposer(candidates=3, prompt_model=lm)
    assert [call(proposer)["a"] for _ in range(3)] == ["one", "two", "three"]
    assert len(lm.history) == 2


def test_round_robin_components_get_separate_batches():
    """The default round robin component selector sends one component at a
    time. Each component then starts its own batch, and the served texts
    do not condition on each other the way they would under
    component_selector="all"."""
    lm = batch_lm(["a1", "a2"], ["b1", "b2"])
    proposer = SkilledProposer(candidates=2, prompt_model=lm)
    parent = {"a": "old a", "b": "old b"}
    served = [
        proposer(
            candidate=parent,
            reflective_dataset={"a": [], "b": []},
            components_to_update=components,
        )
        for components in (["a"], ["b"], ["a"])
    ]
    assert [r[c] for r, c in zip(served, ["a", "b", "a"])] == ["a1", "b1", "a2"]
    assert len(lm.history) == 2


def test_cache_is_per_component():
    lm = batch_lm(["a1", "a2"], ["b1", "b2"])
    proposer = SkilledProposer(candidates=2, prompt_model=lm)
    both = dict(candidate={"a": "old a", "b": "old b"}, reflective_dataset={"a": [], "b": []},
                components_to_update=["a", "b"])
    assert proposer(**both) == {"a": "a1", "b": "b1"}
    assert proposer(**both) == {"a": "a2", "b": "b2"}
    assert len(lm.history) == 2


def test_single_path_makes_one_call_per_component():
    lm = DummyLM([{"new_instruction": "New."}])
    proposer = SkilledProposer(prompt_model=lm)
    assert call(proposer)["a"] == "New."
    assert proposer._cache == {}


def test_unusable_batch_follows_on_error_policy():
    same = {"new_instructions": ["old"], "change_summaries": ["no change"]}
    proposer = SkilledProposer(candidates=2, prompt_model=DummyLM([same, same]))
    assert call(proposer) == {}
    proposer = SkilledProposer(candidates=2, prompt_model=DummyLM([same]), on_error="raise")
    with pytest.raises(ValueError):
        call(proposer)


def test_journal_opens_one_entry_per_served_candidate():
    lm = batch_lm(["one", "two"], ["three", "four"])
    proposer = SkilledProposer(candidates=2, journal=True, prompt_model=lm)
    call(proposer, parent="old")
    call(proposer, parent="old")
    entries = proposer.journal.entries
    assert [(e.iteration, e.proposed_text, e.change_summary) for e in entries] == [
        (1, "one", "route one"),
        (2, "two", "route two"),
    ]
    call(proposer, parent="two")
    assert entries[1].accepted is True
    assert entries[0].accepted is None
    assert entries[2].proposed_text == "three"
    assert len(lm.history) == 2
