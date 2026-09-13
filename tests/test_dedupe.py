import pytest

from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledProposer
from skilled_proposer.dedupe import DedupeConfig, find_duplicates, render_duplicates, similarity


def test_config_defaults():
    c = DedupeConfig()
    assert c.threshold == 0.85
    assert c.max_retries == 1
    assert c.against == ("rejected", "pool")
    assert c.on_duplicate == "return"


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0.0},
    {"threshold": 1.5},
    {"max_retries": -1},
    {"against": ()},
    {"against": ("batch",)},
    {"on_duplicate": "loop"},
])
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        DedupeConfig(**kwargs)


def test_identical_and_whitespace_variants_score_one():
    assert similarity("Answer with the name only.", "answer   with the name only.") == 1.0


def test_reordered_sentences_score_high_through_jaccard():
    a = "State the task. Return only the name. Drop the prefix."
    b = "Drop the prefix. State the task. Return only the name."
    assert similarity(a, b) >= 0.85


def test_one_word_change_scores_high():
    a = "Return only the committee name from the disclaimer, without any prefix or address."
    b = "Return only the committee name from the disclaimer, without any preface or address."
    assert similarity(a, b) >= 0.85


def test_different_approach_scores_low():
    a = "Return only the committee name from the disclaimer."
    b = "First list every organization mentioned. Then pick the one after 'Paid for by'. Output that string."
    assert similarity(a, b) < 0.85


def test_empty_cases():
    assert similarity("", "") == 1.0
    assert similarity("", "x") == 0.0


def test_find_duplicates_sorted_and_filtered():
    text = "Return only the committee name from the disclaimer."
    matches = find_duplicates(text, [
        ("candidate 1", "Return only the committee name from the disclaimer!"),
        ("rejected entry 2", "Write a poem about the email."),
        ("candidate 3", "Return only the committee name from the disclaimer, nothing else."),
    ], threshold=0.85)
    assert [m[0] for m in matches] == ["candidate 1", "candidate 3"]
    assert matches[0][2] >= matches[1][2]


def test_render_duplicates_lists_label_and_text():
    out = render_duplicates([("candidate 1", "text one", 0.9), ("rejected entry 2", "text two", 0.88)])
    assert "1. candidate 1 (similarity 0.90)\ntext one" in out
    assert "2. rejected entry 2 (similarity 0.88)\ntext two" in out


PARENT = "Return only the committee name from the disclaimer."
NEAR = "Return only the committee name from the disclaimer!"
FAR = "First list every organization mentioned. Then pick the one after 'Paid for by'. Output that string."


def propose(proposer, parent=PARENT):
    return proposer(candidate={"p": parent}, reflective_dataset={"p": []}, components_to_update=["p"])


def test_dedupe_arg_forms():
    assert SkilledProposer().dedupe is None
    assert SkilledProposer(dedupe=True).dedupe == DedupeConfig()
    cfg = DedupeConfig(threshold=0.5)
    assert SkilledProposer(dedupe=cfg).dedupe is cfg
    with pytest.raises(ValueError):
        SkilledProposer(dedupe="yes")


def test_no_match_returns_first_proposal_with_one_call():
    lm = DummyLM([{"new_instruction": FAR}])
    proposer = SkilledProposer(prompt_model=lm, dedupe=True)
    assert propose(proposer) == {"p": FAR}
    assert len(lm.history) == 1
    assert proposer.stats == {"duplicates": 0, "duplicates_after_retry": 0}


def test_parent_near_duplicate_triggers_one_diversify_call():
    lm = DummyLM([{"new_instruction": NEAR}, {"new_instruction": FAR}])
    proposer = SkilledProposer(prompt_model=lm, dedupe=True)
    assert propose(proposer) == {"p": FAR}
    assert len(lm.history) == 2
    diversify_prompt = lm.history[1]["messages"][-1]["content"]
    assert "the current instruction" in diversify_prompt
    assert NEAR in diversify_prompt
    assert proposer.stats == {"duplicates": 1, "duplicates_after_retry": 0}


def test_retry_cap_holds_and_returns_by_default():
    lm = DummyLM([{"new_instruction": NEAR}] * 10)
    proposer = SkilledProposer(prompt_model=lm, dedupe=DedupeConfig(max_retries=2))
    assert propose(proposer) == {"p": NEAR}
    assert len(lm.history) == 3
    assert proposer.stats == {"duplicates": 1, "duplicates_after_retry": 1}


def test_zero_retries_screens_but_never_diversifies():
    lm = DummyLM([{"new_instruction": NEAR}] * 3)
    proposer = SkilledProposer(prompt_model=lm, dedupe=DedupeConfig(max_retries=0))
    assert propose(proposer) == {"p": NEAR}
    assert len(lm.history) == 1
    assert proposer.stats["duplicates_after_retry"] == 1


def test_on_duplicate_skip_drops_component():
    lm = DummyLM([{"new_instruction": NEAR}] * 3)
    proposer = SkilledProposer(prompt_model=lm, dedupe=DedupeConfig(on_duplicate="skip"))
    assert propose(proposer) == {}
    assert len(lm.history) == 2


def test_screens_against_pool_snapshot_and_rejected_journal():
    lm = DummyLM([
        {"new_instruction": "Return the bare committee name only.", "change_summary": "a"},
        {"new_instruction": "Return the bare committee name only!", "change_summary": "b"},
        {"new_instruction": FAR, "change_summary": "c"},
    ])
    proposer = SkilledProposer(prompt_model=lm, journal=True, dedupe=True, distill_every=None)
    # Iteration 1 proposes a short instruction and GEPA rejects it.
    proposer.on_candidate_selected({"iteration": 1, "candidate_idx": 0, "candidate": {"p": PARENT}, "score": 0})
    propose(proposer)
    proposer.on_candidate_rejected({"iteration": 1, "old_score": 0, "new_score": 0, "reason": "tie"})

    class State:
        program_candidates = [{"p": PARENT}, {"p": "some other candidate text"}]
        full_program_trace = [{"tasks": [{"subsample_scores": [0.0], "new_subsample_scores": [0.0]}]}]

    proposer.on_iteration_end({"iteration": 1, "state": State(), "proposal_accepted": False})
    screening = dict(proposer._screening_set("p", PARENT))
    assert screening["rejected entry 1"] == "Return the bare committee name only."
    assert screening["the current instruction"] == PARENT
    assert screening["candidate 2"] == "some other candidate text"

    # Iteration 2 proposes a near copy of the rejected entry, then diversifies.
    proposer.on_candidate_selected({"iteration": 2, "candidate_idx": 0, "candidate": {"p": PARENT}, "score": 0})
    assert propose(proposer) == {"p": FAR}
    entry = proposer.journal.entries[-1]
    assert entry.near_duplicates == ["rejected entry 1"]
    assert entry.duplicate_after_retry is False


def test_against_rejected_only_ignores_parent():
    lm = DummyLM([{"new_instruction": NEAR}])
    proposer = SkilledProposer(prompt_model=lm, dedupe=DedupeConfig(against=("rejected",)))
    assert propose(proposer) == {"p": NEAR}
    assert len(lm.history) == 1


def test_diversified_text_gets_length_enforced():
    long_far = FAR + " " + " ".join(["extra"] * 40)
    lm = DummyLM([
        {"new_instruction": NEAR},
        {"new_instruction": long_far},
        {"shortened_instruction": FAR},
    ])
    proposer = SkilledProposer(prompt_model=lm, dedupe=True, max_words=30)
    assert propose(proposer) == {"p": FAR}
