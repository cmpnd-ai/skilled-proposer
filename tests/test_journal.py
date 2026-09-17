import json

import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledProposer
from skilled_proposer.journal import Journal, JournalEntry


def entry(iteration=1, component="predict", **kw):
    base = dict(
        iteration=iteration,
        component=component,
        parent_text="old text",
        proposed_text="new text",
    )
    base.update(kw)
    return JournalEntry(**base)


def propose(proposer, parent, component="p"):
    return proposer(candidate={component: parent}, reflective_dataset={component: []},
                    components_to_update=[component])


# -- Data model --------------------------------------------------------------

def test_entry_verdict_wording():
    assert entry().verdict == "not chosen as a parent so far"
    assert entry(accepted=True).verdict == "accepted"


def test_open_entries_and_close():
    j = Journal()
    j.open(entry(iteration=1, component="a"))
    j.open(entry(iteration=1, component="b"))
    j.open(entry(iteration=2, component="a"))
    assert [e.component for e in j.open_entries(1)] == ["a", "b"]
    assert j.close(1) == 2
    assert j.open_entries(1) == []
    assert len(j.open_entries(2)) == 1
    assert j.closed_since_distill == 2
    assert j.close(3) == 0


def test_json_round_trip_keeps_everything():
    j = Journal(lessons="Keep rules short.", closed_since_distill=3)
    j.open(entry(accepted=True, reason="Chosen as the parent in iteration 2."))
    j.close(1)
    text = j.to_json()
    assert json.loads(text)["lessons"] == "Keep rules short."
    back = Journal.from_json(text)
    assert back.lessons == "Keep rules short."
    assert back.closed_since_distill == 4
    assert back.entries == j.entries


def test_from_json_ignores_fields_from_older_files():
    text = json.dumps({"entries": [{"iteration": 1, "component": "p", "parent_text": "a",
                                    "proposed_text": "b", "valset_average": 0.5, "task": 0}]})
    j = Journal.from_json(text)
    assert j.entries[0].proposed_text == "b"


def test_save_and_load(tmp_path):
    path = tmp_path / "journal.json"
    j = Journal()
    j.open(entry())
    j.save(path)
    assert Journal.load(path).entries == j.entries
    assert Journal.load(tmp_path / "missing.json").entries == []


# -- Rendering ---------------------------------------------------------------

def test_render_empty():
    assert Journal().render(limit=5) == "No proposals have been recorded yet."


def test_render_lessons_first_then_entries():
    j = Journal(lessons="Short rules win.")
    j.open(entry(iteration=3, accepted=True, reason="Chosen as the parent in iteration 4.",
                 proposed_text="one two three four", parent_text="one two", change_summary="Added two words."))
    text = j.render(limit=5)
    assert text.index("## Lessons") < text.index("## Recent proposals")
    assert "Short rules win." in text
    assert "### Iteration 3, component `predict`, accepted" in text
    assert "Change: Added two words." in text
    assert "Size: 4 words, 2 more than the parent." in text
    assert "Reason: Chosen as the parent in iteration 4." in text
    assert "new text" not in text


def test_render_unchosen_entry():
    j = Journal()
    j.open(entry(proposed_text="a b", parent_text="a b c"))
    text = j.render()
    assert "not chosen as a parent so far" in text
    assert "Change: No change summary was given." in text
    assert "Size: 2 words, 1 fewer than the parent." in text
    assert "Reason:" not in text


def test_render_limit_keeps_newest():
    j = Journal()
    for i in range(1, 6):
        j.open(entry(iteration=i))
    text = j.render(limit=2)
    assert "Iteration 4," in text and "Iteration 5," in text
    assert "Iteration 3," not in text


# -- Proposer wiring ---------------------------------------------------------

def test_journal_args_validation(tmp_path):
    with pytest.raises(ValueError):
        SkilledProposer(journal_path=tmp_path / "j.json")
    with pytest.raises(ValueError):
        SkilledProposer(journal=True, journal_entries=0)
    with pytest.raises(ValueError):
        SkilledProposer(journal=True, distill_every=0)


def test_journal_off_renders_none_in_prompt():
    lm = DummyLM([{"new_instruction": "New."}])
    proposer = SkilledProposer(prompt_model=lm)
    propose(proposer, "old", component="a")
    prompt = lm.history[-1]["messages"][-1]["content"]
    assert "[[ ## proposal_journal ## ]]\nNone" in prompt


def test_lineage_marks_parents_accepted():
    lm = DummyLM([
        {"new_instruction": "a", "change_summary": "x"},
        {"new_instruction": "b", "change_summary": "y"},
        {"new_instruction": "c", "change_summary": "z"},
    ])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=None)
    propose(proposer, "seed")
    propose(proposer, "a")
    propose(proposer, "a")
    first, second, third = proposer.journal.entries
    assert [e.iteration for e in (first, second, third)] == [1, 2, 3]
    assert first.accepted is True and first.reason == "Chosen as the parent in iteration 2."
    assert second.accepted is None and third.accepted is None
    assert first.closed and second.closed and not third.closed
    third_prompt = lm.history[2]["messages"][-1]["content"]
    assert "### Iteration 1, component `p`, accepted" in third_prompt
    assert "Reason: Chosen as the parent in iteration 2." in third_prompt
    assert "### Iteration 2, component `p`, not chosen as a parent so far" in third_prompt


def test_first_prompt_says_nothing_recorded():
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}])
    proposer = SkilledProposer(prompt_model=lm, journal=True)
    propose(proposer, "seed")
    assert "No proposals have been recorded yet." in lm.history[0]["messages"][-1]["content"]


def test_distill_runs_after_enough_closed_entries(tmp_path):
    lm = DummyLM([
        {"new_instruction": "a", "change_summary": "x"},
        {"new_instruction": "b", "change_summary": "y"},
        {"lessons": "Short rules win."},
        {"new_instruction": "c", "change_summary": "z"},
    ])
    path = tmp_path / "j.json"
    proposer = SkilledProposer(prompt_model=lm, journal=True, journal_path=path, distill_every=2)
    propose(proposer, "seed")
    propose(proposer, "a")
    assert proposer.journal.lessons == ""
    propose(proposer, "a")
    assert proposer.journal.lessons == "Short rules win."
    assert proposer.journal.closed_since_distill == 0
    third_prompt = lm.history[-1]["messages"][-1]["content"]
    assert "## Lessons" in third_prompt and "Short rules win." in third_prompt
    saved = Journal.load(path)
    assert saved.lessons == "Short rules win."
    assert [e.proposed_text for e in saved.entries] == ["a", "b", "c"]


def test_distill_failure_keeps_prior_lessons(caplog):
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}, {"new_instruction": "b", "change_summary": "y"}])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=1)
    proposer.journal.lessons = "keep me"

    def boom(**kwargs):
        raise RuntimeError("distill exploded")

    proposer.module.distill = boom
    with caplog.at_level("WARNING"):
        propose(proposer, "seed")
        propose(proposer, "a")
    assert proposer.journal.lessons == "keep me"
    assert proposer.journal.closed_since_distill == 0
    assert any("distill" in r.message.lower() for r in caplog.records)


def test_distill_off_when_none():
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}] * 3)
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=None)
    for parent in ("seed", "a", "a"):
        propose(proposer, parent)
    assert proposer.journal.lessons == ""
    assert proposer.journal.closed_since_distill == 2


def test_journal_loads_from_existing_path(tmp_path):
    path = tmp_path / "j.json"
    j = Journal(lessons="old lesson")
    j.open(entry(iteration=4, accepted=True))
    j.close(4)
    j.save(path)
    proposer = SkilledProposer(journal=True, journal_path=path)
    assert proposer.journal.lessons == "old lesson"
    assert len(proposer.journal.entries) == 1


def test_bad_journal_path_logs_and_continues(tmp_path, caplog):
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}])
    proposer = SkilledProposer(prompt_model=lm, journal=True, journal_path=tmp_path / "missing" / "j.json")
    with caplog.at_level("WARNING"):
        assert propose(proposer, "seed") == {"p": "a"}
    assert any("Failed to save the journal" in r.message for r in caplog.records)
