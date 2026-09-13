import json

import pytest

from skilled_proposer.journal import Journal, JournalEntry


def entry(iteration=1, task=0, component="predict", **kw):
    base = dict(
        iteration=iteration,
        task=task,
        component=component,
        parent_text="old text",
        proposed_text="new text",
    )
    base.update(kw)
    return JournalEntry(**base)


def test_entry_verdict_wording():
    assert entry().verdict == "not evaluated"
    assert entry(accepted=True).verdict == "accepted"
    assert entry(accepted=False).verdict == "rejected"


def test_open_entries_and_tasks_keep_order():
    j = Journal()
    j.open(entry(iteration=1, task=0, component="a"))
    j.open(entry(iteration=1, task=0, component="b"))
    j.open(entry(iteration=1, task=1, component="a"))
    j.open(entry(iteration=2, task=0, component="a"))
    assert [e.component for e in j.open_entries(1)] == ["a", "b", "a"]
    assert j.open_tasks(1) == [0, 1]
    assert [e.component for e in j.entries_for_task(1, 0)] == ["a", "b"]


def test_close_marks_entries_and_counts():
    j = Journal()
    j.open(entry(iteration=1))
    j.open(entry(iteration=1, task=1))
    j.open(entry(iteration=2))
    assert j.close(1) == 2
    assert j.open_entries(1) == []
    assert len(j.open_entries(2)) == 1
    assert j.closed_since_distill == 2
    assert j.close(3) == 0


def test_rejected_texts_only_closed_rejected_for_component():
    j = Journal()
    j.open(entry(iteration=1, component="a", proposed_text="r1", accepted=False))
    j.open(entry(iteration=1, component="b", proposed_text="r2", accepted=False))
    j.open(entry(iteration=1, component="a", proposed_text="ok", accepted=True))
    j.open(entry(iteration=2, component="a", proposed_text="open", accepted=False))
    j.close(1)
    assert j.rejected_texts("a") == ["r1"]


def test_json_round_trip_keeps_everything():
    j = Journal(lessons="Keep rules short.", closed_since_distill=3)
    j.open(entry(near_duplicates=["candidate 1"], duplicate_after_retry=True, accepted=False, reason="tie"))
    j.close(1)
    text = j.to_json()
    data = json.loads(text)
    assert data["lessons"] == "Keep rules short."
    back = Journal.from_json(text)
    assert back.lessons == "Keep rules short."
    assert back.closed_since_distill == 4
    assert back.entries == j.entries


def test_save_and_load(tmp_path):
    path = tmp_path / "journal.json"
    j = Journal()
    j.open(entry())
    j.save(path)
    assert Journal.load(path).entries == j.entries


def test_load_missing_file_is_empty(tmp_path):
    j = Journal.load(tmp_path / "missing.json")
    assert j.entries == []
    assert j.lessons == ""


def test_render_empty():
    assert Journal().render(limit=5) == "No proposals have been recorded yet."


def test_render_lessons_first_then_entries():
    j = Journal(lessons="Short rules win.")
    j.open(entry(iteration=3, accepted=False, reason="tie", minibatch_before=0.5, minibatch_after=0.5,
                 proposed_text="one two three four", parent_text="one two", change_summary="Added two words."))
    text = j.render(limit=5)
    assert text.index("## Lessons") < text.index("## Recent proposals")
    assert "Short rules win." in text
    assert "### Iteration 3, component `predict`, rejected" in text
    assert "Minibatch 0.50 -> 0.50. Not on the valset." in text
    assert "Change: Added two words." in text
    assert "Size: 4 words, 2 more than the parent." in text
    assert "Reason: tie" in text
    assert "new text" not in text


def test_render_accepted_entry_shows_valset():
    j = Journal()
    j.open(entry(accepted=True, minibatch_before=0.2, minibatch_after=0.6, valset_average=0.55,
                 proposed_text="a b", parent_text="a b c"))
    text = j.render()
    assert "accepted" in text
    assert "Valset average 0.55." in text
    assert "Size: 2 words, 1 fewer than the parent." in text
    assert "Change: No change summary was given." in text


def test_render_not_evaluated_entry_has_no_scores():
    j = Journal()
    j.open(entry(proposed_text="x y", parent_text="x y"))
    text = j.render()
    assert "not evaluated" in text
    assert "Minibatch" not in text
    assert "Size: 2 words, the same as the parent." in text


def test_render_duplicate_flags():
    j = Journal()
    j.open(entry(near_duplicates=["candidate 1", "rejected entry 2"], duplicate_after_retry=True))
    text = j.render()
    assert "Near duplicate of candidate 1, rejected entry 2. Still a near duplicate after a rewrite." in text


def test_render_limit_keeps_newest():
    j = Journal()
    for i in range(1, 6):
        j.open(entry(iteration=i))
    text = j.render(limit=2)
    assert "Iteration 4," in text
    assert "Iteration 5," in text
    assert "Iteration 3," not in text
    assert "Iteration 1," not in text
