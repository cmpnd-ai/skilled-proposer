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
