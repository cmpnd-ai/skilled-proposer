import json

import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledProposer
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


class FakeState:
    def __init__(self, candidates, trace):
        self.program_candidates = candidates
        self.full_program_trace = [trace]


def run_iteration(proposer, iteration, parent, new_text, accepted, reflect_lm_answers=None):
    """Drive the callbacks GEPA fires for one single-task iteration."""
    proposer.on_candidate_selected(
        {"iteration": iteration, "candidate_idx": 0, "candidate": {"predict": parent}, "score": 0.5}
    )
    out = proposer(
        candidate={"predict": parent},
        reflective_dataset={"predict": [{"Inputs": "x", "Feedback": "wrong"}]},
        components_to_update=["predict"],
    )
    assert out == {"predict": new_text}
    if accepted:
        proposer.on_valset_evaluated(
            {"iteration": iteration, "candidate_idx": 1, "candidate": {"predict": new_text},
             "average_score": 0.8, "is_best_program": True}
        )
        proposer.on_candidate_accepted(
            {"iteration": iteration, "new_candidate_idx": 1, "new_score": 2.0, "parent_ids": [0]}
        )
        trace = {"tasks": [{"parent_idx": 0, "subsample_scores": [0.0, 1.0], "new_subsample_scores": [1.0, 1.0]}]}
        candidates = [{"predict": parent}, {"predict": new_text}]
    else:
        proposer.on_candidate_rejected(
            {"iteration": iteration, "old_score": 1.0, "new_score": 1.0, "reason": "not better"}
        )
        trace = {"tasks": [{"parent_idx": 0, "subsample_scores": [0.0, 1.0], "new_subsample_scores": [0.0, 1.0]}]}
        candidates = [{"predict": parent}]
    proposer.on_iteration_end(
        {"iteration": iteration, "state": FakeState(candidates, trace), "proposal_accepted": accepted}
    )


def test_journal_args_validation(tmp_path):
    with pytest.raises(ValueError):
        SkilledProposer(journal_path=tmp_path / "j.json")
    with pytest.raises(ValueError):
        SkilledProposer(journal=True, journal_entries=0)
    with pytest.raises(ValueError):
        SkilledProposer(journal=True, distill_every=0)


def test_journal_off_by_default_renders_none():
    lm = DummyLM([{"new_instruction": "New."}])
    proposer = SkilledProposer(prompt_model=lm)
    proposer(candidate={"a": "old"}, reflective_dataset={"a": []}, components_to_update=["a"])
    prompt = lm.history[-1]["messages"][-1]["content"]
    assert "proposal_journal" in prompt
    assert "No proposals have been recorded yet." not in prompt


def test_rejected_then_accepted_entries(tmp_path):
    lm = DummyLM([
        {"new_instruction": "meh", "change_summary": "Reworded."},
        {"new_instruction": "GOOD", "change_summary": "Added a rule."},
    ])
    proposer = SkilledProposer(prompt_model=lm, journal=True, journal_path=tmp_path / "j.json", distill_every=None)
    run_iteration(proposer, 1, "seed", "meh", accepted=False)
    run_iteration(proposer, 2, "seed", "GOOD", accepted=True)

    first, second = proposer.journal.entries
    assert first.iteration == 1 and first.task == 0 and first.component == "predict"
    assert first.parent_idx == 0 and first.parent_text == "seed" and first.proposed_text == "meh"
    assert first.change_summary == "Reworded."
    assert first.accepted is False and first.reason == "not better"
    assert first.minibatch_before == 0.5 and first.minibatch_after == 0.5
    assert first.valset_average is None and first.closed

    assert second.accepted is True and second.valset_average == 0.8
    assert second.minibatch_before == 0.5 and second.minibatch_after == 1.0
    assert second.change_summary == "Added a rule."

    second_prompt = lm.history[1]["messages"][-1]["content"]
    assert "### Iteration 1, component `predict`, rejected" in second_prompt
    assert "Change: Reworded." in second_prompt

    saved = Journal.load(tmp_path / "j.json")
    assert [e.proposed_text for e in saved.entries] == ["meh", "GOOD"]


def test_two_tasks_in_one_iteration_pair_by_position():
    lm = DummyLM([
        {"new_instruction": "p1", "change_summary": "a"},
        {"new_instruction": "p2", "change_summary": "b"},
    ])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=None)
    for _ in range(2):
        proposer.on_candidate_selected(
            {"iteration": 1, "candidate_idx": 0, "candidate": {"predict": "seed"}, "score": 0.0}
        )
    proposer(candidate={"predict": "seed"}, reflective_dataset={"predict": []}, components_to_update=["predict"])
    proposer(candidate={"predict": "seed"}, reflective_dataset={"predict": []}, components_to_update=["predict"])
    proposer.on_candidate_rejected({"iteration": 1, "old_score": 0, "new_score": 0, "reason": "first"})
    proposer.on_valset_evaluated({"iteration": 1, "candidate_idx": 1, "candidate": {"predict": "p2"},
                                  "average_score": 0.9, "is_best_program": True})
    proposer.on_candidate_accepted({"iteration": 1, "new_candidate_idx": 1, "new_score": 2, "parent_ids": [0]})
    trace = {"tasks": [
        {"parent_idx": 0, "subsample_scores": [0.0], "new_subsample_scores": [0.0]},
        {"parent_idx": 0, "subsample_scores": [0.0], "new_subsample_scores": [1.0]},
    ]}
    proposer.on_iteration_end({"iteration": 1, "state": FakeState([{"predict": "seed"}, {"predict": "p2"}], trace),
                               "proposal_accepted": True})
    a, b = proposer.journal.entries
    assert (a.task, a.accepted, a.reason) == (0, False, "first")
    assert (b.task, b.accepted, b.valset_average, b.minibatch_after) == (1, True, 0.9, 1.0)


def test_merge_iteration_and_seed_valset_event_are_tolerated():
    proposer = SkilledProposer(journal=True, distill_every=None)
    proposer.on_valset_evaluated({"iteration": 0, "candidate_idx": 0, "candidate": {"predict": "seed"},
                                  "average_score": 0.1, "is_best_program": True})
    proposer.on_iteration_end({"iteration": 1, "state": FakeState([{"predict": "seed"}], {"tasks": []}),
                               "proposal_accepted": False})
    assert proposer.journal.entries == []


def test_skipped_iteration_leaves_no_entry():
    proposer = SkilledProposer(journal=True, distill_every=None)
    proposer.on_candidate_selected({"iteration": 1, "candidate_idx": 0, "candidate": {"predict": "seed"}, "score": 1.0})
    proposer.on_iteration_end({"iteration": 1, "state": FakeState([{"predict": "seed"}], {"tasks": [{"parent_idx": 0}]}),
                               "proposal_accepted": False})
    assert proposer.journal.entries == []


def test_unregistered_callbacks_warn_once_and_still_record(caplog):
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}, {"new_instruction": "b", "change_summary": "y"}])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=None)
    with caplog.at_level("WARNING"):
        proposer(candidate={"p": "seed"}, reflective_dataset={"p": []}, components_to_update=["p"])
        proposer(candidate={"p": "seed"}, reflective_dataset={"p": []}, components_to_update=["p"])
    assert sum("gepa_kwargs" in r.message for r in caplog.records) == 1
    assert [e.iteration for e in proposer.journal.entries] == [1, 2]
    assert proposer.journal.entries[0].parent_idx is None


def test_gepa_kwargs_registers_self():
    proposer = SkilledProposer()
    kwargs = proposer.gepa_kwargs()
    assert kwargs == {"callbacks": [proposer]}
    other = object()
    merged = proposer.gepa_kwargs(callbacks=[other], reflection_minibatch_size=5)
    assert merged["callbacks"] == [proposer, other]
    assert merged["reflection_minibatch_size"] == 5


def test_journal_loads_from_existing_path(tmp_path):
    path = tmp_path / "j.json"
    j = Journal(lessons="old lesson")
    j.open(entry(iteration=4, accepted=False))
    j.close(4)
    j.save(path)
    proposer = SkilledProposer(journal=True, journal_path=path)
    assert proposer.journal.lessons == "old lesson"
    assert len(proposer.journal.entries) == 1


def test_distill_runs_after_enough_closed_entries():
    lm = DummyLM([
        {"new_instruction": "a", "change_summary": "x"},
        {"new_instruction": "b", "change_summary": "y"},
        {"lessons": "Short rules were accepted."},
        {"new_instruction": "c", "change_summary": "z"},
    ])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=2)
    run_iteration(proposer, 1, "seed", "a", accepted=False)
    assert proposer.journal.lessons == ""
    run_iteration(proposer, 2, "seed", "b", accepted=True)
    assert proposer.journal.lessons == "Short rules were accepted."
    assert proposer.journal.closed_since_distill == 0
    run_iteration(proposer, 3, "b", "c", accepted=False)
    third_prompt = lm.history[-1]["messages"][-1]["content"]
    assert "## Lessons" in third_prompt
    assert "Short rules were accepted." in third_prompt


def test_distill_failure_keeps_prior_lessons(caplog):
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}])
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=1)
    proposer.journal.lessons = "keep me"

    def boom(**kwargs):
        raise RuntimeError("distill exploded")

    proposer.module.distill = boom
    with caplog.at_level("WARNING"):
        run_iteration(proposer, 1, "seed", "a", accepted=False)
    assert proposer.journal.lessons == "keep me"
    assert proposer.journal.closed_since_distill == 0
    assert any("distill" in r.message.lower() for r in caplog.records)


def test_distill_off_when_none():
    lm = DummyLM([{"new_instruction": "a", "change_summary": "x"}] * 3)
    proposer = SkilledProposer(prompt_model=lm, journal=True, distill_every=None)
    for i in range(1, 4):
        run_iteration(proposer, i, "seed", "a", accepted=False)
    assert proposer.journal.lessons == ""
    assert proposer.journal.closed_since_distill == 3
