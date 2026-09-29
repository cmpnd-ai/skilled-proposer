import pytest

import skilled_proposer.proposer as proposer_module
from skilled_proposer import SkilledProposer

from rlm_support import ExecInterpreter


def test_engine_args_validation(tmp_path):
    with pytest.raises(ValueError):
        SkilledProposer(engine="fast")
    with pytest.raises(ValueError):
        SkilledProposer(review="everything")
    with pytest.raises(ValueError):
        SkilledProposer(review="seen")  # engine="predict" cannot review seen
    with pytest.raises(ValueError):
        SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter, seen_path=tmp_path / "s.json")
    for bad in ({"rlm_threshold": 0}, {"max_iters": 0}, {"max_llm_calls": 0}):
        with pytest.raises(ValueError):
            SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter, **bad)
    with pytest.raises(ValueError, match="compaction"):
        SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter, compaction=True)


def test_compaction_is_predict_only():
    # engine="predict" (the default) accepts compaction.
    predict_proposer = SkilledProposer(compaction=True)
    assert predict_proposer.compaction is not None

    # engine="rlm" rejects it outright rather than silently ignoring it.
    with pytest.raises(ValueError, match="compaction"):
        SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter, compaction=True)


def test_rlm_engine_needs_monty(monkeypatch):
    monkeypatch.setattr(proposer_module, "_monty_error", lambda: "no monty here")
    with pytest.raises(RuntimeError, match=r"skilled-proposer\[rlm\]"):
        SkilledProposer(engine="rlm")
    with pytest.raises(RuntimeError):
        SkilledProposer(engine="auto", review="seen")
    SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter)
    SkilledProposer(engine="auto")  # decided on the first call


def test_predict_engine_builds_no_rlm_and_no_store():
    proposer = SkilledProposer()
    assert proposer._engine == "predict"
    assert not hasattr(proposer.module, "rlm")
    assert proposer.store is None


def test_seen_review_builds_a_store(tmp_path):
    proposer = SkilledProposer(engine="rlm", review="seen", interpreter_factory=ExecInterpreter,
                               seen_path=tmp_path / "s.json")
    assert proposer.store is not None and proposer.store.records == {}
    assert proposer.module.rlm is not None


def test_auto_picks_predict_for_small_prompts():
    proposer = SkilledProposer(engine="auto", rlm_threshold=10_000, interpreter_factory=ExecInterpreter)
    assert proposer._resolve_engine("Do it.", [{"Feedback": "wrong"}]) == "predict"


def test_auto_picks_rlm_for_large_prompts():
    proposer = SkilledProposer(engine="auto", rlm_threshold=50, interpreter_factory=ExecInterpreter)
    big = [{"Inputs": "word " * 400, "Feedback": "wrong"}]
    assert proposer._resolve_engine("Do it.", big) == "rlm"


def test_auto_with_no_records_resolves_to_predict():
    proposer = SkilledProposer(engine="auto", rlm_threshold=10_000, interpreter_factory=ExecInterpreter)
    assert proposer._resolve_engine("Do it.", []) == "predict"


def test_auto_with_seen_is_rlm_from_the_start():
    proposer = SkilledProposer(engine="auto", review="seen", interpreter_factory=ExecInterpreter)
    assert proposer._engine == "rlm"


def test_engine_choice_is_locked_after_first_resolution():
    proposer = SkilledProposer(engine="auto", rlm_threshold=50, interpreter_factory=ExecInterpreter)
    assert proposer._resolve_engine("Do it.", [{"Inputs": "word " * 400}]) == "rlm"
    assert proposer._resolve_engine("Do it.", []) == "rlm"


def test_auto_resolving_to_rlm_without_monty_raises(monkeypatch):
    monkeypatch.setattr(proposer_module, "_monty_error", lambda: "no monty here")
    proposer = SkilledProposer(engine="auto", rlm_threshold=50)
    with pytest.raises(RuntimeError, match=r"skilled-proposer\[rlm\]"):
        proposer._resolve_engine("Do it.", [{"Inputs": "word " * 400}])


from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMError

from skilled_proposer.skill import Skill

from rlm_support import scripted, step


def rlm_proposer(lm, **kw):
    return SkilledProposer(engine="rlm", interpreter_factory=ExecInterpreter, prompt_model=lm, **kw)


def call(proposer, candidate, dataset):
    return proposer(candidate=candidate, reflective_dataset=dataset, components_to_update=list(candidate))


def prompts(lm):
    return [h["messages"][0]["content"] + h["messages"][-1]["content"] for h in lm.history]


def test_rlm_proposal_explores_records_and_submits():
    lm = scripted(
        "print(len(examples['records']), examples['records'][0]['tags']['component'])",
        "SUBMIT(new_instruction=current_instruction + ' Rule.', change_summary='1 of 1 failures.')",
    )
    out = call(rlm_proposer(lm), {"p": "Do it."}, {"p": [{"Inputs": {"q": "x"}, "Feedback": "wrong"}]})
    assert out == {"p": "Do it. Rule."}
    assert "list of 1 record dicts" in prompts(lm)[0]
    assert "1 p" in prompts(lm)[1]
    assert not any("examples_with_feedback" in p for p in prompts(lm))


def test_rlm_sees_skill_files():
    lm = scripted("SUBMIT(new_instruction=skill_files['s/models/openai.md'], change_summary='x')")
    skill = Skill(name="s", content="Main.", files={"models/openai.md": "Use XML tags."})
    out = call(rlm_proposer(lm, skills=[skill]), {"p": "Do it."}, {"p": []})
    assert out == {"p": "Use XML tags."}


def test_rlm_with_no_records_still_proposes():
    lm = scripted("SUBMIT(new_instruction='N' + str(len(examples['records'])), change_summary='x')")
    out = call(rlm_proposer(lm), {"p": "Do it."}, {"p": []})
    assert out == {"p": "N0"}
    assert "list of 0 record dicts" in prompts(lm)[0]


def test_non_json_record_values_reach_the_sandbox_as_text():
    class Thing:
        def __str__(self):
            return "thing"

    lm = scripted("SUBMIT(new_instruction=examples['records'][0]['Inputs']['img'], change_summary='x')")
    out = call(rlm_proposer(lm), {"p": "Do it."}, {"p": [{"Inputs": {"img": Thing()}, "Feedback": "wrong"}]})
    assert out == {"p": "thing"}


def test_journal_notes_carry_forward_and_distillation_is_skipped():
    lm = scripted(
        "SUBMIT(new_instruction='A', change_summary='x', "
        "journal_notes={'lessons': ['L1'], 'hypotheses': [], 'entry_analysis': {}})",
        "eid = journal['entries'][0]['id']\n"
        "SUBMIT(new_instruction=journal['lessons'][0] + ' B', change_summary='y', "
        "journal_notes={'lessons': ['L2'], 'hypotheses': ['H'], 'entry_analysis': {eid: 'Kept.'}})",
    )
    proposer = rlm_proposer(lm, journal=True, distill_every=1)
    assert call(proposer, {"p": "seed"}, {"p": []}) == {"p": "A"}
    assert call(proposer, {"p": "A"}, {"p": []}) == {"p": "L1 B"}
    assert proposer.journal.lessons == "L2"
    assert proposer.journal.hypotheses == ["H"]
    first = proposer.journal.entries[0]
    assert first.analysis == "Kept." and first.accepted is True
    assert len(lm.history) == 2  # no distillation call


def test_two_components_get_their_own_records_and_share_notes():
    lm = scripted(
        "SUBMIT(new_instruction='A' + str(len(examples['records'])), change_summary='x', "
        "journal_notes={'lessons': ['from a'], 'hypotheses': [], 'entry_analysis': {}})",
        "SUBMIT(new_instruction=journal['lessons'][0] + ' ' + str(len(examples['records'])), "
        "change_summary='y', journal_notes={'lessons': ['from b'], 'hypotheses': [], 'entry_analysis': {}})",
    )
    proposer = rlm_proposer(lm, journal=True, distill_every=None)
    dataset = {"a": [{"Feedback": "1"}, {"Feedback": "2"}], "b": [{"Feedback": "3"}]}
    out = call(proposer, {"a": "old a", "b": "old b"}, dataset)
    assert out == {"a": "A2", "b": "from a 1"}


def test_seen_review_shows_earlier_records_with_relations(tmp_path):
    lm = scripted(
        "SUBMIT(new_instruction='A' + str(len(seen['records'])), change_summary='x')",
        "r = seen['records'][0]\n"
        "t = r['tags']\n"
        "SUBMIT(new_instruction=':'.join([r['Feedback'], t['relation'], str(t['edits_back']), "
        "seen['instructions'][t['instruction']]['text'], str(len(examples['records']))]), change_summary='y')",
    )
    path = tmp_path / "seen.json"
    proposer = rlm_proposer(lm, review="seen", seen_path=path)
    assert call(proposer, {"p": "seed"}, {"p": [{"Feedback": "s1"}]}) == {"p": "A0"}
    assert call(proposer, {"p": "A0"}, {"p": [{"Feedback": "a1"}]}) == {"p": "s1:ancestor:1:seed:1"}
    assert path.exists()
    reloaded = rlm_proposer(scripted(), review="seen", seen_path=path)
    assert [r["Feedback"] for r in reloaded.store.records["p"]] == ["s1", "a1"]


def test_minibatch_review_passes_an_empty_seen():
    lm = scripted("SUBMIT(new_instruction=str(len(seen['records'])), change_summary='x')")
    assert call(rlm_proposer(lm), {"p": "Do it."}, {"p": [{"Feedback": "wrong"}]}) == {"p": "0"}
    assert "`seen` is empty this run." in prompts(lm)[0]


def out_of_iterations_lm():
    """One step that never calls SUBMIT, then the reply for dspy's extract step."""
    return DummyLM([step("print('thinking')"), {"new_instruction": "Extracted.", "change_summary": "y"}])


def test_extract_fallback_is_a_failure():
    with pytest.raises(RuntimeError, match="max_iters"):
        call(rlm_proposer(out_of_iterations_lm(), max_iters=1, on_error="raise"), {"p": "Do it."}, {"p": []})


def test_extract_fallback_with_skip_leaves_the_component_out():
    proposer = rlm_proposer(out_of_iterations_lm(), max_iters=1, retries=0)
    assert call(proposer, {"p": "Do it."}, {"p": []}) == {}


def test_empty_instruction_is_a_failure():
    lm = scripted("SUBMIT(new_instruction='   ', change_summary='x')")
    with pytest.raises(ValueError, match="empty"):
        call(rlm_proposer(lm, on_error="raise"), {"p": "Do it."}, {"p": []})


def test_retries_rerun_the_rlm_never_predict():
    lm = scripted(
        "SUBMIT(new_instruction='', change_summary='x')",
        "SUBMIT(new_instruction='Second try.', change_summary='y')",
    )
    out = call(rlm_proposer(lm, retries=1), {"p": "Do it."}, {"p": []})
    assert out == {"p": "Second try."}
    assert not any("examples_with_feedback" in p for p in prompts(lm))


def test_lm_error_propagates_from_the_rlm():
    proposer = rlm_proposer(scripted())

    def boom(**kwargs):
        raise LMError("provider down")

    proposer.module.rlm = boom
    with pytest.raises(LMError):
        call(proposer, {"p": "Do it."}, {"p": []})


def test_resumed_run_continues_iterations_from_saved_state(tmp_path):
    submit = "SUBMIT(new_instruction='A' + str(len(seen['records'])), change_summary='x', " \
             "journal_notes={'lessons': [], 'hypotheses': [], 'entry_analysis': {}})"
    paths = dict(journal=True, journal_path=tmp_path / "j.json", review="seen",
                 seen_path=tmp_path / "s.json", distill_every=None)
    first = rlm_proposer(scripted(submit, submit), **paths)
    call(first, {"p": "seed"}, {"p": [{"Feedback": "r1"}]})
    call(first, {"p": "seed"}, {"p": [{"Feedback": "r2"}]})

    resumed = rlm_proposer(scripted(submit), **paths)
    out = call(resumed, {"p": "seed"}, {"p": [{"Feedback": "r3"}]})

    assert out == {"p": "A2"}  # both saved records are visible on the first resumed call
    ids = [e.entry_id for e in resumed.journal.entries]
    assert ids == ["1:p", "2:p", "3:p"]


def test_auto_without_monty_warns_at_construction(monkeypatch, caplog):
    monkeypatch.setattr(proposer_module, "_monty_error", lambda: "no monty here")
    SkilledProposer(engine="auto")
    assert "dspy-monty-interpreter" in caplog.text and "skilled-proposer[rlm]" in caplog.text


def test_auto_with_an_interpreter_factory_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(proposer_module, "_monty_error", lambda: "no monty here")
    SkilledProposer(engine="auto", interpreter_factory=ExecInterpreter)
    assert "dspy-monty-interpreter" not in caplog.text
