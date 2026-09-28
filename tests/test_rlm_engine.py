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


def test_rlm_engine_needs_deno(monkeypatch):
    monkeypatch.setattr(proposer_module, "_deno_error", lambda: "no deno here")
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


def test_auto_resolving_to_rlm_without_deno_raises(monkeypatch):
    monkeypatch.setattr(proposer_module, "_deno_error", lambda: "no deno here")
    proposer = SkilledProposer(engine="auto", rlm_threshold=50)
    with pytest.raises(RuntimeError, match=r"skilled-proposer\[rlm\]"):
        proposer._resolve_engine("Do it.", [{"Inputs": "word " * 400}])
