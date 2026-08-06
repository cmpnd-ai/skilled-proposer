import pytest

pytest.importorskip("dspy.predict.flex")

from dspy.teleprompt.gepa import gepa_utils

from skilled_proposer import patch


@pytest.fixture(autouse=True)
def clean_patch_state():
    yield
    patch.uninstall_code_proposer()


def _stub_proposer(calls):
    def proposer(candidate, reflective_dataset, components_to_update,
                 task_descriptions, context_blurbs):
        calls.append(
            {
                "candidate": candidate,
                "reflective_dataset": reflective_dataset,
                "components_to_update": components_to_update,
                "task_descriptions": task_descriptions,
                "context_blurbs": context_blurbs,
            }
        )
        return {k: "class New(dspy.Module):\n    pass" for k in components_to_update}

    return proposer


def test_install_and_uninstall_restores_original():
    original = gepa_utils.propose_code
    patch.install_code_proposer(_stub_proposer([]))
    assert gepa_utils.propose_code is not original
    patch.uninstall_code_proposer()
    assert gepa_utils.propose_code is original


def test_double_install_raises():
    patch.install_code_proposer(_stub_proposer([]))
    with pytest.raises(RuntimeError, match="already installed"):
        patch.install_code_proposer(_stub_proposer([]))


def test_uninstall_without_install_is_noop():
    original = gepa_utils.propose_code
    patch.uninstall_code_proposer()
    assert gepa_utils.propose_code is original


def test_wrapper_maps_builtin_args_to_contract():
    calls = []
    with patch.use_code_proposer(_stub_proposer(calls)):
        result = gepa_utils.propose_code(
            code_keys=["flex_step"],
            candidate={"flex_step": "class Old(dspy.Module):\n    pass"},
            reflective_dataset={"flex_step": [{"Inputs": "x"}]},
            task_descriptions={"flex_step": "Signature: q -> a"},
            context_blurbs={"flex_step": "(no extra context)"},
            reflection_lm=None,
        )
    assert result == {"flex_step": "class New(dspy.Module):\n    pass"}
    assert calls[0]["components_to_update"] == ["flex_step"]
    assert calls[0]["candidate"] == {"flex_step": "class Old(dspy.Module):\n    pass"}
    assert calls[0]["task_descriptions"] == {"flex_step": "Signature: q -> a"}
    assert calls[0]["context_blurbs"] == {"flex_step": "(no extra context)"}


def test_wrapper_sets_reflection_lm_context():
    import dspy
    from dspy.utils.dummies import DummyLM

    seen = {}

    def proposer(candidate, reflective_dataset, components_to_update,
                 task_descriptions, context_blurbs):
        seen["lm"] = dspy.settings.lm
        return dict.fromkeys(components_to_update, "class New(dspy.Module):\n    pass")

    lm = DummyLM([{"revised_source": "unused"}])
    with patch.use_code_proposer(proposer):
        gepa_utils.propose_code(
            code_keys=["flex_step"],
            candidate={"flex_step": "class Old(dspy.Module):\n    pass"},
            reflective_dataset={},
            task_descriptions={},
            context_blurbs={},
            reflection_lm=lm,
        )
    assert seen["lm"] is lm


def test_context_manager_uninstalls_on_error():
    original = gepa_utils.propose_code
    with pytest.raises(RuntimeError, match="boom"):
        with patch.use_code_proposer(_stub_proposer([])):
            raise RuntimeError("boom")
    assert gepa_utils.propose_code is original


def test_drift_check_raises_on_unexpected_signature(monkeypatch):
    def wrong_shape(a, b):  # pragma: no cover - never called
        return {}

    monkeypatch.setattr(gepa_utils, "propose_code", wrong_shape)
    with pytest.raises(RuntimeError, match="cannot patch"):
        patch.install_code_proposer(_stub_proposer([]))
