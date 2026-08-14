import pytest

from skilled_proposer.code_proposer import (
    _primitives_catalog,
    _strip_code_fences,
    _validate_module_source,
)

VALID_SRC = '''class GeneratedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.Predict("question -> answer")

    def forward(self, **inputs):
        return dspy.Prediction(answer=self.step(**inputs).answer)
'''


def test_strip_code_fences_plain_text_unchanged():
    assert _strip_code_fences(VALID_SRC) == VALID_SRC


def test_strip_code_fences_removes_python_fence():
    fenced = f"```python\n{VALID_SRC}\n```"
    assert _strip_code_fences(fenced).strip() == VALID_SRC.strip()


def test_strip_code_fences_removes_bare_fence():
    fenced = f"```\n{VALID_SRC}\n```\n"
    assert _strip_code_fences(fenced).strip() == VALID_SRC.strip()


def test_validate_accepts_module_source():
    assert _validate_module_source(VALID_SRC) is None


def test_validate_rejects_non_python():
    error = _validate_module_source("Here is my revised module: use two steps.")
    assert error is not None
    assert "parse" in error


def test_validate_rejects_source_without_class():
    error = _validate_module_source("x = 1\n\ndef forward(**inputs):\n    return x")
    assert error is not None
    assert "class" in error


def test_validate_rejects_class_without_forward():
    src = "class Old(dspy.Module):\n    def __init__(self):\n        super().__init__()\n"
    error = _validate_module_source(src)
    assert error is not None
    assert "forward" in error


def test_primitives_catalog_available():
    pytest.importorskip("dspy.predict.flex")
    catalog = _primitives_catalog()
    assert isinstance(catalog, str) and catalog.strip()


from dspy.utils.dummies import DummyLM

from skilled_proposer.code_proposer import SkilledCodeProposer
from skilled_proposer.skill import Skill


class _Result:
    def __init__(self, revised_source):
        self.revised_source = revised_source


def _call(proposer, source="class Old(dspy.Module):\n    pass"):
    return proposer(
        candidate={"flex_step": source},
        reflective_dataset={"flex_step": [{"Inputs": "x", "Feedback": "wrong"}]},
        components_to_update=["flex_step"],
        task_descriptions={"flex_step": "Signature: question -> answer"},
        context_blurbs={"flex_step": "(no extra context)"},
    )


def test_init_validation():
    with pytest.raises(ValueError):
        SkilledCodeProposer(on_error="explode")


def test_proposes_revised_source_per_component():
    pytest.importorskip("dspy.predict.flex")
    lm = DummyLM([{"revised_source": VALID_SRC}])
    out = _call(SkilledCodeProposer(prompt_model=lm))
    assert "class GeneratedModule" in out["flex_step"]


def test_fenced_response_is_stripped():
    pytest.importorskip("dspy.predict.flex")
    lm = DummyLM([{"revised_source": f"```python\n{VALID_SRC}\n```"}])
    out = _call(SkilledCodeProposer(prompt_model=lm))
    assert not out["flex_step"].startswith("```")
    assert "class GeneratedModule" in out["flex_step"]


def test_invalid_source_kept_by_default():
    pytest.importorskip("dspy.predict.flex")
    lm = DummyLM([{"revised_source": "not python at all ((("}])
    out = _call(SkilledCodeProposer(prompt_model=lm), source="class Old: pass")
    assert out["flex_step"] == "class Old: pass"


def test_invalid_source_raises_when_asked():
    pytest.importorskip("dspy.predict.flex")
    lm = DummyLM([{"revised_source": "not python at all ((("}])
    with pytest.raises(ValueError, match="parse"):
        _call(SkilledCodeProposer(prompt_model=lm, on_error="raise"))


def test_source_without_forward_kept_by_default():
    pytest.importorskip("dspy.predict.flex")
    no_forward = "class Old(dspy.Module):\n    def __init__(self):\n        super().__init__()\n"
    lm = DummyLM([{"revised_source": no_forward}])
    out = _call(SkilledCodeProposer(prompt_model=lm), source="class Old: pass")
    assert out["flex_step"] == "class Old: pass"


def test_source_without_forward_raises_when_asked():
    pytest.importorskip("dspy.predict.flex")
    no_forward = "class Old(dspy.Module):\n    def __init__(self):\n        super().__init__()\n"
    lm = DummyLM([{"revised_source": no_forward}])
    with pytest.raises(ValueError, match="forward"):
        _call(SkilledCodeProposer(prompt_model=lm, on_error="raise"))


def test_on_error_keep_still_raises_lm_error():
    from dspy.utils.exceptions import LMError

    proposer = SkilledCodeProposer()

    def boom(**kwargs):
        raise LMError("provider down")

    proposer.module = boom
    with pytest.raises(LMError):
        _call(proposer)


def test_prompt_inputs_passed_through():
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _Result(VALID_SRC)

    proposer = SkilledCodeProposer(
        skills=[Skill(name="guide", content="Be terse.", description="A guide.")],
        additional_instructions="Prefer one predictor.",
        max_examples=1,
    )
    proposer.module = capture
    proposer._catalog = lambda: "CATALOG TEXT"
    proposer(
        candidate={"flex_step": "class Old(dspy.Module):\n    pass"},
        reflective_dataset={"flex_step": [{"Inputs": "one"}, {"Inputs": "two"}]},
        components_to_update=["flex_step"],
        task_descriptions={"flex_step": "Signature: question -> answer"},
        context_blurbs={"flex_step": "tools: search"},
    )
    assert captured["task_description"] == "Signature: question -> answer"
    assert captured["available_context"] == "tools: search"
    assert captured["primitives_catalog"] == "CATALOG TEXT"
    assert "class Old" in captured["current_source"]
    assert "# Example 1" in captured["examples_with_feedback"]
    assert "# Example 2" not in captured["examples_with_feedback"]
    assert "<skill name='guide' description='A guide.'>" in captured["reference_skills"]
    assert captured["additional_guidance"] == "Prefer one predictor."


def test_missing_context_uses_defaults():
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _Result(VALID_SRC)

    proposer = SkilledCodeProposer()
    proposer.module = capture
    proposer._catalog = lambda: "CATALOG TEXT"
    proposer(
        candidate={"flex_step": "class Old(dspy.Module):\n    pass"},
        reflective_dataset={},
        components_to_update=["flex_step"],
        task_descriptions={},
        context_blurbs={},
    )
    assert captured["task_description"] == "flex_step"
    assert captured["available_context"] == "(no extra context)"
    assert captured["reference_skills"] == "None"
    assert captured["additional_guidance"] == "None"
    assert captured["examples_with_feedback"] == "No examples were provided."
