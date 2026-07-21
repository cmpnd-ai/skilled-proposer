import pytest
from dspy.utils.dummies import DummyLM

from skilled_proposer.proposer import SkilledProposer, _render_examples
from skilled_proposer.skill import Skill


def test_init_validation():
    with pytest.raises(ValueError):
        SkilledProposer(max_tokens=0)
    with pytest.raises(ValueError):
        SkilledProposer(max_words=-1)
    with pytest.raises(ValueError):
        SkilledProposer(on_error="explode")


def test_call_proposes_for_each_component():
    lm = DummyLM([{"new_instruction": "New A."}, {"new_instruction": "New B."}])
    proposer = SkilledProposer(prompt_model=lm)
    out = proposer(
        candidate={"a": "old a", "b": "old b"},
        reflective_dataset={"a": [{"Inputs": "x", "Feedback": "wrong"}], "b": []},
        components_to_update=["a", "b"],
    )
    assert out == {"a": "New A.", "b": "New B."}


def test_on_error_keep_returns_current_instruction():
    proposer = SkilledProposer()

    def boom(**kwargs):
        raise RuntimeError("boom")

    proposer.module = boom
    out = proposer(
        candidate={"a": "old a"},
        reflective_dataset={"a": []},
        components_to_update=["a"],
    )
    assert out == {"a": "old a"}


def test_on_error_raise_propagates():
    proposer = SkilledProposer(on_error="raise")

    def boom(**kwargs):
        raise RuntimeError("boom")

    proposer.module = boom
    with pytest.raises(RuntimeError, match="boom"):
        proposer(
            candidate={"a": "old a"},
            reflective_dataset={"a": []},
            components_to_update=["a"],
        )


def test_render_skills_includes_name_and_description():
    proposer = SkilledProposer(
        skills=[Skill(name="guide", content="Be terse.", description="A guide.")]
    )
    rendered = proposer._render_skills()
    assert "<skill name='guide' description='A guide.'>" in rendered
    assert "Be terse." in rendered


def test_render_skills_without_description():
    proposer = SkilledProposer(skills=["# Inline\nBody."])
    rendered = proposer._render_skills()
    assert "<skill name='Inline'>" in rendered


def test_render_skills_none():
    assert SkilledProposer()._render_skills() == "None"


def test_max_examples_caps_rendered_examples():
    captured = {}

    class Result:
        new_instruction = "ok"

    def capture(**kwargs):
        captured.update(kwargs)
        return Result()

    proposer = SkilledProposer(max_examples=1)
    proposer.module = capture
    proposer(
        candidate={"a": "old"},
        reflective_dataset={"a": [{"Inputs": "one"}, {"Inputs": "two"}]},
        components_to_update=["a"],
    )
    assert "# Example 1" in captured["examples_with_feedback"]
    assert "# Example 2" not in captured["examples_with_feedback"]


def test_render_examples_empty():
    assert _render_examples([]) == "No examples were provided."


def test_render_examples_nested():
    out = _render_examples(
        [{"Inputs": {"question": "q1"}, "Feedback": "too vague"}]
    )
    assert "# Example 1" in out
    assert "## Inputs" in out
    assert "### question" in out
    assert "q1" in out
    assert "too vague" in out


def test_render_examples_list_items():
    out = _render_examples([{"Outputs": ["first", "second"]}])
    assert "### Item 1" in out
    assert "### Item 2" in out
