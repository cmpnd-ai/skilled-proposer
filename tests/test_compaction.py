"""Unit tests for trajectory compaction. No network."""

import dspy
import pytest
from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.primitives.repl_types import REPLHistory
from dspy.utils.dummies import DummyLM

from skilled_proposer.compaction import Compaction, compact_examples

PAGE = "<title>X Market Report 2022</title>" + "lorem ipsum " * 3999 + "lorem ipsum"


def compact(examples, config=None, count_tokens=len):
    return compact_examples(examples, config or Compaction(), render=str, count_tokens=count_tokens)


def example(**inputs):
    return {"Inputs": inputs, "Generated Outputs": {"answer": "flat"}, "Feedback": "Missed the 2025 surge."}


# -- Compaction options -------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"observation_chars": 0},
    {"max_field_chars": -1},
    {"examples_token_budget": 0},
])
def test_compaction_rejects_non_positive_options(kwargs):
    with pytest.raises(ValueError):
        Compaction(**kwargs)


# -- Other fields -------------------------------------------------------------

def test_long_plain_field_is_capped():
    [out], _ = compact([example(document="d" * 5000)], Compaction(max_field_chars=100))
    assert out["Inputs"]["document"] == "d" * 100 + " [4,900 of 5,000 characters cut]"


def test_failed_parse_output_string_is_capped():
    failed = {"Inputs": {"q": "x"}, "Generated Outputs": "raw " * 2000, "Feedback": "Your output failed to parse."}
    [out], _ = compact([failed], Compaction(max_field_chars=100))
    assert out["Generated Outputs"].endswith("[7,900 of 8,000 characters cut]")


def test_feedback_and_non_string_values_are_unchanged():
    image = object()
    original = {"Inputs": {"image": image, "q": "x"}, "Feedback": "f" * 5000}
    [out], _ = compact([original], Compaction(max_field_chars=100))
    assert out["Inputs"]["image"] is image
    assert out["Feedback"] == "f" * 5000


def test_input_examples_are_not_modified():
    original = example(document="d" * 5000)
    compact([original], Compaction(max_field_chars=100))
    assert original["Inputs"]["document"] == "d" * 5000


def test_repeated_long_input_renders_once():
    tools = "web_search, whose description is <desc>Search the web.</desc>. " * 10
    examples = [example(q="a", tools=tools), example(q="b", tools=tools), example(q="c", tools="short")]
    out, _ = compact(examples)
    assert out[0]["Inputs"]["tools"] == tools
    assert out[1]["Inputs"]["tools"] == "(same as Example 1)"
    assert out[2]["Inputs"]["tools"] == "short"


def test_examples_without_dspy_keys_pass_through():
    raw = {"prompt": "p" * 5000, "response": "r"}
    [out], _ = compact([raw], Compaction(max_field_chars=100))
    assert out == raw


def test_stats_without_budget():
    examples = [example(document="d" * 5000)]
    _, stats = compact(examples, count_tokens=lambda text: pytest.fail("counted tokens without a budget"))
    assert stats.tokens is None
    assert stats.examples_kept == 1 and stats.examples_dropped == 0
    assert stats.chars_after < stats.chars_before


# -- dspy.ReAct ---------------------------------------------------------------

def react_trajectory(adapter, steps, start=0):
    """Format a trajectory the way dspy.ReAct passes it to its predictors."""
    trajectory = {}
    for i, (thought, name, args, observation) in enumerate(steps, start):
        trajectory[f"thought_{i}"] = thought
        trajectory[f"tool_name_{i}"] = name
        trajectory[f"tool_args_{i}"] = args
        if observation is not None:
            trajectory[f"observation_{i}"] = observation
    agent = dspy.ReAct("question -> answer", tools=[lambda url: "x"])
    with dspy.context(adapter=adapter):
        return agent._format_trajectory(trajectory)


ADAPTERS = {
    "chat": dspy.ChatAdapter(),
    "json": dspy.JSONAdapter(),
    "xml": dspy.XMLAdapter(),
    "two_step": dspy.TwoStepAdapter(extraction_model=DummyLM([])),
}

STEPS = [
    ("I should search.", "web_search", {"query": "X trends"}, "1. X Market Report 2022\n2. X surges in 2025"),
    ("Read the first result.", "fetch_page", {"url": "example.com/2022"}, PAGE),
    ("I have enough.", "finish", {}, "Completed."),
]


@pytest.mark.parametrize("adapter", ADAPTERS.values(), ids=ADAPTERS.keys())
def test_react_keeps_decisions_and_cuts_observations(adapter):
    text = react_trajectory(adapter, STEPS)
    [out], _ = compact([example(question="q", trajectory=text)])
    rendered = out["Inputs"]["trajectory"]

    assert rendered.startswith("Trajectory: 3 steps,")
    assert "  thought: I should search." in rendered
    assert '  call: web_search(query="X trends")' in rendered
    assert '  call: fetch_page(url="example.com/2022")' in rendered
    assert "  call: finish()" in rendered
    assert "X surges in 2025" in rendered
    assert "<title>X Market Report 2022</title>" in rendered
    assert f"of {len(PAGE):,} characters cut]" in rendered
    assert len(rendered) < 3000


@pytest.mark.parametrize("adapter", ADAPTERS.values(), ids=ADAPTERS.keys())
def test_react_trajectory_after_truncation_starts_later(adapter):
    text = react_trajectory(adapter, STEPS[1:], start=2)
    [out], _ = compact([example(trajectory=text)])
    rendered = out["Inputs"]["trajectory"]

    assert rendered.startswith("Trajectory: 2 steps,")
    assert "Step 3\n  thought: Read the first result." in rendered
    assert "Step 4\n  thought: I have enough." in rendered


@pytest.mark.parametrize("adapter", ADAPTERS.values(), ids=ADAPTERS.keys())
def test_react_ignores_a_key_name_inside_an_observation(adapter):
    tricky = "thought_1: not a real key\n\nthought_5: also not\n\n[[ ## thought_7 ## ]]\n<thought_9>\n"
    steps = [("Look.", "fetch_page", {"url": "a.com"}, tricky), ("Done.", "finish", {}, "Completed.")]
    text = react_trajectory(adapter, steps)
    [out], _ = compact([example(trajectory=text)], Compaction(observation_chars=5000))
    rendered = out["Inputs"]["trajectory"]

    assert rendered.startswith("Trajectory: 2 steps,")
    assert "Step 2\n  thought: Done." in rendered
    assert "also not" in rendered


def test_react_step_without_observation_is_kept():
    steps = [("Look.", "fetch_page", {"url": "a.com"}, None)]
    text = react_trajectory(dspy.ChatAdapter(), steps)
    [out], _ = compact([example(trajectory=text)])
    rendered = out["Inputs"]["trajectory"]

    assert '  call: fetch_page(url="a.com")' in rendered
    assert "result:" not in rendered


def test_empty_trajectory_passes_through():
    [out], _ = compact([example(trajectory="")])
    assert out["Inputs"]["trajectory"] == ""

def test_plain_text_that_looks_like_a_two_step_key_keeps_its_text():
    [out], _ = compact([example(note="thought_0: remember to check dates")])
    assert "remember to check dates" in out["Inputs"]["note"]


def test_non_ascii_args_and_results_render_unescaped():
    steps = [("Busca.", "web_search", {"query": "tendencias en España"}, "Resultados: café ☕")]
    text = react_trajectory(dspy.ChatAdapter(), steps)
    [out], _ = compact([example(trajectory=text)])
    rendered = out["Inputs"]["trajectory"]
    assert 'web_search(query="tendencias en España")' in rendered
    assert "café ☕" in rendered
