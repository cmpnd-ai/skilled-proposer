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
