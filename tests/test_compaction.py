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


# -- dspy.ReActV2 (dspy.History) ---------------------------------------------

def history_context(messages):
    """Format a dspy.History the way make_reflective_dataset writes Context."""
    s = "```json\n"
    for i, message in enumerate(messages):
        s += f"  {i}: {message}\n"
    s += "```"
    return s


def tool_calls(*calls, results=None):
    tc = ToolCalls(tool_calls=[
        ToolCalls.ToolCall(name=name, args=args, id=f"call_{i}") for i, (name, args) in enumerate(calls)
    ])
    if results is not None:
        values = [value for value, _ in results]
        errors = [is_error for _, is_error in results]
        tcr = ToolCallResults.from_tool_calls_and_values(tc, values, errors)
        tc = tc.model_copy(update={"tool_call_results": tcr})
    return tc


def test_history_renders_turns_with_parallel_calls_and_errors():
    messages = [
        {
            "question": "What is trending in X?",
            "next_thought": "Search twice.",
            "tool_calls": tool_calls(
                ("web_search", {"query": "X trends"}),
                ("fetch_page", {"url": "example.com/2022"}),
                results=[("1. X surges in 2025", False), (PAGE, False)],
            ),
        },
        {
            "next_thought": "Try the other page.",
            "tool_calls": tool_calls(("fetch_page", {"url": "bad"}), results=[("Execution error in fetch_page: 404", True)]),
        },
        {
            "next_thought": "Answer.",
            "tool_calls": tool_calls(("submit", {"answer": "flat"}), results=[({"answer": "flat"}, False)]),
            "answer": "flat",
        },
    ]
    [out], _ = compact([{"Inputs": {"Context": history_context(messages)}, "Feedback": "f"}])
    rendered = out["Inputs"]["Context"]

    assert rendered.startswith("Trajectory: 3 steps,")
    assert "Step 1\n  question: What is trending in X?\n  thought: Search twice." in rendered
    assert '  call: web_search(query="X trends")\n  result: 1. X surges in 2025' in rendered
    assert f"of {len(PAGE):,} characters cut]" in rendered
    assert "  error: Execution error in fetch_page: 404" in rendered
    assert '  call: submit(answer="flat")' in rendered
    assert len(rendered) < 3000


def test_history_line_that_fails_to_parse_keeps_its_text_and_others_parse():
    good = {"next_thought": "Look.", "tool_calls": tool_calls(("fetch_page", {"url": "a.com"}), results=[("ok", False)])}
    text = history_context([good, "<Foo object at 0x1234> " + "z" * 5000])
    [out], _ = compact([{"Inputs": {"Context": text}}])
    rendered = out["Inputs"]["Context"]

    assert '  call: fetch_page(url="a.com")' in rendered
    assert "Step 2\n  message: <Foo object at 0x1234>" in rendered
    assert "of 5,023 characters cut]" in rendered


def test_history_tool_result_that_is_not_a_literal_renders_as_source():
    tc = tool_calls(("fetch_page", {"url": "a.com"}))
    line = {"next_thought": "Look.", "tool_calls": tc}
    text = history_context([line]).replace(
        "tool_call_results=None",
        "tool_call_results=ToolCallResults(tool_call_results=[ToolCallResult(call_id='call_0', "
        "name='fetch_page', value=Page(url='a.com'), is_error=False)])",
    )
    [out], _ = compact([{"Inputs": {"Context": text}}])
    assert "  result: Page(url='a.com')" in out["Inputs"]["Context"]

def test_history_from_a_chat_program_renders_each_message():
    messages = [{"question": "Hi?", "answer": "Hello."}, {"question": "More?", "answer": "a" * 5000}]
    [out], _ = compact([{"Inputs": {"Context": history_context(messages)}}], Compaction(max_field_chars=100))
    rendered = out["Inputs"]["Context"]
    assert "Step 1\n  question: Hi?\n  answer: Hello." in rendered
    assert "[4,900 of 5,000 characters cut]" in rendered


def test_history_message_with_a_raw_newline_stays_in_its_step():
    text = history_context([{"next_thought": "a"}, "Weird(\nmulti-line)", {"next_thought": "c"}])
    [out], _ = compact([{"Inputs": {"Context": text}}])
    rendered = out["Inputs"]["Context"]
    assert rendered.startswith("Trajectory: 3 steps,")
    assert "Step 2\n  message: Weird(\n    multi-line)" in rendered
    assert "Step 3\n  thought: c" in rendered


# -- dspy.RLM -----------------------------------------------------------------

def test_rlm_history_renders_what_the_student_saw():
    history = (
        REPLHistory(max_output_chars=1000)
        .append(reasoning="Check the length.", code="print(len(doc))", output="120000")
        .append(reasoning="Read it.", code="print(doc)", output="A" * 3000 + "THE END")
    )
    [out], _ = compact([{"Inputs": {"repl_history": str(history)}}])
    rendered = out["Inputs"]["repl_history"]

    assert rendered == history.format()
    assert "THE END" in rendered
    assert "Output (3,007 chars)" in rendered


def test_rlm_empty_history_renders_the_student_message():
    [out], _ = compact([{"Inputs": {"repl_history": str(REPLHistory())}}])
    assert out["Inputs"]["repl_history"] == REPLHistory().format()


# -- Token budget -------------------------------------------------------------

def page_example(i):
    steps = [("Read.", "fetch_page", {"url": f"site{i}.com"}, PAGE)]
    return example(q=f"q{i}", trajectory=react_trajectory(dspy.ChatAdapter(), steps))


def test_budget_shrinks_caps_before_dropping_examples():
    examples = [page_example(i) for i in range(3)]
    out, stats = compact(examples, Compaction(examples_token_budget=2200), count_tokens=len)
    assert stats.examples_kept == 3 and stats.examples_dropped == 0
    assert stats.tokens <= 2200
    assert "[47," in out[0]["Inputs"]["trajectory"]


def test_budget_drops_trailing_examples_after_caps_reach_floors():
    examples = [page_example(i) for i in range(3)]
    out, stats = compact(examples, Compaction(examples_token_budget=1000), count_tokens=len)
    assert stats.examples_dropped >= 1
    assert [e["Inputs"]["q"] for e in out] == [f"q{i}" for i in range(stats.examples_kept)]
    assert not stats.over_budget


def test_budget_keeps_one_example_when_nothing_fits():
    examples = [page_example(i) for i in range(3)]
    out, stats = compact(examples, Compaction(examples_token_budget=10), count_tokens=len)
    assert len(out) == 1 and out[0]["Inputs"]["q"] == "q0"
    assert stats.over_budget
