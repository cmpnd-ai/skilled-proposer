"""Runs dspy.GEPA on a dspy.ReAct agent with stub LMs. No network."""

import dspy
from dspy.adapters.types.tool import ToolCalls
from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledProposer

from lm_support import RespondingLM

PAGE = "<title>Report</title>" + "lorem ipsum " * 3999 + "lorem ipsum"


def fetch_page(url: str) -> str:
    """Fetch a web page."""
    return PAGE


class AgentLM(RespondingLM):
    """Fetches one page, then finishes. Answers correctly only when the
    instruction that writes the answer contains GOOD."""

    def respond(self, messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        answer = "gold" if "GOOD" in system else "bad"
        # dspy.ReAct before DSPy 3.5: one predictor picks tools, another extracts the answer.
        if "next_tool_name" in system:
            if "observation_0" in user:
                return {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}}
            return {"next_thought": "Read it.", "next_tool_name": "fetch_page", "next_tool_args": {"url": "a.com"}}
        if "tool_calls" in system:
            fetched = any(m["role"] == "tool" for m in messages)
            call = {"name": "submit", "args": {"answer": answer}} if fetched else {
                "name": "fetch_page", "args": {"url": "a.com"}}
            return {"next_thought": "Next.", "tool_calls": ToolCalls.from_dict_list([call])}
        return {"reasoning": "r", "answer": answer}


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ok = (pred.answer or "").strip() == gold.answer
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback="ok" if ok else f"expected {gold.answer}")


def reflection_prompts(proposer):
    dspy.configure(lm=AgentLM())
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    reflection = DummyLM([{"new_instruction": "GOOD instruction"}] * 20)
    train = [dspy.Example(question=f"q{i}", answer="gold").with_inputs("question") for i in range(4)]
    val = [dspy.Example(question=f"v{i}", answer="gold").with_inputs("question") for i in range(2)]
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection,
        instruction_proposer=proposer,
        max_metric_calls=16,
        reflection_minibatch_size=2,
        num_threads=1,
        use_merge=False,
        seed=0,
    )
    optimizer.compile(dspy.ReAct("question -> answer", tools=[fetch_page]), trainset=train, valset=val)
    return [h["messages"][-1]["content"] for h in reflection.history]


def test_gepa_reflection_prompt_is_compacted():
    prompts = reflection_prompts(SkilledProposer(compaction=True))
    assert prompts
    assert any("Trajectory:" in p and "characters cut]" in p for p in prompts)
    assert all(PAGE not in p for p in prompts)
    assert max(len(p) for p in prompts) < 15000


def test_gepa_reflection_prompt_without_compaction_has_the_full_page():
    prompts = reflection_prompts(SkilledProposer())
    assert any(PAGE in p for p in prompts)
