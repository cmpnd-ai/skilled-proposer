"""Runs SkilledCodeProposer through dspy.GEPA's code_proposer hook on a
dspy.Flex program, in dspy's real Deno sandbox. Stub LMs, no network."""

import dspy
from dspy.utils.dummies import DummyLM

from skilled_proposer import SkilledCodeProposer, SkilledProposer

from deno_support import needs_deno

pytestmark = needs_deno

FIXED_SOURCE = '''class AnswerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(dspy.Signature("question -> answer", "GOOD"))

    def forward(self, **inputs):
        result = self.answer(**inputs)
        return dspy.Prediction(answer=result.answer)'''


class StudentLM(DummyLM):
    """Answers correctly only when the instruction contains GOOD."""

    def __init__(self):
        super().__init__([])

    def forward(self, prompt=None, messages=None, **kwargs):
        system = messages[0]["content"] if messages else (prompt or "")
        self.answers = iter([{"answer": "gold" if "GOOD" in system else "bad"}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class ReflectionLM(DummyLM):
    """Proposes FIXED_SOURCE and records which code prompt asked for it."""

    def __init__(self):
        super().__init__([])
        self.prompts = []

    def forward(self, prompt=None, messages=None, **kwargs):
        system = messages[0]["content"] if messages else (prompt or "")
        self.prompts.append(system)
        self.answers = iter([{"revised_source": FIXED_SOURCE}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.flex = dspy.Flex("question -> answer")

    def forward(self, question):
        return self.flex(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ok = (pred.answer or "").strip() == gold.answer
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback="ok" if ok else f"expected {gold.answer}")


def test_gepa_routes_flex_components_to_skilled_code_proposer():
    train = [dspy.Example(question=f"q{i}", answer="gold").with_inputs("question") for i in range(4)]
    val = train[:3]
    reflection_lm = ReflectionLM()
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        instruction_proposer=SkilledProposer(),
        code_proposer=SkilledCodeProposer(),
        max_metric_calls=40,
        reflection_minibatch_size=2,
    )
    with dspy.context(lm=StudentLM()):
        optimized = optimizer.compile(Program(), trainset=train, valset=val)

    assert reflection_lm.prompts
    assert all("memorization table" in p for p in reflection_lm.prompts)
    assert optimized.flex.module_src == FIXED_SOURCE
