"""The response program for the IFEval-style task.

The student and reflection model builders are shared with the committee
task, so both benchmarks talk to the same LM Studio and reflection setup.
"""

from __future__ import annotations

import dspy
from dspy.utils.exceptions import AdapterParseError

from benchmarks.committee.task import (  # noqa: F401  re-exported for run.py
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_STUDENT_API_BASE,
    DEFAULT_STUDENT_MODEL,
    ReflectionAdapterScope,
    build_reflection,
    build_student,
)


class Respond(dspy.Signature):
    """Write a response to the request that satisfies every stated constraint."""

    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()


class RespondProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(Respond)

    def forward(self, prompt):
        try:
            return self.respond(prompt=prompt)
        except AdapterParseError as e:
            # The student answered without the adapter's field markers. Score
            # that text as the response rather than dropping the example, which
            # leaves GEPA's valset bookkeeping one output short and crashes it.
            return dspy.Prediction(response=(e.lm_response or "").strip())
        except Exception as e:
            # A request too long for the server's per-slot context is a failure
            # of that one example, not of the run. Anything else stays loud.
            if "context" in str(e).lower():
                return dspy.Prediction(response="")
            raise


def build_program() -> dspy.Module:
    return RespondProgram()
