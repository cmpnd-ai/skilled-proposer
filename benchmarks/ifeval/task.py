"""The response program for the IFEval-style task.

The student and reflection model builders are shared with the committee
task, so both benchmarks talk to the same LM Studio and reflection setup.
"""

from __future__ import annotations

import dspy

from benchmarks.committee.task import (  # noqa: F401  re-exported for run.py
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_STUDENT_API_BASE,
    DEFAULT_STUDENT_MODEL,
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
        return self.respond(prompt=prompt)


def build_program() -> dspy.Module:
    return RespondProgram()
