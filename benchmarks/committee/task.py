"""The extraction program and the two models the harness talks to."""

from __future__ import annotations

import dspy

DEFAULT_STUDENT_MODEL = "lfm2.5-1.2b-instruct-mlx"
DEFAULT_STUDENT_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_REFLECTION_MODEL = "openai/gpt-5.6-luna"


class ExtractCommittee(dspy.Signature):
    """Identify the specific campaign committee responsible for sending a
    political fundraising email.
    """

    email_body: str = dspy.InputField()
    committee: str = dspy.OutputField()


class CommitteeProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractCommittee)

    def forward(self, email_body):
        return self.extract(email_body=email_body)


def build_program() -> dspy.Module:
    return CommitteeProgram()


class SchemaForcedLM(dspy.LM):
    """A dspy.LM that reports structured output support.

    LM Studio accepts response_format json_schema and rejects json_object.
    litellm has no model info for a locally served model, so dspy would
    otherwise send json_object on the JSON retry path.
    """

    @property
    def supports_response_schema(self) -> bool:
        return True


def build_student(model: str, api_base: str, api_key: str, **lm_kwargs) -> dspy.LM:
    return SchemaForcedLM(f"openai/{model}", api_base=api_base, api_key=api_key or "lm-studio", **lm_kwargs)


def build_reflection(model: str = DEFAULT_REFLECTION_MODEL) -> dspy.LM:
    return dspy.LM(model)


class ReflectionAdapterScope:
    """Run a proposer under a fixed adapter so a student-only adapter never
    formats the reflection prompt.

    GEPA sets the reflection LM around each proposer call but keeps the
    process-wide adapter. When the student needs its own adapter, wrap the
    proposer in this so reflection keeps the adapter it was tested with.
    """

    def __init__(self, proposer, adapter):
        self.proposer = proposer
        self.adapter = adapter

    def __call__(self, *args, **kwargs):
        with dspy.context(adapter=self.adapter):
            return self.proposer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.proposer, name)
