"""Runs the RLM engine in dspy's real Deno/Pyodide sandbox (the rlm-deno extra,
opted into explicitly since Monty is the default). Skips without Deno."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter import PythonInterpreter

from skilled_proposer import SkilledProposer

from rlm_support import scripted


def _deno_error() -> str | None:
    """Why dspy's Deno sandbox cannot start here, or None when it can."""
    from dspy.primitives.python_interpreter import _find_deno_executable, _validate_deno_version

    try:
        _validate_deno_version(_find_deno_executable())
    except CodeInterpreterError as e:
        return str(e)
    return None


pytestmark = [
    pytest.mark.deno,
    pytest.mark.skipif(_deno_error() is not None, reason="Deno is not installed"),
]


def test_records_and_seen_arrive_in_the_real_sandbox():
    code = (
        "r = examples['records'][0]\n"
        "SUBMIT(new_instruction=r['Feedback'] + ' ' + r['tags']['component'] + ' ' "
        "+ str(len(seen['instructions'])), change_summary='x')"
    )
    proposer = SkilledProposer(
        engine="rlm", review="seen", prompt_model=scripted(code), interpreter_factory=PythonInterpreter
    )
    out = proposer(
        candidate={"p": "Do it."},
        reflective_dataset={"p": [{"Inputs": {"q": "x"}, "Feedback": "wrong"}]},
        components_to_update=["p"],
    )
    assert out == {"p": "wrong p 0"}
