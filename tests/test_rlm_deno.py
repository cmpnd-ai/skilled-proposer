"""Runs the RLM engine in dspy's real Deno/Pyodide sandbox. Skips without Deno."""

import pytest

from skilled_proposer import SkilledProposer
from skilled_proposer.proposer import _deno_error

from rlm_support import scripted

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
    proposer = SkilledProposer(engine="rlm", review="seen", prompt_model=scripted(code))
    out = proposer(
        candidate={"p": "Do it."},
        reflective_dataset={"p": [{"Inputs": {"q": "x"}, "Feedback": "wrong"}]},
        components_to_update=["p"],
    )
    assert out == {"p": "wrong p 0"}
