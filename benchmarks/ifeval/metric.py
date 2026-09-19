"""Score a response against its verifiable constraint and explain the result."""

from __future__ import annotations

import dspy

from benchmarks.ifeval.verifiers import check


def _describe_constraint(gold) -> str:
    args = {k: v for k, v in gold.ground_truth.items() if k != "func_name" and v is not None}
    args.pop("original_prompt", None)
    text = f"'{gold.constraint}'"
    if args:
        text += " with " + ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text


def ifeval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    response = getattr(pred, "response", "") or ""
    constraint = _describe_constraint(gold)
    if not response.strip():
        return dspy.Prediction(score=0.0, feedback=f"No response produced. Violated the constraint {constraint}.")
    try:
        passed, detail = check(response, gold.ground_truth)
    except Exception as e:  # a malformed response should score zero, not crash the run
        return dspy.Prediction(score=0.0, feedback=f"Checker error ({e}). Violated the constraint {constraint}.")
    verdict = "Satisfied" if passed else "Violated"
    return dspy.Prediction(score=1.0 if passed else 0.0, feedback=f"{verdict} the constraint {constraint}. Measured: {detail}.")


# (ground_truth, response, expected score)
SELFTEST_CASES = [
    ({"func_name": "validate_lowercase"}, "all quiet here", 1.0),
    ({"func_name": "validate_lowercase"}, "Not quiet", 0.0),
    ({"func_name": "validate_word_constraint", "N": 5, "quantifier": "at most"}, "one two three", 1.0),
    ({"func_name": "validate_word_constraint", "N": 5, "quantifier": "at least"}, "one two three", 0.0),
    ({"func_name": "verify_keywords", "keyword_list": ["apple", "pear"]}, "An Apple and a pear.", 1.0),
    ({"func_name": "validate_json_format"}, '{"a": 1}', 1.0),
    ({"func_name": "validate_end", "end_phrase": "Is there anything else?"}, "Done. Is there anything else?", 1.0),
    ({"func_name": "validate_two_responses"}, "yes ****** no", 1.0),
    ({"func_name": "validate_two_responses"}, "yes ****** yes", 0.0),
]
