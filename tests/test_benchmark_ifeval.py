import collections

import dspy

from benchmarks.ifeval.ablations import ABLATIONS, resolve
from benchmarks.ifeval.data import DATA_PATH, load_rows, make_examples, split
from benchmarks.ifeval.metric import SELFTEST_CASES, ifeval_metric
from benchmarks.ifeval.task import Respond, build_program
from benchmarks.ifeval.verifiers import CHECKERS, check


def test_prompts_are_stratified_and_unique():
    rows = load_rows(DATA_PATH)
    assert len(rows) == 1200
    counts = collections.Counter(r["ground_truth"]["func_name"] for r in rows)
    assert len(counts) == 24
    assert set(counts.values()) == {50}
    assert set(counts) == set(CHECKERS)
    assert len({r["prompt"] for r in rows}) == 1200
    assert max(len(r["prompt"]) for r in rows) < 1500


def test_split_is_deterministic_and_disjoint():
    examples = make_examples(load_rows(DATA_PATH))
    train, val, test = split(examples, seed=13)
    assert (len(train), len(val), len(test)) == (900, 50, 200)
    assert split(examples, seed=13)[2][0].prompt == test[0].prompt
    prompts = [e.prompt for e in train + val + test]
    assert len(set(prompts)) == len(prompts)
    assert set(train[0].inputs().keys()) == {"prompt"}


def test_every_checker_runs_on_its_rows():
    for r in load_rows(DATA_PATH):
        passed, detail = check("Sample response. * * * Another part.", r["ground_truth"])
        assert isinstance(passed, bool) and detail


def test_metric_selftest_cases():
    for ground_truth, response, expected in SELFTEST_CASES:
        gold = dspy.Example(constraint="c", ground_truth=ground_truth)
        out = ifeval_metric(gold, dspy.Prediction(response=response))
        assert out.score == expected, (ground_truth, response)
        assert ("Satisfied" if expected else "Violated") in out.feedback


def test_metric_feedback_reports_measurement_and_arguments():
    gold = dspy.Example(
        constraint="Answer with at least / around / at most {N} words",
        ground_truth={"func_name": "validate_word_constraint", "N": 5, "quantifier": "at least"},
    )
    out = ifeval_metric(gold, dspy.Prediction(response="one two three"))
    assert out.score == 0.0
    assert "3 words, required at least 5" in out.feedback
    assert "N=5" in out.feedback and "quantifier='at least'" in out.feedback


def test_metric_empty_response_scores_zero():
    gold = dspy.Example(constraint="c", ground_truth={"func_name": "validate_lowercase"})
    out = ifeval_metric(gold, dspy.Prediction(response=""))
    assert out.score == 0.0 and "No response" in out.feedback


def test_program_shape():
    program = build_program()
    assert set(Respond.input_fields) == {"prompt"}
    assert set(Respond.output_fields) == {"response"}
    assert [name for name, _ in program.named_predictors()] == ["respond"]


def test_ablations():
    assert list(ABLATIONS) == ["stock", "baseline", "pxn-long", "batch-pxn-long"]
    control, batch = resolve(["pxn-long", "batch-pxn-long"])
    assert "candidates" not in control.proposer_kwargs
    assert batch.proposer_kwargs["candidates"] == 4
    assert control.engine_kwargs.keys() == batch.engine_kwargs.keys() == {"sampling_strategy", "selection_strategy"}
