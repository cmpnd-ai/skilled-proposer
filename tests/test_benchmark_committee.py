import dspy

from benchmarks.committee.data import DATA_PATH, load_rows, make_examples, split
from benchmarks.committee.metric import SELFTEST_CASES, committee_metric
from benchmarks.committee.task import ExtractCommittee, build_program


def test_data_loads_and_splits_without_overlap():
    rows = load_rows(DATA_PATH)
    assert len(rows) == 1000
    examples = make_examples(rows)
    train, val, test = split(examples, sizes=(600, 200, 200), seed=13)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    bodies = [e.email_body for e in train + val + test]
    assert len(set(bodies)) == len(bodies)
    assert train[0].inputs().keys() == {"email_body"}


def test_split_is_deterministic():
    examples = make_examples(load_rows(DATA_PATH))
    a = split(examples, seed=13)[2][0].committee
    b = split(examples, seed=13)[2][0].committee
    assert a == b


def test_metric_selftest_cases():
    for gold, pred, expected in SELFTEST_CASES:
        out = committee_metric(dspy.Example(committee=gold), dspy.Prediction(committee=pred))
        if expected is not None:
            assert abs(out.score - expected) < 1e-9, (gold, pred, out.score)
        assert isinstance(out.feedback, str) and out.feedback


def test_metric_flags_leaked_tags():
    out = committee_metric(
        dspy.Example(committee="Ted Cruz for Senate"),
        dspy.Prediction(committee="<committee>Ted Cruz for Senate</committee>"),
    )
    assert "XML/HTML tags" in out.feedback


def test_program_shape():
    program = build_program()
    assert set(ExtractCommittee.input_fields) == {"email_body"}
    assert set(ExtractCommittee.output_fields) == {"committee"}
    assert [name for name, _ in program.named_predictors()] == ["extract"]
