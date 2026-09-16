import dspy

from benchmarks.committee.data import DATA_PATH, load_rows, make_examples, split
from benchmarks.committee.metric import SELFTEST_CASES, committee_metric
from benchmarks.committee.task import ExtractCommittee, build_program


def test_data_loads_and_splits_without_overlap():
    rows = load_rows(DATA_PATH)
    assert len(rows) == 956
    examples = make_examples(rows)
    train, val, test = split(examples, sizes=(556, 200, 200), seed=13)
    assert (len(train), len(val), len(test)) == (556, 200, 200)
    bodies = [e.email_body for e in train + val + test]
    assert len(set(bodies)) == len(bodies)
    assert set(train[0].inputs().keys()) == {"email_body"}


def test_clean_body_strips_invisible_padding():
    from benchmarks.committee.data import clean_body
    padded = "Let’s finish what we started ͏ ͏ ͏ ͏​‌⁠ Paid for by Kean for Congress Inc"
    assert clean_body(padded) == "Let’s finish what we started Paid for by Kean for Congress Inc"
    assert clean_body("  plain   text \n next ") == "plain text \n next"


def test_loaded_bodies_have_no_invisible_padding():
    rows = load_rows(DATA_PATH)
    assert all("͏" not in r["body"] and "​" not in r["body"] for r in rows)
    target = [r for r in rows if r["body"].startswith("Let’s finish what we started")]
    assert len(target) == 1
    assert len(target[0]["body"]) < 2400


def test_load_rows_drops_repeated_bodies():
    rows = load_rows(DATA_PATH)
    bodies = [r["body"].strip() for r in rows]
    assert len(set(bodies)) == len(bodies)


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


import json

from benchmarks.committee.ablations import ABLATIONS, LONG_GUIDANCE, resolve
from benchmarks.committee import report
from skilled_proposer import SkilledProposer


def test_ablation_registry_names():
    assert list(ABLATIONS) == [
        "stock", "baseline", "journal", "dedupe", "journal+dedupe",
        "baseline-long", "baseline-short", "journal-long", "dedupe-long",
    ]
    dedupe_long = ABLATIONS["dedupe-long"]()
    assert dedupe_long.proposer_kwargs["dedupe"].threshold == 0.5
    assert dedupe_long.proposer_kwargs["additional_instructions"] == LONG_GUIDANCE
    long = ABLATIONS["baseline-long"]()
    short = ABLATIONS["baseline-short"]()
    journal_long = ABLATIONS["journal-long"]()
    assert "500 words" in long.proposer_kwargs["additional_instructions"]
    assert short.proposer_kwargs == {"max_words": 150}
    assert journal_long.proposer_kwargs["journal"] is True
    assert journal_long.proposer_kwargs["additional_instructions"] == long.proposer_kwargs["additional_instructions"]
    assert [a.name for a in resolve(["all"])] == list(ABLATIONS)
    assert [a.name for a in resolve(["journal", "stock"])] == ["journal", "stock"]


def test_ablation_builds_proposer_or_none(tmp_path):
    stock = ABLATIONS["stock"]()
    assert stock.build_proposer(tmp_path) is None
    both = ABLATIONS["journal+dedupe"]()
    proposer = both.build_proposer(tmp_path)
    assert isinstance(proposer, SkilledProposer)
    assert proposer.journal_enabled and proposer.dedupe is not None
    assert proposer.journal_path == tmp_path / "journal.json"
    kwargs = both.gepa_kwargs(proposer, extra_callbacks=[object()])
    assert kwargs["callbacks"][0] is proposer and len(kwargs["callbacks"]) == 2


def test_report_table_and_curves(tmp_path):
    for name, score in (("stock", 0.5), ("journal", 0.7)):
        d = tmp_path / name / "13"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "ablation": name, "seed": 13, "test_score": score, "best_valset_score": score,
            "metric_calls_to_best": 100, "accept_rate": 0.25, "reflection_calls": 10,
            "duplicates": 1, "duplicates_after_retry": 0, "auc_valset_vs_calls": 12.0,
        }))
        (d / "trajectory.jsonl").write_text(json.dumps({"iteration": 1, "metric_calls_used": 10,
                                                        "best_valset_score": score}) + "\n")
    summaries = report.load_summaries(tmp_path)
    table = report.build_table(summaries)
    assert "journal" in table and "0.700" in table
    out = tmp_path / "curves.csv"
    report.write_curves(tmp_path, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "ablation,seed,iteration,metric_calls_used,best_valset_score"
    assert len(lines) == 3


def test_report_backfills_instruction_words_from_program(tmp_path):
    d = tmp_path / "stock" / "13"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"ablation": "stock", "seed": 13, "test_score": 0.5}))
    (d / "program.json").write_text(json.dumps({
        "extract": {"signature": {"instructions": "one two three four five"}},
        "metadata": {"dependency_versions": {}},
    }))
    summaries = report.load_summaries(tmp_path)
    assert summaries[0]["best_instruction_words"] == 5
    assert report.saved_instruction_words(tmp_path / "missing.json") is None
    table = report.build_table(summaries)
    assert "words" in table.splitlines()[0]
    assert table.splitlines()[1].split()[-1] == "5"


def test_instruction_words_counts_predictor_instructions():
    from benchmarks.committee.run import instruction_words
    program = build_program()
    program.extract.signature = program.extract.signature.with_instructions("Return only the committee name.")
    assert instruction_words(program) == 5


def test_run_dry_run_makes_no_lm_calls(capsys):
    from benchmarks.committee.run import main
    main(["--dry-run"])
    out = capsys.readouterr().out
    assert "556 train / 200 val / 200 test" in out
    assert "metric self-check passed" in out


def test_tracing_off_without_key(monkeypatch, capsys):
    from benchmarks.committee.run import configure_tracing
    monkeypatch.delenv("CMPND_API_KEY", raising=False)
    assert configure_tracing() is False
    assert "tracing is off" in capsys.readouterr().out


def test_tracing_on_with_key(monkeypatch, capsys):
    import cmpnd
    from benchmarks.committee import run
    calls = []
    monkeypatch.setenv("CMPND_API_KEY", "test-key")
    monkeypatch.setattr(cmpnd, "configure", lambda **kw: calls.append(("configure", kw)))
    monkeypatch.setattr(cmpnd, "auto_instrument", lambda: calls.append(("auto_instrument", {})))
    assert run.configure_tracing() is True
    assert calls == [("configure", {"project_tags": run.TRACE_TAGS}), ("auto_instrument", {})]
    assert "tracing on" in capsys.readouterr().out
