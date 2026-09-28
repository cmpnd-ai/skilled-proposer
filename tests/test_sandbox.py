from skilled_proposer.sandbox import SandboxJSON, summarize_journal, summarize_records, to_jsonable


def test_sandbox_json_round_trips_through_its_own_assignment():
    data = {"records": [{"Inputs": {"q": "x"}, "Feedback": "wrong"}]}
    value = SandboxJSON(data, "summary")
    ns = {"_raw": value.to_sandbox().decode("utf-8")}
    exec(value.sandbox_setup() + "\n" + value.sandbox_assignment("examples", "_raw"), ns)
    assert ns["examples"] == data
    assert value.rlm_preview() == "summary"
    assert str(value) == "summary"


def test_to_jsonable_stringifies_unknown_types():
    class Thing:
        def __str__(self):
            return "thing"

    assert to_jsonable({"a": (1, Thing()), 2: None}) == {"a": [1, "thing"], "2": None}


def test_record_summary_shows_shape_not_content():
    data = {"records": [
        {"Inputs": "SECRET-ALPHA", "Feedback": "wrong", "tags": {}},
        {"Inputs": "beta beta", "Feedback": "ok", "tags": {}},
    ]}
    text = summarize_records("examples", data)
    assert "list of 2 record dicts" in text
    assert "Inputs" in text and "Feedback" in text
    assert "SECRET-ALPHA" not in text and "wrong" not in text


def test_empty_record_summary():
    assert "list of 0 record dicts" in summarize_records("examples", {"records": []})


def test_seen_summary_counts_relations():
    data = {
        "instructions": {"h1": {}, "h2": {}},
        "records": [
            {"Feedback": "x", "tags": {"relation": "current"}},
            {"Feedback": "y", "tags": {"relation": "ancestor"}},
            {"Feedback": "z", "tags": {"relation": "ancestor"}},
        ],
    }
    text = summarize_records("seen", data)
    assert "maps 2 instruction hashes" in text
    assert "current 1, ancestor 2" in text


def test_journal_summary():
    data = {"lessons": ["a"], "hypotheses": [], "entries": [
        {"id": "1:p", "iteration": 1, "verdict": "accepted"},
        {"id": "2:p", "iteration": 2, "verdict": "not chosen as a parent so far"},
    ]}
    text = summarize_journal(data)
    assert "1 lessons and 0 hypotheses" in text
    assert "2 proposals: 1 accepted, 1 not chosen" in text
    assert "Iterations 1 to 2" in text
