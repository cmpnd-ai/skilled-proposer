import pytest

from skilled_proposer.dedupe import DedupeConfig, find_duplicates, render_duplicates, similarity


def test_config_defaults():
    c = DedupeConfig()
    assert c.threshold == 0.85
    assert c.max_retries == 1
    assert c.against == ("rejected", "pool")
    assert c.on_duplicate == "return"


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0.0},
    {"threshold": 1.5},
    {"max_retries": -1},
    {"against": ()},
    {"against": ("batch",)},
    {"on_duplicate": "loop"},
])
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        DedupeConfig(**kwargs)


def test_identical_and_whitespace_variants_score_one():
    assert similarity("Answer with the name only.", "answer   with the name only.") == 1.0


def test_reordered_sentences_score_high_through_jaccard():
    a = "State the task. Return only the name. Drop the prefix."
    b = "Drop the prefix. State the task. Return only the name."
    assert similarity(a, b) >= 0.85


def test_one_word_change_scores_high():
    a = "Return only the committee name from the disclaimer, without any prefix or address."
    b = "Return only the committee name from the disclaimer, without any preface or address."
    assert similarity(a, b) >= 0.85


def test_different_approach_scores_low():
    a = "Return only the committee name from the disclaimer."
    b = "First list every organization mentioned. Then pick the one after 'Paid for by'. Output that string."
    assert similarity(a, b) < 0.85


def test_empty_cases():
    assert similarity("", "") == 1.0
    assert similarity("", "x") == 0.0


def test_find_duplicates_sorted_and_filtered():
    text = "Return only the committee name from the disclaimer."
    matches = find_duplicates(text, [
        ("candidate 1", "Return only the committee name from the disclaimer!"),
        ("rejected entry 2", "Write a poem about the email."),
        ("candidate 3", "Return only the committee name from the disclaimer, nothing else."),
    ], threshold=0.85)
    assert [m[0] for m in matches] == ["candidate 1", "candidate 3"]
    assert matches[0][2] >= matches[1][2]


def test_render_duplicates_lists_label_and_text():
    out = render_duplicates([("candidate 1", "text one", 0.9), ("rejected entry 2", "text two", 0.88)])
    assert "1. candidate 1 (similarity 0.90)\ntext one" in out
    assert "2. rejected entry 2 (similarity 0.88)\ntext two" in out
