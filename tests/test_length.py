from dspy.utils.dummies import DummyLM

from skilled_proposer.proposer import SkilledProposer, _count_tokens, _count_words


def _propose(proposer):
    return proposer(
        candidate={"a": "old"},
        reflective_dataset={"a": []},
        components_to_update=["a"],
    )["a"]


def test_no_limit_passes_through():
    lm = DummyLM([{"new_instruction": "A long enough instruction."}])
    assert _propose(SkilledProposer(prompt_model=lm)) == "A long enough instruction."


def test_within_word_budget_no_compression():
    lm = DummyLM([{"new_instruction": "Three word answer."}])
    assert _propose(SkilledProposer(prompt_model=lm, max_words=5)) == "Three word answer."


def test_compression_pass_used_when_over_budget():
    long_text = " ".join(["word"] * 20)
    lm = DummyLM(
        [
            {"new_instruction": long_text},
            {"shortened_instruction": "Short now."},
        ]
    )
    assert _propose(SkilledProposer(prompt_model=lm, max_words=5)) == "Short now."


def test_truncation_when_compression_still_over_budget():
    long_text = " ".join(f"w{i}" for i in range(20))
    still_long = " ".join(f"s{i}" for i in range(10))
    lm = DummyLM(
        [
            {"new_instruction": long_text},
            {"shortened_instruction": still_long},
        ]
    )
    result = _propose(SkilledProposer(prompt_model=lm, max_words=5))
    assert result == "s0 s1 s2 s3 s4"


def test_length_limit_text():
    assert SkilledProposer()._length_limit_text() == "None"
    assert (
        SkilledProposer(max_words=10)._length_limit_text()
        == "The new instruction must be at most 10 words."
    )
    assert (
        SkilledProposer(max_words=10, max_tokens=40)._length_limit_text()
        == "The new instruction must be at most 10 words and at most 40 tokens."
    )


def test_truncate_by_tokens_fallback_counter():
    # With no model name, _count_tokens is ~len/4. The truncated prefix
    # must fit the token budget; the full text must not.
    proposer = SkilledProposer(max_tokens=10)
    text = " ".join(["abcde"] * 40)
    truncated = proposer._truncate(text)
    assert _count_tokens(truncated, None) <= 10
    assert _count_words(truncated) < 40
    assert text.startswith(truncated)


def test_count_words():
    assert _count_words("one two  three\nfour") == 4
    assert _count_words("") == 0


def test_count_tokens_fallback():
    assert _count_tokens("x" * 40, None) == 10
    assert _count_tokens("", None) == 1
