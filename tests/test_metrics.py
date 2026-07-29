"""Unit tests for ares.metrics — pure SQuAD-style EM/F1, recall, grounding.

These import the real functions and need only stdlib, so they run anywhere the
package imports (no torch/transformers/network).
"""
from ares import metrics


def test_normalize_lowercases_strips_punct_and_articles():
    assert metrics.normalize("The Quick, Brown FOX!") == "quick brown fox"
    # leading article + standalone articles are removed, whitespace collapsed
    assert metrics.normalize("a an the") == ""
    assert metrics.normalize("  Multiple   spaces  ") == "multiple spaces"


def test_normalize_handles_none_and_empty():
    assert metrics.normalize(None) == ""
    assert metrics.normalize("") == ""


def test_em_is_normalized_exact_match():
    assert metrics.em("Yes.", "yes") == 1.0
    assert metrics.em("The cat", "a cat") == 1.0  # articles stripped
    assert metrics.em("dog", "cat") == 0.0


def test_f1_identical_is_one():
    assert metrics.f1("Quentin Tarantino", "Quentin Tarantino") == 1.0


def test_f1_partial_overlap():
    # pred has 2 tokens, gold 1, overlap 1 -> prec=1/2, rec=1/1 -> F1=2/3
    assert abs(metrics.f1("Quentin Tarantino", "Tarantino") - (2 / 3)) < 1e-9


def test_f1_no_overlap_is_zero():
    assert metrics.f1("cat", "dog") == 0.0


def test_f1_both_empty_after_normalize_is_one():
    # two strings that normalize to empty are considered an exact (vacuous) match
    assert metrics.f1("the", "a") == 1.0


def test_f1_one_empty_is_zero():
    assert metrics.f1("", "cat") == 0.0
    assert metrics.f1("cat", "") == 0.0


def test_retrieval_recall():
    assert metrics.retrieval_recall(["A", "B", "C"], ["A", "B"]) == 1.0
    assert metrics.retrieval_recall(["A", "X"], ["A", "B"]) == 0.5
    assert metrics.retrieval_recall(["X"], ["A", "B"]) == 0.0


def test_retrieval_recall_empty_gold_is_zero():
    assert metrics.retrieval_recall(["A"], []) == 0.0
    assert metrics.retrieval_recall(["A"], None) == 0.0


def test_grounding_fraction_of_answer_tokens_in_context():
    # answer tokens {seine, river}; both in context -> 1.0
    assert metrics.grounding("Seine river", "The Seine is a river in France") == 1.0
    # only one of two answer tokens present -> 0.5
    assert metrics.grounding("Seine Tokyo", "The Seine flows through Paris") == 0.5


def test_grounding_empty_answer_is_zero():
    assert metrics.grounding("", "some context") == 0.0


def test_adversarial_variants_includes_prefix_and_optional_typo():
    kinds = dict(metrics.adversarial_variants("What is the capital city of France today"))
    assert "prefix" in kinds
    assert kinds["prefix"].startswith("Could you please tell me,")
    # long enough question + a long mid word -> a typo variant is produced
    assert "typo" in kinds


def test_adversarial_variants_short_question_has_no_typo():
    kinds = dict(metrics.adversarial_variants("Why so"))
    assert "typo" not in kinds
    assert "prefix" in kinds
