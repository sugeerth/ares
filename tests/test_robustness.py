"""Unit tests for ares.robustness — pure string perturbations + the stability
aggregation in robustness_eval (driven by an injected fake answer_fn).
"""
from ares import robustness


def test_perturb_meaning_preserving_kinds_present():
    variants = robustness.perturb("What is the capital city of France")
    kinds = [v["kind"] for v in variants]
    assert "paraphrase_template" in kinds
    assert "distractor_prefix" in kinds
    assert "negation_probe" in kinds


def test_perturb_typo_swaps_interior_chars():
    variants = robustness.perturb("Which mountain is the highest in Brazil")
    typo = next((v for v in variants if v["kind"] == "typo"), None)
    assert typo is not None
    assert typo["text"] != "Which mountain is the highest in Brazil"


def test_perturb_skips_typo_when_no_long_word():
    # No word has a 4+ alphabetic core -> typo kind is skipped (3 variants).
    variants = robustness.perturb("a b c")
    kinds = [v["kind"] for v in variants]
    assert "typo" not in kinds
    assert len(variants) == 3


def test_paraphrase_template_is_meaning_preserving_wrapper():
    out = robustness._paraphrase_template("Where is Paris?")
    assert out == "Based on the passages, can you tell me where is Paris?"


def test_distractor_prefix_prepends_known_distractor():
    out = robustness._distractor_prefix("Where is Paris?", seed=1)
    assert out.endswith("Where is Paris?")
    assert out.startswith(robustness._DISTRACTORS[1])


def test_negation_probe_negates_leading_auxiliary():
    assert robustness._negation_probe("Is Paris in France?").startswith(
        "Is it not the case that"
    )
    assert robustness._negation_probe("Are the bands English?").startswith(
        "Are there not"
    )


def test_negation_probe_fallback_for_wh_question():
    out = robustness._negation_probe("Which river flows through Paris?")
    assert out.startswith("Which of the following is NOT true:")


def _fake_answer_fn(example, cfg):
    """Echo a fixed answer keyed off the question so we can drive f1 deterministically."""
    q = example["question"]
    # Return the gold for the clean question, a wrong answer for perturbations.
    return {"answer": "Paris" if q == "What is the capital of France" else "wrong"}


def test_robustness_eval_stability_drops_under_perturbation():
    example = {
        "question": "What is the capital of France",
        "answer": "Paris",
        "paragraphs": [],
        "gold_titles": [],
    }
    res = robustness.robustness_eval(example, cfg=None, answer_fn=_fake_answer_fn)
    assert res["base_f1"] == 1.0
    # every perturbed question returns "wrong" -> f1 0.0
    assert res["perturbed_f1_mean"] == 0.0
    assert res["stability"] == 0.0
    assert res["drops"]  # non-empty per-perturbation breakdown
    assert all(d["drop"] == 1.0 for d in res["drops"])


def test_robustness_eval_perfect_stability_when_answer_invariant():
    example = {"question": "stable q", "answer": "same", "paragraphs": [], "gold_titles": []}
    res = robustness.robustness_eval(
        example, cfg=None, answer_fn=lambda e, c: {"answer": "same"}
    )
    assert res["base_f1"] == 1.0
    assert res["perturbed_f1_mean"] == 1.0
    assert res["stability"] == 1.0


def test_robustness_eval_zero_base_f1_gives_zero_stability():
    example = {"question": "q", "answer": "gold", "paragraphs": [], "gold_titles": []}
    res = robustness.robustness_eval(
        example, cfg=None, answer_fn=lambda e, c: {"answer": "nevermatches"}
    )
    assert res["base_f1"] == 0.0
    assert res["stability"] == 0.0


def test_robustness_eval_one_bad_variant_does_not_crash():
    example = {"question": "q ok", "answer": "gold", "paragraphs": [], "gold_titles": []}
    calls = {"n": 0}

    def flaky(e, c):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"answer": "gold"}  # clean question succeeds
        raise RuntimeError("boom on a perturbation")

    res = robustness.robustness_eval(example, cfg=None, answer_fn=flaky)
    assert res["base_f1"] == 1.0
    # failed variants are scored 0.0 and recorded with an <error: ...> answer
    assert any(d["answer"].startswith("<error:") for d in res["drops"])
