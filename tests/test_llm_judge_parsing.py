"""Unit tests for ares.llm_judge's pure parsing helpers.

These cover the defensive score/label parsing and the shared `_after_label`
helper extracted during refactoring. They do not invoke an LLM (the helpers are
pure string functions), so they run with only stdlib.
"""
from ares import llm_judge as j


# --- _first_float ---------------------------------------------------------- #
def test_first_float_parses_leading_number():
    assert j._first_float("the score is 4.5 out of 5") == 4.5
    assert j._first_float("-2 and 3") == -2.0
    assert j._first_float("no numbers here") is None
    assert j._first_float("") is None


# --- _parse_score ---------------------------------------------------------- #
def test_parse_score_prefers_labeled_value():
    # labeled "Score: 4" wins over a stray number in surrounding text
    assert j._parse_score("blah 9 ... Score: 4", lo=1, hi=5, default=3) == 4.0


def test_parse_score_falls_back_to_fraction_then_bare():
    assert j._parse_score("rated 4/5 overall", lo=1, hi=5, default=3) == 4.0
    assert j._parse_score("just 2 here", lo=1, hi=5, default=3) == 2.0


def test_parse_score_clamps_to_range():
    # hallucinated 10 on a 1-5 scale clamps to 5
    assert j._parse_score("Score: 10", lo=1, hi=5, default=3) == 5.0
    assert j._parse_score("Score: -4", lo=0, hi=1, default=0.5) == 0.0


def test_parse_score_uses_default_when_unparseable():
    assert j._parse_score("totally non-numeric", lo=1, hi=5, default=3) == 3.0
    assert j._parse_score("", lo=1, hi=5, default=3) == 3.0


# --- _parse_label ---------------------------------------------------------- #
def test_parse_label_earliest_option_wins():
    # The earliest whole-word option (case-insensitive) wins regardless of the
    # order of `options`: here "tie" precedes any standalone "A"/"B".
    assert j._parse_label("tie is best, A is worse", ("A", "B", "tie"),
                          default="x") == "tie"


def test_parse_label_is_case_insensitive():
    # a lowercase standalone "a" matches the option "A"
    assert j._parse_label("the answer is a", ("A", "B", "tie"), default="x") == "A"


def test_parse_label_whole_word_only():
    # "Bad" should not match the option "B" (word boundary)
    assert j._parse_label("Bad answer", ("A", "B", "tie"), default="tie") == "tie"


def test_parse_label_default_when_absent():
    assert j._parse_label("nothing matches", ("yes", "no"), default="no") == "no"


# --- _rationale ------------------------------------------------------------ #
def test_rationale_collapses_whitespace():
    assert j._rationale("  it   is\n good  ") == "it is good"
    assert j._rationale("", fallback="fb") == "fb"


# --- _after_label (extracted helper) --------------------------------------- #
def test_after_label_returns_text_following_marker():
    assert j._after_label("Verdict: x\nRationale: because reasons",
                          "Rationale") == " because reasons"


def test_after_label_returns_full_text_when_marker_missing():
    assert j._after_label("no marker present", "Rationale") == "no marker present"


def test_after_label_splits_on_first_occurrence_only():
    assert j._after_label("Rationale: a Rationale: b", "Rationale") == " a Rationale: b"


def test_after_label_equivalent_to_inline_split_pattern():
    """Behavior-preservation: matches the inline pattern the refactor replaced."""
    def inline(text, label):
        marker = f"{label}:"
        if marker in text:
            return text.split(marker, 1)[1]
        return text

    for text in ["Rationale: ok", "no marker", "Winner: A\nRationale: b", "Rationale:"]:
        for label in ("Rationale", "Winner", "Reasoning"):
            assert j._after_label(text, label) == inline(text, label)
