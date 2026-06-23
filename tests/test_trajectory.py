"""Unit tests for ares.trajectory — pure trajectory/tool-use analysis.

All functions inspect already-recorded step dicts; nothing touches the model.
"""
from ares import trajectory


def _traj(*names):
    return [{"step": n} for n in names]


def test_tool_use_summary_full_trajectory():
    steps = [
        {"step": "decompose", "subs": ["q1", "q2"]},
        {"step": "generate", "answer": "a"},
        {"step": "reflect_retry", "answer": "b"},
    ]
    s = trajectory.tool_use_summary(steps)
    assert s["decomposed"] is True
    assert s["n_subquestions"] == 2
    assert s["reflected"] is True
    assert s["retried"] is True
    assert s["n_steps"] == 3


def test_tool_use_summary_single_generate():
    s = trajectory.tool_use_summary(_traj("generate"))
    assert s == {
        "decomposed": False,
        "n_subquestions": 0,
        "reflected": False,
        "retried": False,
        "n_steps": 1,
    }


def test_tool_use_summary_tolerates_none_and_junk():
    s = trajectory.tool_use_summary(None)
    assert s["n_steps"] == 0
    s2 = trajectory.tool_use_summary([None, {"step": "generate"}, "garbage"])
    assert s2["n_steps"] == 3
    assert s2["decomposed"] is False


def test_step_efficiency_inverse_of_length():
    assert trajectory.step_efficiency(_traj("generate")) == 1.0
    assert trajectory.step_efficiency(_traj("a", "b")) == 0.5
    assert trajectory.step_efficiency([]) == 0.0
    assert trajectory.step_efficiency(None) == 0.0


def test_trajectory_score_no_expected_falls_back_to_efficiency():
    res = trajectory.trajectory_score(_traj("decompose", "generate"))
    assert res["score"] == 0.5
    assert res["detail"]["mode"] == "efficiency_fallback"
    assert res["detail"]["expected"] is None


def test_trajectory_score_exact_match_is_one():
    steps = _traj("decompose", "generate")
    res = trajectory.trajectory_score(steps, expected_tools=["decompose", "generate"])
    assert res["score"] == 1.0
    assert res["detail"]["missing"] == []
    assert res["detail"]["extra"] == []
    assert res["detail"]["mode"] == "lcs"


def test_trajectory_score_missing_and_extra_steps():
    # actual is missing "reflect_retry" and has an extra "generate"
    steps = _traj("decompose", "generate", "generate")
    res = trajectory.trajectory_score(
        steps, expected_tools=["decompose", "generate", "reflect_retry"]
    )
    # LCS = ["decompose","generate"] = 2; denom = max(3,3) = 3
    assert abs(res["score"] - (2 / 3)) < 1e-9
    assert res["detail"]["missing"] == ["reflect_retry"]
    assert res["detail"]["extra"] == ["generate"]


def test_trajectory_score_both_empty_is_one():
    res = trajectory.trajectory_score([], expected_tools=[])
    assert res["score"] == 1.0


def test_reflection_effectiveness_improved():
    res = trajectory.reflection_effectiveness(
        before_answer="wrong", after_answer="Paris", gold="Paris"
    )
    assert res["f1_before"] == 0.0
    assert res["f1_after"] == 1.0
    assert res["delta"] == 1.0
    assert res["improved"] is True
    assert res["regressed"] is False


def test_reflection_effectiveness_regressed():
    res = trajectory.reflection_effectiveness(
        before_answer="Paris", after_answer="wrong", gold="Paris"
    )
    assert res["improved"] is False
    assert res["regressed"] is True


def test_reflection_effectiveness_no_change():
    res = trajectory.reflection_effectiveness("Paris", "Paris", "Paris")
    assert res["delta"] == 0.0
    assert res["improved"] is False
    assert res["regressed"] is False


def test_lcs_length_internal():
    assert trajectory._lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert trajectory._lcs_length(["a", "b"], ["b", "a"]) == 1
    assert trajectory._lcs_length([], ["a"]) == 0


def test_multiset_difference_internal():
    assert trajectory._multiset_difference(["a", "a", "b"], ["a"]) == ["a", "b"]
    assert trajectory._multiset_difference(["a", "b"], ["a", "b", "c"]) == []
