"""Agent trajectory & tool-use evaluation.

Pure, side-effect-free analysis of the ``steps`` list returned by
:func:`ares.agent.answer`. None of these functions execute the model or touch
the GPU -- they only inspect the already-recorded trajectory structure.

The trajectory is a list of step dicts, each with a ``"step"`` name. The agent
in :mod:`ares.agent` emits these step kinds:

* ``{"step": "decompose", "subs": [...]}``      -- question decomposition
* ``{"step": "generate", "answer": ...}``       -- first answer attempt
* ``{"step": "reflect_retry", "answer": ...}``  -- self-reflection re-answer

These helpers turn a raw trajectory into interpretable signals (was the task
decomposed? did the agent reflect and retry? how efficient was it?) and compare
the realized tool sequence against an expected one. ``reflection_effectiveness``
uses :func:`ares.metrics.f1` to quantify whether a reflect-retry actually
improved the answer against the gold reference.
"""
from __future__ import annotations

from typing import Optional

from . import metrics

# Canonical step names emitted by ares.agent.answer. The agent records a
# "reflect_retry" step only when self-reflection rejected the first answer and
# a second retrieval+generation pass was performed.
_DECOMPOSE = "decompose"
_GENERATE = "generate"
_REFLECT_RETRY = "reflect_retry"


def _step_name(step) -> str:
    """Return the ``"step"`` key of a trajectory entry, or "" if malformed."""
    if isinstance(step, dict):
        return step.get("step", "") or ""
    return ""


def _step_names(steps) -> list:
    """Ordered list of step names from a trajectory (tolerant of None/junk)."""
    if not steps:
        return []
    return [_step_name(s) for s in steps]


def tool_use_summary(steps) -> dict:
    """Summarize which agent capabilities ("tools") fired in a trajectory.

    Args:
        steps: The ``steps`` list from :func:`ares.agent.answer`. Each element
            is a dict with a ``"step"`` name; ``decompose`` steps also carry a
            ``"subs"`` list of sub-questions.

    Returns:
        dict with keys:
            * ``decomposed`` (bool): a decompose step occurred.
            * ``n_subquestions`` (int): number of sub-questions produced by the
              decompose step (0 if no decomposition).
            * ``reflected`` (bool): the agent ran a self-reflection retry.
            * ``retried`` (bool): an answer was regenerated after the first one
              (alias of ``reflected`` for the current agent, kept separate so
              future multi-retry agents report it independently).
            * ``n_steps`` (int): total number of trajectory steps.
    """
    names = _step_names(steps)

    n_subquestions = 0
    for s in steps or []:
        if _step_name(s) == _DECOMPOSE:
            subs = s.get("subs") if isinstance(s, dict) else None
            if isinstance(subs, (list, tuple)):
                n_subquestions = len(subs)
            break

    reflected = _REFLECT_RETRY in names
    # A "retry" is any answer-producing step beyond the first generate.
    answer_steps = sum(1 for n in names if n in (_GENERATE, _REFLECT_RETRY))
    retried = answer_steps > 1

    return {
        "decomposed": _DECOMPOSE in names,
        "n_subquestions": n_subquestions,
        "reflected": reflected,
        "retried": retried,
        "n_steps": len(names),
    }


def step_efficiency(steps) -> float:
    """Efficiency score in ``(0, 1]`` -- fewer steps scores higher.

    For two trajectories that reach an equally good answer, the one using fewer
    steps is more efficient and should score higher. The score is ``1 / n``
    where ``n`` is the number of steps, so a single-step trajectory scores 1.0
    and longer trajectories decay toward 0. An empty trajectory scores 0.0.

    Args:
        steps: The ``steps`` list from :func:`ares.agent.answer`.

    Returns:
        float in ``[0, 1]``.
    """
    n = len(_step_names(steps))
    if n <= 0:
        return 0.0
    return 1.0 / n


def trajectory_score(steps, expected_tools: Optional[list] = None) -> dict:
    """Score the realized tool sequence against an expected sequence.

    Compares the ordered list of step names actually taken against
    ``expected_tools`` using a longest-common-subsequence (LCS) overlap. LCS is
    used (rather than exact equality) so that extra or missing steps degrade the
    score gracefully while preserving credit for steps that appear in the right
    relative order.

    If ``expected_tools`` is None, there is no reference to compare against, so
    the score falls back to :func:`step_efficiency` (a shorter trajectory is
    preferred when no plan is specified).

    Args:
        steps: The ``steps`` list from :func:`ares.agent.answer`.
        expected_tools: Optional ordered list of expected step names, e.g.
            ``["decompose", "generate"]``.

    Returns:
        dict with keys:
            * ``score`` (float in ``[0, 1]``): trajectory match quality.
            * ``detail`` (dict): diagnostics including the actual sequence, the
              expected sequence, the matched (LCS) length, and the names of
              missing / extra steps relative to the expectation.
    """
    actual = _step_names(steps)

    if expected_tools is None:
        return {
            "score": step_efficiency(steps),
            "detail": {
                "actual": actual,
                "expected": None,
                "matched": len(actual),
                "missing": [],
                "extra": [],
                "mode": "efficiency_fallback",
            },
        }

    expected = list(expected_tools)
    lcs = _lcs_length(actual, expected)

    # Normalize by the longer of the two sequences so that both missing and
    # spurious steps are penalized. Two empty sequences are a perfect (vacuous)
    # match and score 1.0.
    denom = max(len(actual), len(expected))
    score = 1.0 if denom == 0 else lcs / denom

    missing = _multiset_difference(expected, actual)
    extra = _multiset_difference(actual, expected)

    return {
        "score": score,
        "detail": {
            "actual": actual,
            "expected": expected,
            "matched": lcs,
            "missing": missing,
            "extra": extra,
            "mode": "lcs",
        },
    }


def reflection_effectiveness(before_answer: str, after_answer: str, gold: str) -> dict:
    """Measure whether a reflect-retry improved the answer, via token F1.

    Uses :func:`ares.metrics.f1` to score the pre-reflection and
    post-reflection answers against the gold reference, and reports the change.

    Args:
        before_answer: The answer before reflection (first ``generate`` step).
        after_answer: The answer after the ``reflect_retry`` step. If the agent
            did not retry, pass the same string as ``before_answer``; the delta
            will be 0.0.
        gold: The gold reference answer.

    Returns:
        dict with keys:
            * ``f1_before`` (float): F1 of the pre-reflection answer.
            * ``f1_after`` (float): F1 of the post-reflection answer.
            * ``delta`` (float): ``f1_after - f1_before`` (positive = improved).
            * ``improved`` (bool): reflection strictly increased F1.
            * ``regressed`` (bool): reflection strictly decreased F1.
    """
    f1_before = metrics.f1(before_answer, gold)
    f1_after = metrics.f1(after_answer, gold)
    delta = f1_after - f1_before
    return {
        "f1_before": f1_before,
        "f1_after": f1_after,
        "delta": delta,
        "improved": delta > 0.0,
        "regressed": delta < 0.0,
    }


# --------------------------------------------------------------------------- #
# Internal sequence helpers
# --------------------------------------------------------------------------- #
def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of two lists (DP, O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _multiset_difference(items: list, other: list) -> list:
    """Items present in ``items`` but not accounted for by ``other`` (multiset).

    Each occurrence in ``other`` cancels one occurrence in ``items``, preserving
    ``items`` order in the result.
    """
    import collections

    remaining = collections.Counter(other)
    out = []
    for it in items:
        if remaining.get(it, 0) > 0:
            remaining[it] -= 1
        else:
            out.append(it)
    return out
