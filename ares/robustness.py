"""Adversarial robustness harness for the ARES agentic-RAG system.

What robustness tells you
-------------------------
A model can post a high F1 on a clean benchmark and still be brittle: a single
typo, a reworded prompt, an irrelevant distractor sentence, or a negated framing
can flip its answer. Aggregate accuracy hides this because it averages over
*independent* questions, whereas robustness asks a different question — does the
answer stay *stable* when the **same** question is perturbed in a meaning-
preserving (or meaning-probing) way?

This module produces semantically-controlled perturbations of a question and
measures how far quality drops under them:

  * ``base_f1``            F1 of the answer on the original question.
  * ``perturbed_f1_mean``  mean F1 across all perturbations.
  * ``stability``          perturbed_f1_mean / base_f1, clamped to [0, 1]; how
                           much of the clean quality survives perturbation. 1.0
                           means perturbations had no effect; values near 0 mean
                           the system collapses under trivial input noise.
  * ``drops``              per-perturbation breakdown so you can see *which*
                           perturbation kind hurts (e.g. typos vs. negation).

High average F1 + low stability is the classic "brittle benchmark champion"
signature and is exactly what an adversarial harness is meant to surface.

Design notes
------------
* Perturbations are pure string transforms — no LLM is invoked here, so probing
  is cheap and deterministic.
* The model call is *injected* via ``answer_fn`` rather than imported, so the
  caller decides when (and on which device) the GPU runs. ``answer_fn`` is
  expected to have the same contract as :func:`ares.agent.answer`, i.e.
  ``answer_fn(example, cfg) -> {"answer", "retrieved_titles", "context", ...}``.
* Everything in this module is a pure function with no module-level execution.
"""
from __future__ import annotations

import re
from typing import Callable

from . import metrics

# Perturbation kinds emitted by :func:`perturb`. Useful for callers that want to
# whitelist/blacklist kinds or pre-allocate per-kind reporting.
PERTURBATION_KINDS = (
    "typo",
    "paraphrase_template",
    "distractor_prefix",
    "negation_probe",
)

# Sentences that carry no answer-bearing signal; prepended to test whether the
# system gets distracted by irrelevant lead-in text.
_DISTRACTORS = (
    "I was reading something unrelated earlier, but anyway.",
    "Ignore the weather for a moment.",
    "Just out of curiosity, and this is not important,",
)


def _typo(question: str) -> str | None:
    """Introduce one realistic typo by swapping two adjacent characters inside a
    reasonably long mid-sentence word. Returns ``None`` if no suitable word."""
    words = question.split()
    if len(words) < 2:
        return None
    # Walk outward from the middle to find a word long enough to corrupt.
    mid = len(words) // 2
    order = sorted(range(len(words)), key=lambda i: abs(i - mid))
    for idx in order:
        w = words[idx]
        # Only swap within the alphabetic core so punctuation stays put.
        m = re.match(r"^(\w{4,})(\W*)$", w)
        if not m:
            continue
        core, tail = m.group(1), m.group(2)
        # Swap the 2nd and 3rd characters (interior, so it still reads as a typo).
        swapped = core[0] + core[2] + core[1] + core[3:]
        if swapped == core:
            continue
        new_words = list(words)
        new_words[idx] = swapped + tail
        return " ".join(new_words)
    return None


def _paraphrase_template(question: str) -> str:
    """Wrap the question in a meaning-preserving template. The semantics are
    unchanged, so a robust system should give the same answer."""
    q = question.strip().rstrip("?").strip()
    if not q:
        return question
    lead = q[0].lower() + q[1:]
    return f"Based on the passages, can you tell me {lead}?"


def _distractor_prefix(question: str, seed: int = 0) -> str:
    """Prepend an irrelevant lead-in sentence. The answer should be unaffected;
    if it changes, the system is anchoring on noise."""
    distractor = _DISTRACTORS[seed % len(_DISTRACTORS)]
    return f"{distractor} {question}"


def _negation_probe(question: str) -> str:
    """A meaning-*probing* rather than meaning-preserving perturbation: insert a
    'not' to test whether the system reads the question carefully instead of
    pattern-matching to the original. Because this changes intent, a drop here
    is less alarming than on the meaning-preserving kinds, but a system that
    answers *identically* to the original is clearly not reading the negation."""
    q = question.strip()
    # Try to negate the leading auxiliary/wh+aux so it reads naturally.
    replacements = [
        (r"^(Is)\b", r"Is it not the case that"),
        (r"^(Are)\b", r"Are there not"),
        (r"^(Was)\b", r"Was it not"),
        (r"^(Were)\b", r"Were there not"),
        (r"^(Does)\b", r"Does it not hold that"),
        (r"^(Do)\b", r"Do we not find that"),
        (r"^(Did)\b", r"Did it not happen that"),
    ]
    for pat, repl in replacements:
        new_q, n = re.subn(pat, repl, q, count=1, flags=re.IGNORECASE)
        if n:
            return new_q
    # Fallback for wh-questions / declaratives: ask for what is *not* the case.
    return f"Which of the following is NOT true: {q}"


def perturb(question: str) -> list[dict]:
    """Generate adversarial variants of ``question``.

    Returns a list of ``{"kind": str, "text": str}`` dicts, one per applicable
    perturbation kind in :data:`PERTURBATION_KINDS`. The ``typo`` kind is skipped
    when the question has no word long enough to corrupt, so the list length may
    be 3 or 4. Pure function: no LLM, deterministic for a given input.

    Kinds:
      * ``typo``                  one adjacent-character swap (meaning-preserving)
      * ``paraphrase_template``   reworded via a fixed template (meaning-preserving)
      * ``distractor_prefix``     irrelevant lead-in sentence (meaning-preserving)
      * ``negation_probe``        negated framing (meaning-probing)
    """
    variants: list[dict] = []

    typo = _typo(question)
    if typo is not None and typo != question:
        variants.append({"kind": "typo", "text": typo})

    variants.append({"kind": "paraphrase_template", "text": _paraphrase_template(question)})
    variants.append({"kind": "distractor_prefix", "text": _distractor_prefix(question)})
    variants.append({"kind": "negation_probe", "text": _negation_probe(question)})

    return variants


def robustness_eval(example: dict, cfg, answer_fn: Callable[[dict, object], dict]) -> dict:
    """Measure how stable an agent's answer is under adversarial perturbations.

    Parameters
    ----------
    example : dict
        ``{"question", "answer", "paragraphs": [{title, text}], "gold_titles": [...]}``.
    cfg :
        Agent configuration (an :class:`ares.agent.Config`), passed through
        unchanged to ``answer_fn``.
    answer_fn : callable
        Injected model entry point with the same contract as
        :func:`ares.agent.answer`: ``answer_fn(example, cfg) -> {"answer", ...}``.
        This is *the only* thing that touches the GPU; this function never
        imports or calls the model itself, so the caller controls when it runs.

    Returns
    -------
    dict
        ``{"base_f1", "perturbed_f1_mean", "stability", "drops": [...]}`` where
        each entry of ``drops`` is
        ``{"kind", "question", "answer", "f1", "drop"}`` and ``drop`` is
        ``base_f1 - f1`` (positive means the perturbation hurt). Per-perturbation
        failures are caught and scored as F1 0.0 so one bad variant cannot crash
        the harness.
    """
    gold = example.get("answer", "")

    base = answer_fn(example, cfg)
    base_f1 = metrics.f1(base.get("answer", ""), gold)

    drops: list[dict] = []
    perturbed_f1s: list[float] = []
    for variant in perturb(example.get("question", "")):
        # Reuse the same paragraphs/gold so only the question text changes.
        perturbed_example = dict(example)
        perturbed_example["question"] = variant["text"]
        try:
            out = answer_fn(perturbed_example, cfg)
            ans = out.get("answer", "")
            pf1 = metrics.f1(ans, gold)
        except Exception as ex:  # noqa: BLE001 — never let one variant kill the loop
            ans = f"<error: {ex}>"
            pf1 = 0.0
        perturbed_f1s.append(pf1)
        drops.append({
            "kind": variant["kind"],
            "question": variant["text"],
            "answer": ans,
            "f1": pf1,
            "drop": base_f1 - pf1,
        })

    perturbed_f1_mean = sum(perturbed_f1s) / len(perturbed_f1s) if perturbed_f1s else 0.0
    # Stability: share of clean quality retained, clamped to [0, 1]. If the model
    # was already wrong on the clean question (base_f1 == 0) stability is
    # undefined; report 0.0 since there is no clean quality to preserve.
    if base_f1 <= 0.0:
        stability = 0.0
    else:
        stability = max(0.0, min(1.0, perturbed_f1_mean / base_f1))

    return {
        "base_f1": base_f1,
        "perturbed_f1_mean": perturbed_f1_mean,
        "stability": stability,
        "drops": drops,
    }
