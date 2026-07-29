"""LLM-as-judge metrics for ARES.

This module implements *reference-free* (and reference-light) evaluation of
RAG answers by prompting an LLM to act as a judge. It complements the lexical
metrics in :mod:`ares.metrics` (EM/F1/grounding), which only measure surface
overlap, by asking a model to reason about semantic faithfulness, relevance and
overall quality.

Judges built on LLMs are convenient but biased. The functions here apply
well-documented mitigations so the scores are more trustworthy:

* **Position / order bias.** When comparing two candidate answers, an LLM
  tends to favor whichever appears first (or, in some models, last). We mitigate
  this in :func:`pairwise` by running *both* orderings (A,B) and (B,A) and only
  declaring a winner when the two runs agree; disagreement collapses to a tie.

* **Verbosity / length bias.** Judges often reward longer answers. The prompts
  explicitly instruct the judge to ignore length and style and to score only on
  the requested criterion (faithfulness, relevance, correctness, groundedness).

* **Score-extraction brittleness.** Small instruct models rarely emit perfect
  JSON. Instead of demanding strict JSON, we ask for a labeled line and parse it
  defensively with layered regex fallbacks (see :func:`_parse_score` and
  :func:`_parse_label`). Parsing never raises; it degrades to a neutral default.

* **Determinism.** All judging calls use ``temperature=0.0`` so verdicts are
  reproducible and the content-addressed cache in :mod:`ares.llm` can dedupe
  repeated judgments during a config search.

* **Self-consistency for verdicts.** Faithfulness is decided by mapping a model
  verdict token onto a fixed scale rather than trusting a free-form number, so a
  ``score`` and a ``verdict`` can never silently contradict each other.

All functions in this module are **pure**: they build a prompt, delegate to
:func:`ares.llm.generate`, and parse the result into a small dict. They never
execute training, mutate global state, or perform I/O of their own.
"""
from __future__ import annotations

import re
from typing import Optional

from . import llm

# ---------------------------------------------------------------------------
# Shared judge configuration
# ---------------------------------------------------------------------------

#: System prompt shared by all judges. Frames the model as a strict, calibrated
#: evaluator and pre-empts the most common LLM-judge biases (length, style,
#: position) before any task-specific instructions are added.
JUDGE_SYSTEM = (
    "You are a meticulous, impartial evaluation judge. "
    "Assess only the requested criterion. Ignore answer length, verbosity, "
    "formatting, and writing style. Do not be swayed by confident tone. "
    "Be strict: only credit claims that are actually supported. "
    "Always respond in the exact format requested."
)

#: Default rubric used by :func:`g_eval` when the caller does not supply one.
#: It blends correctness against the provided context with groundedness, the two
#: failure modes that matter most for extractive/agentic RAG.
DEFAULT_G_EVAL_RUBRIC = (
    "Correctness & Groundedness (1-5):\n"
    "  5 - Fully correct and every claim is directly supported by the context.\n"
    "  4 - Correct overall; minor unsupported detail or slight imprecision.\n"
    "  3 - Partially correct, or correct but only weakly grounded in the context.\n"
    "  2 - Mostly incorrect or largely unsupported by the context (hallucination).\n"
    "  1 - Wrong, irrelevant, or contradicted by the context."
)


def _truncate(text: str, limit: int = 4000) -> str:
    """Clip overly long context so a single judging prompt stays in budget.

    The judge only needs enough context to verify claims; an unbounded context
    would blow past the model window and dominate latency. Truncation is on a
    whitespace boundary where possible to avoid splitting a token mid-word.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    if sp > limit * 0.6:
        cut = cut[:sp]
    return cut + " ..."


# ---------------------------------------------------------------------------
# Robust parsing helpers
# ---------------------------------------------------------------------------

def _first_float(text: str) -> Optional[float]:
    """Return the first numeric token in ``text`` as a float, or ``None``."""
    m = re.search(r"[-+]?\d*\.?\d+", text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_score(text: str, lo: float, hi: float, default: float,
                 label: str = "score") -> float:
    """Extract a numeric score from free-form judge output and clamp to range.

    Parsing is layered, most-specific first, so a stray number elsewhere in the
    rationale cannot hijack the score:

    1. ``label: <n>`` / ``label = <n>`` on its own (e.g. ``Score: 4``).
    2. A fraction form ``<n>/<hi>`` (e.g. ``4/5`` or ``0.8/1``).
    3. The first number found anywhere in the text.

    If nothing parses, fall back to ``default``. The result is always clamped to
    ``[lo, hi]`` so a hallucinated ``"10"`` on a 1-5 scale becomes ``5``.
    """
    text = text or ""

    # 1. Labeled score, e.g. "Score: 4.5" or "rating = 3".
    m = re.search(rf"{re.escape(label)}\s*[:=]?\s*([-+]?\d*\.?\d+)", text, re.I)
    val: Optional[float] = float(m.group(1)) if m else None

    # 2. Fraction form, e.g. "4/5" or "0.80 / 1".
    if val is None:
        m = re.search(r"([-+]?\d*\.?\d+)\s*/\s*\d*\.?\d+", text)
        if m:
            val = float(m.group(1))

    # 3. First bare number anywhere.
    if val is None:
        val = _first_float(text)

    if val is None:
        return default
    return max(lo, min(hi, val))


def _parse_label(text: str, options, default: str) -> str:
    """Find the first whole-word option token (case-insensitive) in ``text``.

    ``options`` is an ordered iterable of candidate labels (e.g.
    ``("yes", "no")`` or ``("A", "B", "tie")``). The earliest occurrence in the
    text wins, which lets the judge state its verdict mid-sentence. Returns
    ``default`` when no option appears.
    """
    text = text or ""
    best_pos = None
    best_opt = default
    for opt in options:
        m = re.search(rf"\b{re.escape(opt)}\b", text, re.I)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_opt = opt
    return best_opt


def _rationale(text: str, fallback: str = "") -> str:
    """Collapse judge output to a single trimmed rationale string."""
    text = (text or "").strip()
    return " ".join(text.split()) if text else fallback


def _after_label(text: str, label: str) -> str:
    """Return the portion of ``text`` following the first ``label:`` marker.

    The judge prompts ask for labeled lines (``Rationale: ...``, ``Winner: ...``);
    this isolates the value after a label so a single un-labeled fallback (the
    whole ``text``) is used when the model omitted the marker. Centralizing this
    keeps the four judges consistent and avoids repeating the split logic.
    """
    marker = f"{label}:"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

def faithfulness(answer: str, context: str) -> dict:
    """Judge whether ``answer`` is faithful to (entailed by) ``context``.

    Implements claim-supported judging: the judge decides whether the answer's
    claims are supported, contradicted, or unsupported by the retrieved context.
    A low score is a hallucination signal independent of whether the answer is
    *correct* in the world.

    Returns a dict with:

    * ``score`` - float in ``[0, 1]`` (1 = fully supported, 0 = contradicted).
    * ``verdict`` - one of ``"supported"``, ``"partial"``, ``"unsupported"``,
      ``"contradicted"``.
    * ``rationale`` - the judge's short justification.

    The numeric score is derived from the verdict label (not a free-form number)
    so the two can never disagree.
    """
    context = _truncate(context)
    prompt = (
        "Decide whether the ANSWER is faithful to the CONTEXT, i.e. whether "
        "every claim in the answer is supported by the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond on two lines exactly:\n"
        "Verdict: <supported|partial|unsupported|contradicted>\n"
        "Rationale: <one sentence>"
    )
    out = llm.generate(prompt, system=JUDGE_SYSTEM, max_new_tokens=160,
                       temperature=0.0)
    text = out.get("text", "")

    verdict = _parse_label(
        text, ("contradicted", "unsupported", "partial", "supported"),
        default="unsupported",
    )
    score_map = {
        "supported": 1.0,
        "partial": 0.5,
        "unsupported": 0.0,
        "contradicted": 0.0,
    }
    rationale = _after_label(text, "Rationale")
    return {
        "score": score_map[verdict],
        "verdict": verdict,
        "rationale": _rationale(rationale, fallback=text),
    }


def answer_relevancy(question: str, answer: str) -> dict:
    """Judge how directly ``answer`` addresses ``question`` (reference-free).

    This is independent of correctness or grounding: it penalizes evasive,
    off-topic, or padded answers even if some sentence in them is true. Useful
    for catching "insufficient context" style non-answers and rambling output.

    Returns ``{"score": float in [0, 1], "rationale": str}``.
    """
    prompt = (
        "Rate how directly and completely the ANSWER addresses the QUESTION, "
        "from 0.0 (off-topic or evasive) to 1.0 (fully on-point). Judge only "
        "relevance to the question, not factual correctness.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond on two lines exactly:\n"
        "Score: <number between 0 and 1>\n"
        "Rationale: <one sentence>"
    )
    out = llm.generate(prompt, system=JUDGE_SYSTEM, max_new_tokens=120,
                       temperature=0.0)
    text = out.get("text", "")
    score = _parse_score(text, lo=0.0, hi=1.0, default=0.0, label="score")
    rationale = _after_label(text, "Rationale")
    return {"score": score, "rationale": _rationale(rationale, fallback=text)}


def g_eval(question: str, answer: str, context: str,
           rubric: Optional[str] = None) -> dict:
    """Score an answer 1-5 with a G-Eval style chain-of-thought-then-score.

    Following the G-Eval method, the judge is asked to *first* reason step by
    step against the rubric and *then* emit a single integer. Generating the
    reasoning before the number empirically improves calibration and gives an
    auditable explanation. When ``rubric`` is ``None`` the default
    correctness+groundedness rubric (:data:`DEFAULT_G_EVAL_RUBRIC`) is used.

    Returns ``{"score": float in [1, 5], "reasoning": str}``.
    """
    rubric = rubric or DEFAULT_G_EVAL_RUBRIC
    context = _truncate(context)
    prompt = (
        "Evaluate the ANSWER to the QUESTION using the RUBRIC and the CONTEXT.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "First think step by step, checking the answer against the rubric and "
        "the context. Then give your verdict.\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Score: <single integer from 1 to 5>"
    )
    out = llm.generate(prompt, system=JUDGE_SYSTEM, max_new_tokens=320,
                       temperature=0.0)
    text = out.get("text", "")
    score = _parse_score(text, lo=1.0, hi=5.0, default=3.0, label="score")

    reasoning = _after_label(text, "Reasoning")
    # Drop the trailing "Score:" tail from the reasoning if present.
    reasoning = re.split(r"Score\s*[:=]", reasoning, maxsplit=1)[0]
    return {"score": score, "reasoning": _rationale(reasoning, fallback=text)}


def _pairwise_once(question: str, first: str, second: str,
                   context: Optional[str]) -> tuple:
    """Run a single A-vs-B judgment and return ``(winner_label, rationale)``.

    ``first``/``second`` are the texts shown in slot A and slot B respectively;
    the caller is responsible for swapping them to average out position bias.
    The winner label is one of ``"A"``, ``"B"``, ``"tie"``.
    """
    ctx_block = ""
    if context:
        ctx_block = f"CONTEXT:\n{_truncate(context)}\n\n"
    prompt = (
        "Two answers (A and B) are given for the same question. Decide which is "
        "better. Judge by correctness and, if context is given, faithfulness to "
        "it. Ignore length and style. If they are equally good, answer 'tie'.\n\n"
        f"{ctx_block}"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER A:\n{first}\n\n"
        f"ANSWER B:\n{second}\n\n"
        "Respond on two lines exactly:\n"
        "Winner: <A|B|tie>\n"
        "Rationale: <one sentence>"
    )
    out = llm.generate(prompt, system=JUDGE_SYSTEM, max_new_tokens=160,
                       temperature=0.0)
    text = out.get("text", "")
    # Restrict the verdict search to the line after "Winner:" (before any
    # "Rationale:") so a stray "A"/"B" inside the rationale cannot override the
    # stated winner.
    winner_line = _after_label(text, "Winner").split("Rationale:", 1)[0]
    label = _parse_label(winner_line, ("tie", "A", "B"), default="tie")
    rationale = ""
    if "Rationale:" in text:
        rationale = _rationale(_after_label(text, "Rationale"))
    return label, rationale


def pairwise(question: str, answer_a: str, answer_b: str,
             context: Optional[str] = None) -> dict:
    """Compare two answers and pick a winner, mitigating position bias.

    Position bias (an LLM judge favoring the answer in slot A regardless of
    quality) is averaged out by running the judgment twice with the answers
    swapped:

    * round 1 shows (A, B); round 2 shows (B, A).
    * We translate each raw slot-verdict back to the *real* candidate.
    * If both rounds agree on the same real candidate, that candidate wins.
    * If they disagree (the judge flipped with position) or either round is a
      tie, the result is ``"tie"`` -- the only honest verdict under ambiguity.

    Returns ``{"winner": "A" | "B" | "tie", "rationale": str}`` where the winner
    refers to the original ``answer_a`` / ``answer_b``.
    """
    # Round 1: real A in slot A, real B in slot B.
    raw1, rat1 = _pairwise_once(question, answer_a, answer_b, context)
    real1 = {"A": "A", "B": "B", "tie": "tie"}[raw1]

    # Round 2: swap, so real B is now in slot A and real A in slot B.
    raw2, rat2 = _pairwise_once(question, answer_b, answer_a, context)
    real2 = {"A": "B", "B": "A", "tie": "tie"}[raw2]

    if real1 == real2 and real1 in ("A", "B"):
        winner = real1
        rationale = rat1 or rat2
    else:
        winner = "tie"
        rationale = (
            "Verdict flipped with answer order (position bias) or judged equal; "
            f"order1 -> {real1}, order2 -> {real2}. {rat1}"
        ).strip()

    return {"winner": winner, "rationale": _rationale(rationale)}
