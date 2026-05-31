"""Evaluation metrics: SQuAD-style EM/F1, retrieval recall vs gold supporting
titles, lexical groundedness, and an adversarial-robustness helper."""
from __future__ import annotations
import re, string, collections


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def retrieval_recall(retrieved_titles, gold_titles) -> float:
    g = set(gold_titles or [])
    if not g:
        return 0.0
    return len(set(retrieved_titles) & g) / len(g)


def grounding(answer: str, context: str) -> float:
    """Fraction of answer content-tokens present in the retrieved context.
    Low grounding on a wrong answer is a hallucination signal."""
    a = set(normalize(answer).split())
    c = set(normalize(context).split())
    if not a:
        return 0.0
    return len(a & c) / len(a)


def adversarial_variants(question: str):
    """Cheap perturbations to probe robustness (no LLM needed)."""
    variants = []
    # typo: swap two adjacent chars in a mid word
    words = question.split()
    if len(words) > 3:
        w = words[len(words) // 2]
        if len(w) > 3:
            w2 = w[0] + w[2] + w[1] + w[3:]
            variants.append(("typo", " ".join(words[: len(words) // 2] + [w2] + words[len(words) // 2 + 1:])))
    # polite prefix (distraction)
    variants.append(("prefix", "Could you please tell me, " + question[0].lower() + question[1:]))
    return variants
