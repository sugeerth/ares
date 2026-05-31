"""Agentic RAG pipeline: (optional) decompose -> hybrid retrieve -> generate ->
(optional) self-reflect/verify -> re-retrieve+regenerate. All steps toggle via Config,
so the optimizer can search the agent's own design space."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from . import llm, rag

PROMPTS = {
    "plain": "Answer the question using ONLY the context. Reply with the shortest exact answer (a few words).\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:",
    "cot": "Use the context to reason briefly, then give the shortest final answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\nReason briefly, then end with a line 'Answer: <answer>'.",
    "strict": "You are a careful extractive QA system. Answer ONLY from the context. If the context does not contain the answer, reply 'insufficient context'. Give the shortest exact answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:",
}


@dataclass
class Config:
    name: str = "baseline"
    retrieval_k: int = 4
    hybrid_alpha: float = 0.5
    decompose: bool = False
    reflect: bool = False
    prompt_variant: str = "plain"
    max_new_tokens: int = 160

    def as_dict(self):
        return asdict(self)

    def key(self):
        return (self.retrieval_k, round(self.hybrid_alpha, 2), self.decompose,
                self.reflect, self.prompt_variant)


def _extract(text: str, variant: str) -> str:
    text = (text or "").strip()
    if variant == "cot" and "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return text[:200]


def decompose(question: str):
    out = llm.generate(
        "Break this question into 1-3 simpler sub-questions, one per line, no numbering.\n\n"
        f"Question: {question}", max_new_tokens=110)
    subs = [l.strip("-*•0123456789. ").strip() for l in out["text"].splitlines()]
    subs = [s for s in subs if len(s) > 6][:3]
    return (subs or [question]), out["latency"], out["tokens"]


def answer(example: dict, cfg: Config) -> dict:
    paras = example["paragraphs"]
    lat, tok, steps = 0.0, 0, []
    queries = [example["question"]]
    if cfg.decompose:
        subs, dl, dt = decompose(example["question"])
        lat += dl; tok += dt; queries = subs
        steps.append({"step": "decompose", "subs": subs})

    seen: dict[str, dict] = {}
    for q in queries:
        top, _ = rag.hybrid_retrieve(q, paras, k=cfg.retrieval_k, alpha=cfg.hybrid_alpha)
        for p in top:
            seen[p["title"]] = p
    cap = cfg.retrieval_k + 2 if cfg.decompose else cfg.retrieval_k
    retrieved = list(seen.values())[:cap]
    ctx = rag.format_context(retrieved)

    g = llm.generate(PROMPTS[cfg.prompt_variant].format(ctx=ctx, q=example["question"]),
                     max_new_tokens=cfg.max_new_tokens)
    lat += g["latency"]; tok += g["tokens"]
    ans = _extract(g["text"], cfg.prompt_variant)
    steps.append({"step": "generate", "answer": ans})

    if cfg.reflect:
        j = llm.generate(
            f"Context:\n{ctx}\n\nQuestion: {example['question']}\nProposed answer: {ans}\n\n"
            "Is the proposed answer fully supported by the context? Reply 'YES' or 'NO'.",
            max_new_tokens=12)
        lat += j["latency"]; tok += j["tokens"]
        if j["text"].strip().upper().startswith("NO"):
            top, _ = rag.hybrid_retrieve(f"{example['question']} {ans}", paras,
                                         k=cfg.retrieval_k + 2, alpha=cfg.hybrid_alpha)
            for p in top:
                seen[p["title"]] = p
            retrieved = list(seen.values())
            ctx = rag.format_context(retrieved)
            g2 = llm.generate(PROMPTS[cfg.prompt_variant].format(ctx=ctx, q=example["question"]),
                              max_new_tokens=cfg.max_new_tokens)
            lat += g2["latency"]; tok += g2["tokens"]
            ans = _extract(g2["text"], cfg.prompt_variant)
            steps.append({"step": "reflect_retry", "answer": ans})

    return {"answer": ans, "retrieved_titles": [p["title"] for p in retrieved],
            "context": ctx, "latency": lat, "tokens": tok, "steps": steps}
