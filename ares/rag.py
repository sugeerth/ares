"""Hybrid retrieval (dense + BM25) over a per-example paragraph set."""
from __future__ import annotations
import os, re, threading
import numpy as np
from rank_bm25 import BM25Okapi

_EMB = None
_LOCK = threading.Lock()
EMB_NAME = os.environ.get("ARES_EMB", "sentence-transformers/all-MiniLM-L6-v2")


def _embedder():
    global _EMB
    if _EMB is None:
        with _LOCK:
            if _EMB is None:
                from sentence_transformers import SentenceTransformer
                dev = "mps"
                try:
                    import torch
                    if not torch.backends.mps.is_available():
                        dev = "cpu"
                except Exception:
                    dev = "cpu"
                _EMB = SentenceTransformer(EMB_NAME, device=dev)
    return _EMB


def _tok(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    if rng < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def hybrid_retrieve(query: str, paragraphs: list[dict], k: int = 4, alpha: float = 0.5):
    """paragraphs: [{'title':..., 'text':...}]. alpha=1 pure dense, alpha=0 pure BM25.
    Returns (top_paragraphs, scores)."""
    if not paragraphs:
        return [], []
    texts = [p["text"] for p in paragraphs]
    emb = _embedder()
    qv = emb.encode([query], normalize_embeddings=True)[0]
    dv = emb.encode(texts, normalize_embeddings=True)
    dense = dv @ qv
    bm = BM25Okapi([_tok(t) for t in texts])
    sparse = np.asarray(bm.get_scores(_tok(query)), dtype=float)
    score = alpha * _norm(dense) + (1.0 - alpha) * _norm(sparse)
    order = np.argsort(-score)[:k]
    return [paragraphs[i] for i in order], [float(score[i]) for i in order]


def format_context(paragraphs: list[dict], max_chars: int = 2400) -> str:
    out, used = [], 0
    for i, p in enumerate(paragraphs):
        block = f"[{i+1}] {p.get('title','')}: {p['text']}"
        if used + len(block) > max_chars:
            block = block[: max_chars - used]
        out.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n".join(out)
