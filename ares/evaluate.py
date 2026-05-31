"""Run one Config over a set of examples and aggregate metrics. Robust per-example:
a failure scores zero rather than crashing the 2-hour loop."""
from __future__ import annotations
import numpy as np
from . import agent, metrics


def evaluate(examples, cfg, progress=None):
    rows = []
    for i, e in enumerate(examples):
        try:
            r = agent.answer(e, cfg)
            rows.append({
                "em": metrics.em(r["answer"], e["answer"]),
                "f1": metrics.f1(r["answer"], e["answer"]),
                "recall": metrics.retrieval_recall(r["retrieved_titles"], e["gold_titles"]),
                "grounding": metrics.grounding(r["answer"], r["context"]),
                "latency": r["latency"], "tokens": r["tokens"],
                "pred": r["answer"], "gold": e["answer"],
            })
        except Exception as ex:  # noqa: BLE001
            rows.append({"em": 0.0, "f1": 0.0, "recall": 0.0, "grounding": 0.0,
                         "latency": 0.0, "tokens": 0, "error": str(ex)})
        if progress:
            progress(i + 1, len(examples))
    keys = ["em", "f1", "recall", "grounding", "latency", "tokens"]
    agg = {k: float(np.mean([r[k] for r in rows])) if rows else 0.0 for k in keys}
    agg["n"] = len(rows)
    agg["errors"] = sum(1 for r in rows if "error" in r)
    return agg, rows
