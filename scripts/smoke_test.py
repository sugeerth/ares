"""Tiny end-to-end check: load the local model, run the baseline agent on 3
embedded examples, compute metrics. First run downloads the model (~3GB) + embedder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ares import corpus, evaluate, llm
from ares.agent import Config


def main():
    print(f"device = {llm.device()}")
    ex = corpus.FALLBACK[:3]
    print("running baseline on 3 examples (first run downloads model + embedder)...")
    agg, rows = evaluate.evaluate(ex, Config(name="baseline"))
    print("AGG:", {k: round(v, 3) if isinstance(v, float) else v for k, v in agg.items()})
    for r in rows:
        print(f"  pred={r.get('pred')!r} | gold={r.get('gold')!r} | f1={round(r['f1'],2)} | err={r.get('error')}")
    print("llm stats:", llm.stats())
    assert agg["n"] == 3, "expected 3 rows"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
