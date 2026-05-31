"""Demo the LangGraph agent on one example, printing the node-by-node trace."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ares import corpus, langgraph_agent


def main():
    ex = corpus.FALLBACK[1]
    print("Q:", ex["question"])
    print("gold:", ex["answer"])
    out = langgraph_agent.run_one(ex)
    print("\n--- trace ---")
    for t in out.get("trace", []):
        print(json.dumps(t))
    print("\nfinal answer:", out.get("answer"))


if __name__ == "__main__":
    main()
