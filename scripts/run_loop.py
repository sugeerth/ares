"""Run the ARES self-improvement loop for a time budget, then write FINDINGS.md."""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ares import optimize, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=95)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"=== ARES loop: {args.minutes} min, eval_n={args.eval_n} ===", flush=True)
    optimize.run(minutes=args.minutes, eval_n=args.eval_n, seed=args.seed)
    print("\n=== FINDINGS ===\n", flush=True)
    print(report.generate(), flush=True)


if __name__ == "__main__":
    main()
