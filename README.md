# ARES — Agentic RAG Evaluation Suite

**A self-improving agentic-RAG system with a built-in adversarial evaluation harness, running fully local on Apple Silicon (MPS).** ARES searches the *design space of a retrieval-augmented agent* — retrieval depth, dense/BM25 hybrid weighting, prompt style, query decomposition, and self-reflection — evaluates each configuration on a multi-hop QA benchmark, and reports the quality-vs-cost Pareto frontier.

> Built to answer a concrete question: *for an agentic RAG system, which design choices actually move faithfulness and accuracy — and what do they cost in latency?*

[![python](https://img.shields.io/badge/python-3.12-blue)](#) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![runs](https://img.shields.io/badge/runtime-local%20MPS-orange)](#)

---

## Results (latest run)

Searched **55 agentic-RAG configurations** on HotpotQA (multi-hop), fully local on Apple Silicon. Winner **`combo_best`** — **F1 0.378 (+28.7% over baseline)**, EM 0.30, retrieval recall 0.79, groundedness 0.83, **0.12 s/question** (`retrieval_k=4, hybrid_alpha=0.5, prompt=cot, reflect=True`).

**Headline finding: the lever is the prompt, not retrieval tuning.** Chain-of-thought prompting drove the entire gain (~+0.08 F1); sweeping `retrieval_k`/`hybrid_alpha` moved F1 by <0.02, and a `strict` prompt actively *hurt*. Retrieval recall and answer F1 were **decoupled** (k=8 reached recall 0.91 yet didn't top the F1 board) — i.e. the bottleneck is **answer extraction, not retrieval coverage**, a conclusion a single aggregate score would have hidden.

→ Full results + Pareto frontier: [`results/FINDINGS.md`](results/FINDINGS.md) · Evaluation methodology & SOTA techniques: [`AGENT_EVAL_TECHNIQUES.md`](AGENT_EVAL_TECHNIQUES.md)

---

## Why this exists

Most RAG demos show *a* pipeline. ARES instead treats the agent's configuration as something to **search and measure** — with an evaluation harness rigorous enough to trust the conclusions:

- **Agentic, not just retrieval** — a LangGraph-style pipeline: *decompose → hybrid-retrieve → generate → self-reflect/verify → re-retrieve*.
- **Evaluation-first** — EM/F1 against gold answers, **retrieval recall** against gold supporting passages, **lexical groundedness** (a hallucination signal), and **adversarial perturbations**.
- **Inference-efficiency aware** — every config is measured in latency and tokens, so results land on a **quality-vs-cost Pareto frontier**, not a single number.
- **Self-improving loop** — runs unattended: structured ablations → synthesize the best factors → neighbour-search the running best, checkpointing every config.
- **$0 / local** — runs on a Mac's MPS with a small instruct model (Qwen2.5-1.5B) and local embeddings; a one-line switch points it at Claude/OpenAI.

## Architecture

```
                ┌────────────── self-improvement loop (optimize.py) ──────────────┐
                │  ablations → synthesize best factors → neighbour search          │
                │  (time-bounded, checkpoints every config)                        │
                └───────────────┬──────────────────────────────────────────────────┘
                                │ Config(retrieval_k, hybrid_alpha, prompt, decompose, reflect)
                                ▼
  question ─►  decompose ─►  hybrid retrieve  ─►  generate  ─►  reflect/verify ─►  answer
              (optional)     dense (MiniLM) +      (local         (re-retrieve if
                             BM25, alpha-mixed      LLM)           ungrounded)
                                │                                     │
                                ▼                                     ▼
                        evaluate.py  ──►  EM · F1 · retrieval recall · groundedness · latency · tokens
                                │
                                ▼
                          report.py  ──►  FINDINGS.md  +  Pareto / top-config plots
```

## What it measures

| Metric | What it tells you |
|---|---|
| **F1 / EM** | answer correctness vs gold |
| **Retrieval recall** | did the retriever surface the gold supporting passages |
| **Groundedness** | fraction of the answer supported by retrieved context (low + wrong = hallucination) |
| **Latency / tokens** | inference cost per question — the other axis of the Pareto frontier |
| **Adversarial stability** | answer robustness under typo / distractor perturbations |

## Quickstart

```bash
pip install -r requirements.txt

python scripts/smoke_test.py            # tiny end-to-end check (first run downloads the model)
python scripts/run_loop.py --minutes 95 --eval-n 40   # the self-improvement loop
python scripts/demo_langgraph.py        # show the LangGraph agent trace on one question
```

Results land in `runs/`: `leaderboard.csv`, `results.jsonl`, `progress.log`, and a generated **`runs/FINDINGS.md`** with plots.

### Pointing it at a hosted model
```bash
export ARES_BACKEND=anthropic ANTHROPIC_API_KEY=...   # or ARES_BACKEND=openai OPENAI_API_KEY=...
```

## Design notes

- **Per-example corpus.** On HotpotQA (distractor), each question ships with ~10 paragraphs (2 gold + distractors); retrieval is graded against the gold supporting titles — a clean, self-contained RAG signal.
- **Content-addressed LLM cache.** Identical prompts (e.g. repeated decompositions) are memoised, so the config search spends compute only on genuinely new work.
- **MPS-safe.** `PYTORCH_ENABLE_MPS_FALLBACK=1` and fp16 keep the 2-hour run stable on Apple Silicon; the HF cache stays inside the project.
- **Robust loop.** Per-example failures score zero instead of crashing the run; every config is checkpointed, so a partial run is still a usable result.

## Repo layout

```
ares/        llm.py · rag.py · corpus.py · agent.py · metrics.py · evaluate.py · optimize.py · report.py · langgraph_agent.py
scripts/     smoke_test.py · run_loop.py · demo_langgraph.py
runs/        leaderboard.csv · results.jsonl · FINDINGS.md · plots/   (generated; git-ignored)
```

## Roadmap

- Add an LLM-as-judge faithfulness metric alongside the lexical signal.
- Reranker node (cross-encoder) as another searchable factor.
- Export the Pareto frontier as a deployable serving policy (cost-bounded config selection).

## License

MIT © 2026 Sugeerth Murugesan
