# Agent & RAG Evaluation Techniques — A Working Reference

> A field guide to evaluating retrieval-augmented and agentic LLM systems: the metrics, the math, the failure modes, and the tooling — mapped onto what **ARES** (this repo) actually implements.
>
> **Audience:** a Staff-level AI Scientist prepping for agent-evaluation roles (Amazon *AI Agent Evaluation*, Scale *LLM Evals*, Anthropic *Model Evaluations*). It is written to be both a study sheet and a design doc: every technique comes with *when to use it*, *when not to*, and *how it breaks*.

The thesis throughout: **never report a single number.** A trustworthy eval is layered (retrieval vs. generation), multi-paradigm (reference-based vs. reference-free), bias-aware (LLM judges are correlated estimators, not oracles), and statistically honest (confidence intervals, paired tests). ARES is built as a concrete instance of that philosophy.

---

## 1. Intro & Taxonomy

Evaluating a RAG/agent system is not one measurement — it is a stack of measurements, each answering a different question. The two big organizing axes:

- **Stage:** *retrieval* (did we fetch the right context?) vs. *generation* (is the answer grounded, relevant, correct?). A faithful answer to the *wrong* retrieved context is faithful but useless — so generation metrics are meaningless without retrieval metrics beside them.
- **Paradigm:** *reference-based* (compare to a gold answer / labeled relevant docs — the gold standard for offline regression) vs. *reference-free* (judge the output against only the query and retrieved context — the workhorse in production, because gold labels are expensive).

A third axis appears once the system becomes an **agent**: you stop scoring a single string and start scoring a **trajectory** — a sequence of (thought, tool-call, observation) tuples — across three orthogonal dimensions: *outcome correctness*, *trajectory quality*, and *efficiency/reliability*.

### The evaluation map

```
                          ┌──────────────────────────────────────────────────────┐
                          │            WHAT ARE YOU EVALUATING?                    │
                          └──────────────────────────────────────────────────────┘
                                                   │
        ┌──────────────────┬───────────────────────┼───────────────────┬────────────────────┐
        ▼                  ▼                        ▼                   ▼                    ▼
  ┌───────────┐      ┌────────────┐          ┌─────────────┐    ┌──────────────┐     ┌──────────────┐
  │ RETRIEVAL │      │ GENERATION │          │   ANSWER    │    │    AGENT     │     │  ROBUSTNESS  │
  │  quality  │      │ faithful-  │          │   quality   │    │  TRAJECTORY  │     │ & RIGOR      │
  │           │      │   ness     │          │             │    │              │     │              │
  ├───────────┤      ├────────────┤          ├─────────────┤    ├──────────────┤     ├──────────────┤
  │ Hit@K     │      │ RAGAS      │          │ EM / F1     │    │ outcome /    │     │ perturbation │
  │ MRR       │      │  faithful- │          │ G-Eval      │    │  state check │     │  (PromptBench)│
  │ nDCG@K    │      │  ness      │          │ Answer      │    │ BFCL AST     │     │ contamination│
  │ Ctx Prec  │      │ NLI claim  │          │  Relevancy  │    │  tool-call   │     │ stat. signif.│
  │ Ctx Recall│      │  entailmt  │          │ pairwise /  │    │ pass@k vs    │     │  (bootstrap, │
  │           │      │ SelfCheck  │          │  Arena Elo  │    │  pass^k      │     │  McNemar)    │
  │           │      │ FActScore  │          │ MT-Bench    │    │ traj. judge  │     │ human align. │
  │           │      │ Sem.Entropy│          │             │    │  / ref-match │     │  (κ, ρ, α)   │
  └─────┬─────┘      └─────┬──────┘          └─────┬───────┘    └──────┬───────┘     └──────┬───────┘
        │                  │                       │                   │                    │
   reference-based    mostly reference-free   ref-free + ref-based  outcome=oracle    semantics-preserving
   (labeled docs)     (answer vs context)     (judge / gold)        traj=judge         perturbation +
        │                  │                       │                   │                CI / paired tests
        └──────────────────┴───────────┬───────────┴───────────────────┴────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  HARNESS: dataset → generate → score → aggregate → CI      │
                  │  + LLM-judge bias defenses + statistical significance      │
                  └──────────────────────────────────────────────────────────┘
```

**How to read it:** pick your row by what you're measuring, pick reference-based vs. reference-free by whether you have labels, and *always* close the loop at the bottom — a metric without a confidence interval and a bias defense is a vibe, not an evaluation.

---

## 2. RAG Evaluation Metrics — Reference-Free & Reference-Based

RAG evaluation decomposes into **retrieval** and **generation**, and into **reference-based** vs. **reference-free**. Reference-free is the production workhorse (gold labels are expensive); reference-based is the gold standard for offline regression testing.

### 2.1 RAGAS (EACL 2024)

RAGAS scores four metrics, all in `[0,1]`, mostly LLM-driven and reference-free by default.

**Faithfulness** = (# claims in answer entailed by context) / (total # claims in answer). Algorithm: (1) an LLM decomposes the answer into atomic claims; (2) each claim is verified for entailment against retrieved context; (3) take the supported fraction. Example: *"Einstein born in Germany on 20 March 1879"* → 2 claims, location ✓, date ✗ → **0.5**. A faster variant swaps step 2 for **Vectara HHEM-2.1-Open** (a fine-tuned T5 hallucination classifier) instead of an LLM judge.

**Answer / Response Relevancy** = mean cosine similarity between the embedding of the original question and `N` (default 3) questions an LLM *reverse-generates* from the answer:

```
AR = (1/N) Σ cos(E_gi, E_q)
```

multiplied by 0 if the answer is flagged "noncommittal" (e.g., "I don't know"). It penalizes incomplete/evasive answers and rewards on-topic ones — note it does **not** measure correctness.

**Context Precision@K** (ranking-aware):

```
CP@K = Σ_k (Precision@k · v_k) / (total relevant in top K),   v_k ∈ {0,1}  flags rank-k chunk relevance
```

Variants: `LLMContextPrecisionWithoutReference` (relevance judged vs. the response — reference-free), `LLMContextPrecisionWithReference` (vs. gold answer), `NonLLMContextPrecisionWithReference` (Levenshtein/string sim vs. reference contexts), and `IDBasedContextPrecision = |retrieved IDs ∩ reference IDs| / |retrieved IDs|`.

**Context Recall** = (# reference-answer claims attributable to retrieved context) / (total reference claims). This one is **reference-based** — it needs a ground-truth answer whose claims are checked against retrieved chunks. A non-LLM form uses `|relevant retrieved contexts| / |reference contexts|`.

```python
from ragas import evaluate
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithoutReference, LLMContextRecall)
result = evaluate(dataset, metrics=[Faithfulness(), ResponseRelevancy(),
                  LLMContextPrecisionWithoutReference(), LLMContextRecall()])
```

**Use** RAGAS for fast, label-light iteration. **Avoid** treating relevancy as accuracy, and avoid faithfulness as your only gate (see §2.6). Claim decomposition is judge-model sensitive and cost scales with claim count — pin the judge model.

### 2.2 ARES (NAACL 2024, Stanford) — the namesake

The academic **ARES** scores **context relevance, answer faithfulness, answer relevance** but, unlike RAGAS's prompt-only judges, it *fine-tunes lightweight judges* and statistically de-biases them:

1. **Synthetic data:** FLAN-T5 generates (query, answer) pairs from passages, plus contrastive *negatives* (wrong/irrelevant pairs) per dimension.
2. **Judge training:** a DeBERTa classifier is fine-tuned to discriminate positive vs. contrastive-negative examples — sharper than a zero-shot GPT prompt.
3. **Prediction-Powered Inference (PPI):** the judge labels a large unlabeled eval set, then a *small* human-labeled set (a few hundred points) corrects the judge's systematic bias, yielding a **point estimate plus a confidence interval** on each metric. This is the key differentiator — statistical guarantees, not a single unverified number.

The paper's ARES out-ranked RAGAS on KILT/SuperGLUE/AIS tasks, with Kendall's τ ~**0.065 higher for context relevance** and ~**0.132 higher for answer relevance**, and stayed robust under domain shift. **Use** when you need defensible, CI-backed rankings across many RAG configs and can afford a few hundred annotations. **Avoid** if you lack any labels or need per-example real-time scoring.

> Note on naming: this repo (**ARES — Agentic RAG Evaluation Suite**) shares the acronym but is a *practical harness* that searches an agent's design space; it borrows the philosophy (statistical honesty, reference-free + reference-based layering) rather than the exact PPI pipeline.

### 2.3 Retrieval Metrics (pure IR, against labeled relevant docs)

- **Hit Rate@K:** fraction of queries with ≥1 relevant doc in top-K. `Hit@1` is brutally honest for k=1–3 production setups.
- **MRR** = `(1/|Q|) Σ 1/rank_first_relevant` — only the *first* hit matters; ideal for single-answer QA.
- **nDCG@K** = `DCG@K / IDCG@K`, where `DCG@K = Σ_{i=1..K} rel_i / log2(i+1)`. Handles **graded** relevance and rank position; use for search/recommendation where many docs are partially useful.

```python
def ndcg_at_k(rels, k):
    import numpy as np
    dcg = sum(r/np.log2(i+2) for i,r in enumerate(rels[:k]))
    idcg = sum(r/np.log2(i+2) for i,r in enumerate(sorted(rels, reverse=True)[:k]))
    return dcg/idcg if idcg else 0.0
```

### 2.4 Claim-Level / NLI Faithfulness

Beyond RAGAS, decompose the answer into atomic claims and run an **NLI model** (e.g., DeBERTa-MNLI) for entailment vs. context — cheaper/faster than GPT-4 and more reproducible. **SummaC** aggregates sentence-pair NLI entailment; **Vectara HHEM** and **FaithJudge/FaithBench** (2025) provide hallucination leaderboards combining claim extraction + NLI entailment + HHEM scoring.

### 2.5 The RAG Triad (TruLens framing)

A clean mental model that maps directly onto ARES's metrics: **context relevance** (retrieval) + **groundedness** (answer-vs-context) + **answer relevance** (answer-vs-question). Passing all three implies low hallucination. Each leg is reference-free.

### 2.6 RAG/Faithfulness Failure Modes (build defenses)

- **LLM-judge bias:** length, position, and self-preference bias; different judge models give divergent scores — pin the judge model/version, randomize order, calibrate against human labels.
- **Metric disagreement:** faithfulness metrics disagree on identical outputs; ensemble (NLI + LLM + HHEM) and report variance.
- **Fluency/length conflation:** well-written wrong answers score high; awkward correct answers score low.
- **Faithfulness ≠ correctness:** a perfectly grounded answer to *wrong* retrieved context is faithful but useless — always pair generation metrics with retrieval metrics.
- **Claim-decomposition variance:** re-running decomposition yields different claim sets → score noise; fix `temperature=0` and average over seeds.

---

## 3. LLM-as-a-Judge — Methods, Formulas, Biases, Harness Patterns

LLM-as-a-judge replaces brittle n-gram metrics (BLEU/ROUGE) and expensive human panels by prompting a strong LLM to score or compare outputs. Treat judge scores as **correlated estimates, not ground truth.**

### 3.1 G-Eval (chain-of-thought rubric scoring; Liu et al., EMNLP 2023)

A *single-answer, reference-free, rubric-based* scorer in three steps: (1) a **task intro + evaluation criteria** prompt (e.g., "Coherence (1–5)"); (2) **auto-CoT** — the LLM generates detailed evaluation *steps* from the criteria once, appended to the prompt; (3) a **scoring function** using a form-filling paradigm. The key trick is *probability-weighted scoring*: instead of the single emitted integer, read the token log-probs over candidate scores `S = {s₁…sₙ}` and take the expectation:

```
score = Σ_i p(sᵢ) · sᵢ          (Eq. 1)
```

This yields continuous scores, breaking ties and improving correlation. On **SummEval**, G-Eval-4 reaches **Spearman ρ ≈ 0.514** averaged across dimensions, beating ROUGE/BERTScore/GPTScore; removing probabilities (`-Probs`) drops ρ and inflates Kendall-τ via ties.

```python
resp = client.chat.completions.create(
    model="gpt-4", messages=[{"role":"user","content":prompt}],
    temperature=1, n=20, max_tokens=5, logprobs=True)
# Either average n=20 samples, or use logprobs over {1..5}:
score = sum(prob[s]*s for s in range(1,6)) / sum(prob.values())
```

**Use** for offline NLG quality (summarization, dialogue, RAG faithfulness) when you need a graded rubric and have logprob access. **Avoid** when you lack logprobs (modern chat APIs often hide them — fall back to n-sampling and averaging), or when absolute calibration matters across runs (scores drift if the judge model changes).

### 3.2 Single-answer vs. pairwise vs. reference-guided (Zheng et al., MT-Bench, NeurIPS 2023)

- **Single-answer grading:** judge scores one answer 1–10. Cheap, `O(n)` calls, scales to many models, but the absolute scale is unstable — scores fluctuate run-to-run.
- **Pairwise comparison:** judge picks A, B, or tie. More reliable (relative > absolute judgment), but `O(n²)` pairs — needs sampling at scale.
- **Reference-guided:** judge is given a reference/gold solution before grading. On math, this cut GPT-4's judge failure rate from **70% → 15%**. Strongly recommended whenever ground truth or a good reference exists.

Reported human agreement is high: GPT-4 reaches **~85% agreement with human experts** on MT-Bench (ties excluded) — *above* the 81% human-human agreement, and >80% on crowdsourced Chatbot Arena.

### 3.3 MT-Bench & Chatbot Arena Elo

**MT-Bench:** 80 curated **multi-turn** questions (2 turns each) across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities); judged 1–10 by GPT-4. Probes instruction-following and conversation, not just single-shot QA.

**Chatbot Arena:** crowdsourced anonymous A/B battles. Originally online Elo:

```
E_A = 1 / (1 + 10^((R_B − R_A)/400));   R_A ← R_A + K·(S_A − E_A)
```

LMSYS later switched to a **Bradley-Terry MLE** fit (logistic regression on pairwise outcomes) — the maximum-likelihood estimate of latent Elo strengths assuming order-independence:

```
P(A beats B) = σ(β_A − β_B)
```

with **bootstrapped confidence intervals** (ties = half-win/half-loss). BT is more stable and gives precise CIs vs. order-dependent online Elo.

### 3.4 Judge biases (and mitigations)

| Bias | Evidence | Mitigation |
|---|---|---|
| **Position** | Swap-consistency was low: Claude-v1 23.8%, GPT-3.5 46.2%, GPT-4 65%. | Run both orders; count a win only if consistent in both, else tie. |
| **Verbosity** | "Repetitive list" attack failure: Claude-v1 **91.3%**, GPT-3.5 91.3%, GPT-4 8.7%. | Explicit "do not let length influence you" + length-controlled pairs. |
| **Self-enhancement** | GPT-4 +10% win rate on its own outputs; Claude-v1 +25%. | Use a different judge family than the generator; use a jury/panel. |
| **Limited reasoning** | Mis-grades math/logic. | CoT + reference-guided grading. |

### 3.5 Prompt templates (verbatim, MT-Bench paper)

**Single-answer:** *"Please act as an impartial judge… rate the response on a scale of 1 to 10 by strictly following this format: `[[rating]]`, for example: 'Rating: [[5]]'."*

**Pairwise:** *"…Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible… output your final verdict by strictly following this format: `[[A]]` if assistant A is better, `[[B]]` if assistant B is better, and `[[C]]` for a tie."*

Always force a machine-parseable token (`[[A]]`, `Rating: [[n]]`) and require explanation *before* the verdict (CoT improves accuracy).

### 3.6 Calibration & harness guidance

Validate every judge against a human-labeled gold set; report not just accuracy but **Cohen's κ / Spearman** and per-class confusion (judges over-predict "pass"). Use temperature 0 for determinism, randomize positions, prefer pairwise+swap or reference-guided for high-stakes evals. Recalibrate when you change judge models.

---

## 4. Agentic, Trajectory & Tool-Use Evaluation

Evaluating agents means scoring **trajectories**, not just final strings, across three axes: **outcome correctness**, **trajectory quality**, and **efficiency/reliability**. Log full trajectories and emit metrics on all three.

### 4.1 Outcome-based: state checks beat string match

**WebArena** and **τ-bench** score *functional correctness* of the end state, tolerating the many valid paths to a goal. WebArena: per-task reward `r ∈ {0,1}` where `r_info` compares the emitted answer to a reference (exact / `must_include` / fuzzy-LLM match) and `r_prog` runs a programmatic checker over backend state. **τ-bench** compares the **database hash at end-of-conversation to an annotated goal state** plus required-output assertions — path-agnostic. **SWE-bench Verified** is the execution-based gold standard: apply the patch, run `FAIL_TO_PASS` tests (must flip red→green) and `PASS_TO_PASS` tests (must stay green); `pass@1` = fraction resolved on first attempt.

```python
def swe_bench_resolved(patch, fail2pass, pass2pass, repo):
    repo.apply(patch)
    return (all(repo.run(t).passed for t in fail2pass)     # bug fixed
            and all(repo.run(t).passed for t in pass2pass)) # no regression
```

Prefer execution/state checks over reference-trajectory match whenever a verifiable oracle exists — they don't penalize creative-but-correct paths.

### 4.2 Tool-call correctness: BFCL's AST evaluation

The **Berkeley Function-Calling Leaderboard (BFCL)** grades a generated call *structurally* via AST, not string equality. Parse output → AST, then check: (1) function name matches (allowing dot↔underscore); (2) all required params present; (3) per-arg type+value match with language rules (Python permits int→float; strings case-/whitespace-normalized; lists order-sensitive, dicts order-independent). It is **all-or-nothing** per call. BFCL adds *executable* categories (run the call, compare within a 20% real-time tolerance), **relevance/irrelevance detection** (the agent must *refuse* to call when no tool fits — catches hallucinated calls), and v3+ multi-turn/multi-step state checks; v4 extends to holistic agentic tasks.

```python
def ast_match(pred, gold):
    if pred.fn != gold.fn: return False
    for p in gold.required:
        if p not in pred.args: return False
        if not value_ok(pred.args[p], gold.allowed[p]): return False  # type+value
    return True  # all-or-nothing
```

### 4.3 Reliability: pass^k vs. pass@k

`pass@k` ("≥1 of k succeeds") rewards lucky outliers and hides flakiness. **τ-bench's pass^k** ("*all* k i.i.d. trials succeed") measures consistency: `pass^k ≈ p^k` for per-trial success `p`, decaying exponentially — a 90% agent drops to ~57% at k=8. SOTA agents score **pass^8 < 25% in retail**, exposing brittleness headline numbers mask. Report both `pass@1` and `pass^k`; for production agents, `pass^k` is the honest SLA proxy. **τ²-bench** adds a dual-control setting where the *simulated user* also has tools.

### 4.4 Trajectory grading: rubric LLM-judge & reference-match

When no execution oracle exists (open-ended browsing; **GAIA**'s 466 questions use quasi-exact-match but many real tasks don't), grade the trajectory directly:

1. **Reference-trajectory match** (LangSmith AgentEvals): hard-code an expected tool sequence; score with `strict` (exact order), `unordered` (set equality), or `subset/superset` overlap. Cheap and deterministic, but brittle to valid alternate paths.
2. **LLM-as-judge / Agent-as-a-Judge over trajectories:** give a strong judge the task, full trajectory, rubric, and optional reference; score per-step dimensions (tool selection, argument validity, decision order, efficiency, safety). Rubric/process-reward variants assign per-step credit, lifting the *Faithful Reasoning Rate* (correct answer AND all-correct steps) over outcome-only judging.

```python
JUDGE_RUBRIC = """Score 1-5 per dimension, cite step indices.
- tool_selection: right tool for each subgoal?
- arg_correctness: valid, grounded arguments?
- efficiency: redundant/looping steps? (penalize)
- goal_completion: final state satisfies intent?
Return JSON {dim: {score, evidence_steps}}."""
verdict = judge(task=t, trajectory=traj, rubric=JUDGE_RUBRIC, reference=ref)
```

### 4.5 Efficiency & node-level metrics (three-tier model)

Layer the harness (per Maxim/LangSmith): **system** (latency, tokens, $ cost, #tool-calls) → **session** (task success, `pass^k`, trajectory quality) → **node** (tool-selection precision, argument-exact-match, step utility). Track **step efficiency** = useful-steps / total-steps; flag loops and redundant calls.

### 4.6 Agent failure modes to guard against

- **Reward hacking / weak oracles:** SWE-bench solution-leakage + thin tests can inflate scores ~6 points — audit `FAIL_TO_PASS` strength.
- **Judge gaming:** unfaithful CoT can fool trajectory judges ("Gaming the Judge"); require step-grounded evidence and spot-check vs. humans.
- **Reference-match brittleness:** penalizes valid alternate paths — prefer state/exec checks.
- **pass@k optimism:** masks variance; pair with `pass^k`.
- **User-simulator drift:** an LLM "user" that mis-specifies inflates failure — pin simulator model/temperature and version it.
- **Position/verbosity bias** in LLM judges: randomize order, normalize length, calibrate against a human-labeled golden set.

---

## 5. Eval Rigor — Hallucination, Bias, Robustness, Significance

A production harness must measure *factuality*, *robustness*, and *whether a measured difference is real*.

### 5.1 Hallucination detection

**SelfCheckGPT (zero-resource, black-box).** If the model knows a fact, stochastically sampled responses agree; hallucinations diverge. Draw a greedy response `R` and `N` stochastic samples `{S_1..S_N}` (temp ~1.0). For each sentence `r_i`:
- **NLI variant:** `score(r_i) = (1/N) Σ_n P(contradict | r_i, S_n)` from DeBERTa-MNLI. High → likely hallucinated.
- **Prompt variant:** ask an LLM "Does `S_n` support `r_i`? Yes/No", Yes→0, No→1, average. Best AUC-PR but `N×` calls.

```python
def selfcheck_nli(sentences, samples, nli):  # nli -> P(contradict)
    return [sum(nli(s, smp) for smp in samples)/len(samples) for s in sentences]
```

Use with no ground truth and only API access. Avoid when answers are legitimately diverse (creative writing) — divergence ≠ hallucination. Cost `O(N·sentences)`.

**Semantic Entropy (Farquhar et al., *Nature* 2024).** Targets *confabulations*. Sample `K≈5–10` answers; cluster into meaning-equivalence classes via **bidirectional NLI** (`a` entails `b` AND `b` entails `a` → same cluster). Sum length-normalized sequence probabilities per cluster to get `p(c)`, then `SE = −Σ_c p(c) log p(c)`. High SE = uncertain meaning. Outperforms token-entropy by ignoring paraphrase variation. Cheaper proxy: **Semantic Entropy Probes** (linear probe on hidden states, single forward pass).

**FActScore (Min et al., EMNLP 2023).** For long-form factuality: (1) decompose into **atomic facts**, (2) retrieve evidence per fact, (3) label Supported/Not via NLI/LLM, (4) `FActScore = (#supported)/(#atomic facts)` — a **precision** metric (omitting facts isn't penalized, so pair with a coverage metric). Agrees with humans at <2% error.

**NLI-based faithfulness (grounded/RAG).** Given context `C` and a claim, score `P(entail | C, claim)`; flag below threshold. Strong (AUROC ~0.88) and cheap. Decompose multi-hop claims first.

### 5.2 Adversarial & robustness eval

Robustness = stability under semantics-preserving perturbations. **PromptBench/PromptRobust** attacks the *prompt* across 4 levels: character (typos, DeepWordBug), word (synonyms, TextFooler), sentence (distractors), semantic (paraphrase). Metric: **Performance Drop Rate** `PDR = 1 − Acc(adv)/Acc(clean)`. **AdvGLUE** perturbs the *input sample*. Report clean vs. worst-case-perturbation accuracy and attack success rate. Always include a **distractor/irrelevant-context** test and prompt-injection probes. *Failure mode:* perturbations that change the gold label inflate "fragility" — human spot-check that perturbations are truly label-preserving.

### 5.3 Data contamination

Inflated scores from test data leaking into pretraining. Detection: (1) **n-gram overlap** (GPT-3 used a 13-gram flag); (2) **canary strings / GUIDs** embedded in benchmark files; (3) **perturbation/membership tests** — compare original vs. reworded/reordered (e.g., shuffled MC options); a large drop signals memorization; (4) **PaCoST** paired-confidence test on original vs. paraphrase. Mitigations: private held-out sets, canary-protected splits, dynamically generated/time-stamped (post-cutoff) benchmarks. Caveat: a clean n-gram check does not prove no contamination (paraphrased leakage evades it).

### 5.4 Statistical significance

Never report a single number. For accuracy on `n` IID items: `SE = √(μ(1−μ)/n)`; 95% CI `= μ ± 1.96·SE`. The CLT approximation is unreliable for `n < 100` — use bootstrap or Beta-Binomial there.

**Bootstrap CI:** resample per-item scores with replacement `B≥1000×`, take 2.5/97.5 percentiles of resampled means. Use **BCa** for skewed/small samples.

**Paired comparison (do this for A vs. B).** Both models on the *same* items reduces variance via positive correlation. Per-item delta `d_i = s_{A,i} − s_{B,i}`; bootstrap the mean of `d_i`; A beats B at 95% if <2.5% of resampled mean-deltas fall below 0.

```python
def paired_bootstrap(dA, dB, B=10000):
    d = np.array(dA) - np.array(dB); n = len(d)
    boot = [np.mean(d[np.random.randint(0,n,n)]) for _ in range(B)]
    return np.mean(d), np.mean(np.array(boot) <= 0)  # one-sided p
```

For binary correct/incorrect pairs use **McNemar's test** on discordant pairs (`b`=A-right/B-wrong, `c`=A-wrong/B-right): `χ² = (|b−c|−1)²/(b+c)`. For multi-seed training comparisons, use a **sign-flip permutation test** plus a BCa CI; declare significance only if the CI excludes 0 **and** `p<0.05` (a guardrail against over-claiming +1% gains). *Failure mode:* items are *not* independent (templated/grouped prompts) → use **clustered** standard errors or cluster-bootstrap.

### 5.5 Human alignment & eval-set design

Validate any automatic metric (BLEU/ROUGE/LLM-judge) against human labels via **Pearson/Spearman/Krippendorff's α** — report the correlation, don't assume it. LLM-as-judge reaches ~80%+ human agreement but carries position, verbosity, and self-enhancement bias. Mitigate by averaging over both orderings, concise rubrics with explicit bias disclaimers, and calibrating to a human anchor set. Eval-set design: stratify by difficulty/topic, ensure `n` per slice for powered CIs, keep a private contamination-resistant split, version every dataset, and **report per-slice results** (aggregate scores hide regressions).

---

## 6. Eval Frameworks & Tooling

The 2024–2026 landscape has three layers: **metric libraries** (RAGAS, DeepEval, autoevals/Phoenix) that score outputs; **tracing+experiment platforms** (LangSmith, Braintrust, Arize Phoenix, TruLens) that attach scores to spans and version datasets; and **harnesses** (OpenAI Evals, promptfoo, Inspect) that orchestrate dataset→generate→score loops. A Staff-level harness composes all three: a tracer for observability, a metric lib for scoring, a harness for CI gating. **ARES is a self-contained harness in this spirit** — it owns generation (`agent.py`), scoring (`metrics.py` + `llm_judge.py` + `trajectory.py` + `robustness.py`), aggregation (`evaluate.py`), and a search/report loop (`optimize.py`/`report.py`).

| Tool | Layer | One-liner | Use it when | Watch out |
|---|---|---|---|---|
| **RAGAS** | metric lib | Reference-free RAG metrics (faithfulness, answer relevancy, ctx prec/recall). | RAG pipelines without labels. | Claim decomposition is judge-model sensitive; faithfulness ≠ correctness; pin the judge. |
| **DeepEval** | metric lib | `pytest`-native, 14+ metrics; flagship **G-Eval** (auto-CoT rubric → prob-weighted score). | Unit-test-style gating in CI/CD. | Scores cluster & are non-deterministic; set thresholds empirically, not at 0.5. Pushes toward Confident AI platform. |
| **TruLens** | tracer+eval | Feedback functions + **RAG Triad** (ctx relevance + groundedness + answer relevance). | RAG observability + eval together. | Selector misconfig silently scores the wrong span. |
| **OpenAI Evals** | harness | YAML/JSONL registry; `Match`/`Includes`/`ModelGradedSpec` templates, no code. | Benchmark-style accuracy on fixed datasets. | Exact-match penalizes correct-but-rephrased; now mostly a benchmark registry. |
| **LangSmith** | tracer+eval | Hosted datasets + experiments + pre-built judges (Correctness, Hallucination). | Already on LangChain/LangGraph; best trace↔eval linkage. | Judge alignment (~85%) varies by task — calibrate; vendor lock. |
| **Arize Phoenix** | tracer+eval | OSS, OTel/OpenInference; LLM-judge evaluators (relevance, hallucination, toxicity). | Want OSS, OTel-native, no lock. | Binary "rails" classifiers hide borderline cases; add explanations. |
| **Braintrust** | tracer+eval | `autoevals` model-graded scorers (Factuality, ClosedQA) + experiment diffing. | Fast experiment tracking + scorer reuse. | Factuality's A–E rubric maps to coarse scores. |
| **promptfoo** | harness | Declarative YAML; deterministic asserts + `llm-rubric`/`g-eval` + strong **red-team** suite. | Prompt regression + security scanning in CI; lowest setup cost. | `llm-rubric` needs source inlined via `{{var}}`; pin grading provider. |
| **Inspect (UK AISI)** | harness | Dataset→Task→Solver→Scorer, Docker sandboxing, 200+ benchmarks; used by Anthropic/DeepMind. | Serious capability/safety evals, sandboxed agentic tool use. | Sandbox/tool setup non-trivial; scorer choice shifts results. |

```python
# DeepEval G-Eval — CI-gateable correctness check
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
m = GEval(name="Correctness",
          criteria="Is actual_output factually correct vs expected_output?",
          evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                             LLMTestCaseParams.EXPECTED_OUTPUT])
m.measure(LLMTestCase(input=q, actual_output=a, expected_output=gold))
print(m.score, m.reason)
```

```python
# Inspect — sandboxed, reproducible task
from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_qa
@task
def my_eval():
    return Task(dataset=ds, solver=generate(), scorer=model_graded_qa())
# inspect eval my_eval.py --model openai/gpt-4o
```

---

## 7. Comparison Table of Methods

| Method | What it measures | Reference-free? | Cost | Primary failure mode |
|---|---|---|---|---|
| **EM / F1** (SQuAD) | Answer correctness vs. gold (lexical) | No (needs gold) | Free | Penalizes correct-but-rephrased; misses semantics |
| **Retrieval Recall / Hit@K** | Did we fetch gold supporting docs | No (needs labeled docs) | Free | Recall can be high while F1 is low (extraction is the bottleneck) |
| **MRR** | Rank of first relevant doc | No | Free | Ignores all hits after the first |
| **nDCG@K** | Graded, rank-aware retrieval quality | No | Free | Needs graded relevance labels |
| **Lexical groundedness** | Fraction of answer tokens in context | Yes | Free | Token overlap ≠ semantic entailment; gameable by copying |
| **RAGAS Faithfulness** | Claims entailed by context | Yes | LLM ($/claim) | Decomposition variance; ≠ correctness; judge-model sensitive |
| **RAGAS Answer Relevancy** | On-topic-ness vs. question | Yes | LLM + embeddings | Does **not** measure correctness |
| **RAGAS Context Recall** | Gold-answer claims in context | No (needs gold) | LLM | Requires ground-truth answer |
| **ARES (PPI)** | Ctx relevance / faithfulness / answer relevance, **with CI** | Hybrid (few labels) | Train judge + ~100s labels | Needs annotations; not real-time per-example |
| **G-Eval** | Rubric quality (graded, CoT) | Yes | LLM (n-sample/logprobs) | Score drift across judge versions; logprobs often hidden |
| **LLM-as-judge (pairwise)** | Relative answer quality | Yes (ref-guided opt.) | LLM (O(n²)) | Position / verbosity / self-enhancement bias |
| **MT-Bench / Arena Elo** | Holistic chat ranking | Yes (human or judge) | High (battles) | Crowd noise; needs bootstrapped CIs |
| **SelfCheckGPT** | Consistency-based hallucination | Yes (zero-resource) | High (N samples) | Divergence ≠ hallucination for diverse-valid answers |
| **Semantic Entropy** | Meaning-level uncertainty | Yes | Medium (K samples) | Breaks down on long multi-claim outputs |
| **FActScore** | Long-form factual **precision** | No (needs evidence corpus) | LLM + retrieval | Ignores recall; decomposition can drop context |
| **BFCL AST match** | Tool-call structural correctness | No (needs gold call) | Free (parse) | All-or-nothing; brittle to valid arg variants |
| **SWE-bench / state checks** | Outcome correctness (execution) | No (needs oracle/tests) | Compute (run tests) | Weak tests/leakage inflate scores |
| **pass^k** | Trajectory reliability/consistency | No | k× rollouts | Expensive; exposes (doesn't fix) brittleness |
| **Trajectory judge / ref-match** | Step-level tool/order quality | Yes (judge) / No (ref) | LLM / free | Ref-match penalizes valid alternate paths; judge gaming |
| **PromptRobust PDR** | Robustness under perturbation | Yes | k× variants | Label-changing perturbations inflate fragility |
| **Bootstrap / McNemar CI** | Is the A–B difference real | n/a (meta) | Cheap | Non-IID items → CIs too narrow (use cluster-bootstrap) |

---

## 8. What ARES Implements (and where it maps)

ARES (`/Users/fullfocus/ares/ares/`) is a working harness that searches an agentic-RAG **design space** (`retrieval_k`, `hybrid_alpha`, `decompose`, `reflect`, `prompt_variant`) and measures each config on quality **and** cost. Below, every metric ARES ships is mapped to the taxonomy in §1, with the exact function name and a one-line how-to. The generation backend is `ares.llm.generate(prompt, system=…, max_new_tokens=…, temperature=0.0)` (local Qwen2.5-1.5B on MPS by default; one env var switches it to Anthropic/OpenAI), with a content-addressed cache so the config sweep never recomputes identical prompts.

### 8.1 Lexical / IR metrics — `ares/metrics.py`

| Taxonomy box | Function | How-to |
|---|---|---|
| Answer quality (ref-based) | `em(pred, gold)` | SQuAD-style exact match after `normalize()` (lowercase, strip punctuation/articles). |
| Answer quality (ref-based) | `f1(pred, gold)` | Token-overlap F1 of normalized strings; the primary correctness signal in the sweep. |
| Retrieval (ref-based) | `retrieval_recall(retrieved_titles, gold_titles)` | Fraction of gold supporting titles surfaced — graded against HotpotQA gold passages. |
| Generation faithfulness (ref-free, lexical) | `grounding(answer, context)` | Fraction of answer content-tokens present in retrieved context; low grounding + wrong answer = hallucination signal. |
| Robustness (cheap) | `adversarial_variants(question)` | Two no-LLM perturbations (typo + polite-prefix distraction); superseded by the richer `robustness.py`. |

These are wired into `ares.evaluate.evaluate(examples, cfg)`, which runs a `Config` over a set of examples, scores `em/f1/recall/grounding/latency/tokens` per example, and aggregates to means with `n` and `errors` (per-example failures score zero instead of crashing the run).

### 8.2 LLM-as-judge — `ares/llm_judge.py` (NEW)

Reference-free semantic judging that complements the lexical metrics. All four functions are **pure** (build prompt → `llm.generate` → parse), use `temperature=0.0` for determinism + cache reuse, truncate context to ~4000 chars, and parse scores via layered regex with neutral fallbacks (parsing never raises). The shared `JUDGE_SYSTEM` prompt and `DEFAULT_G_EVAL_RUBRIC` pre-empt length/style/position bias.

| Taxonomy box | Function | How-to |
|---|---|---|
| Generation faithfulness (ref-free) | `faithfulness(answer, context)` | → `{score 0-1, verdict ∈ {supported, partial, unsupported, contradicted}, rationale}`. Score is **derived from the verdict label**, so they can never disagree — mirrors RAGAS faithfulness (§2.1) but at the verdict level. |
| Answer quality (ref-free) | `answer_relevancy(question, answer)` | → `{score 0-1, rationale}`. On-topic-ness decoupled from correctness/grounding — catches "insufficient context" non-answers and rambling. Mirrors RAGAS answer relevancy. |
| Answer quality (rubric, ref-light) | `g_eval(question, answer, context, rubric=None)` | → `{score 1-5, reasoning}`. G-Eval style: CoT reasoning emitted **before** the integer (§3.1). Defaults to a correctness+groundedness rubric. |
| Answer quality (pairwise) | `pairwise(question, answer_a, answer_b, context=None)` | → `{winner ∈ {A, B, tie}, rationale}`. Runs **both** orderings `(A,B)` and `(B,A)`, maps verdicts back to the real candidate, and only declares a winner when both agree — the position-bias defense from §3.4. |

```python
from ares import llm_judge
llm_judge.faithfulness(answer="Einstein was born in Germany.", context=ctx)
llm_judge.g_eval(question=q, answer=a, context=ctx)        # rubric defaults to correctness+groundedness
llm_judge.pairwise(question=q, answer_a=a1, answer_b=a2, context=ctx)  # position-bias-robust
```

### 8.3 Trajectory & tool-use — `ares/trajectory.py` (NEW)

Pure, side-effect-free analysis of the `steps` list returned by `ares.agent.answer` (no model execution). The agent emits three canonical step kinds: `decompose` (`{"subs": [...]}`), `generate` (`{"answer": ...}`), `reflect_retry` (`{"answer": ...}`). Maps to the **agent trajectory** box in §1.

| Taxonomy box | Function | How-to |
|---|---|---|
| Trajectory — tool-use summary | `tool_use_summary(steps)` | → `{decomposed, n_subquestions, reflected, retried, n_steps}`. Which agent capabilities fired. |
| Trajectory — efficiency (node tier) | `step_efficiency(steps)` | → `1/n` (fewer steps scores higher; 0.0 for empty) — the step-efficiency idea from §4.5. |
| Trajectory — reference-match | `trajectory_score(steps, expected_tools=None)` | → `{score 0-1, detail}`. LCS overlap of realized vs. expected step sequence (graceful, order-aware — softer than BFCL's all-or-nothing §4.2); falls back to `step_efficiency` when no reference. |
| Trajectory — reflection payoff | `reflection_effectiveness(before_answer, after_answer, gold)` | → `{f1_before, f1_after, delta, improved, regressed}` via `metrics.f1` — does a reflect-retry actually help. |

```python
from ares import trajectory
out = agent.answer(example, cfg)
trajectory.tool_use_summary(out["steps"])
trajectory.trajectory_score(out["steps"], expected_tools=["decompose", "generate"])
```

### 8.4 Adversarial robustness — `ares/robustness.py` (NEW)

Maps to the **robustness** box in §1; the surfacing of the "high F1 + low stability = brittle benchmark champion" signature (§5.2). Perturbations are pure string transforms (no LLM); the model call is **injected** via `answer_fn` (same contract as `ares.agent.answer`) so the caller controls when the GPU runs.

| Taxonomy box | Function | How-to |
|---|---|---|
| Robustness — perturbation generation | `perturb(question)` | → `list[{kind, text}]`. Kinds: `typo` (one adjacent-char swap, meaning-preserving), `paraphrase_template` (meaning-preserving rewrap), `distractor_prefix` (irrelevant lead-in), `negation_probe` (meaning-*probing* negation). 3–4 variants, deterministic. |
| Robustness — stability eval | `robustness_eval(example, cfg, answer_fn)` | → `{base_f1, perturbed_f1_mean, stability, drops[]}` where `stability = clamp(perturbed_f1_mean / base_f1, 0..1)` (0.0 if `base_f1==0`); each `drops` entry is `{kind, question, answer, f1, drop}`. Per-variant errors score F1 0.0 — one bad variant can't crash the loop. This is a PDR-style robustness check (§5.2). |
| Robustness — kind enumeration | `PERTURBATION_KINDS` | Tuple of the four kind names for callers that filter/pre-allocate. |

```python
from ares import robustness, agent
robustness.robustness_eval(example, cfg, answer_fn=agent.answer)  # answer_fn is the only GPU touch
```

### 8.5 Coverage map — taxonomy → ARES

```
TAXONOMY BOX                         ARES IMPLEMENTATION
─────────────────────────────────────────────────────────────────────────────────
Retrieval quality          ──►  metrics.retrieval_recall          (ref-based)
Generation faithfulness    ──►  metrics.grounding (lexical)       (ref-free)
                                llm_judge.faithfulness            (ref-free, semantic)
Answer quality             ──►  metrics.em / metrics.f1           (ref-based)
                                llm_judge.answer_relevancy        (ref-free)
                                llm_judge.g_eval                  (rubric, ref-light)
                                llm_judge.pairwise                (pairwise, position-robust)
Agent trajectory           ──►  trajectory.tool_use_summary
                                trajectory.step_efficiency
                                trajectory.trajectory_score       (ref-match, LCS)
                                trajectory.reflection_effectiveness
Robustness                 ──►  robustness.perturb / robustness_eval   (PDR-style)
Efficiency / cost          ──►  latency + tokens per config       (Pareto frontier)
Statistical significance   ──►  (gap) n=50/config; add bootstrap CIs + paired tests
Human alignment            ──►  (gap) calibrate llm_judge vs a human-labeled anchor set
─────────────────────────────────────────────────────────────────────────────────
```

The two **gaps** are the honest next steps for Staff-level rigor: ARES currently reports point estimates at `n≈50`/config with no CI, and the LLM judges are not yet calibrated against a human-labeled anchor set (see §5.4–5.5). Wiring in `paired_bootstrap`/McNemar and a small human gold set would close them.

### 8.6 What the live sweep found (partial run, `n=50`/config)

A self-improvement loop (ablations → synthesize best factors → neighbour-search) ran 25 configs at `n=50` each, 0 errors. **Treat as directional** — small `n`, incomplete run, no CIs yet.

- **Best so far:** `#14 combo_best` (synthesis phase) — **F1 0.3775**, EM 0.30, recall 0.79, grounding 0.834, latency 0.122s. Params: `retrieval_k=4, hybrid_alpha=0.5, decompose=False, reflect=True, prompt_variant=cot`. Essentially tied with `#9 prompt_cot` (F1 0.3742). Baseline `#1` (plain, k4, alpha0.5) = F1 0.293.
- **The win is prompt engineering, not retrieval tuning.** All top-5 F1 configs use `cot`; switching plain→cot is worth ~+0.08 F1, while sweeping `k`/`alpha` moves F1 by <0.02.
- **`strict` prompt is actively harmful** — worst F1 (0.17–0.23) and tanks grounding (~0.68–0.78); drop it.
- **`reflect`** = small positive, cheap. **`decompose`** = costliest lever (32–61 tokens vs ~12 baseline), least payoff; only helps combined with cot/reflect.
- **`hybrid_alpha`** mid-values win (0.3–0.5); extremes hurt (alpha0.0→0.246, alpha1.0→0.216).
- **Recall and F1 are decoupled:** high-recall configs (k8→recall 0.91) don't top the F1 board → the bottleneck is **answer extraction/generation, not retrieval coverage.** This is exactly the §5.5 lesson that aggregate scores hide where the real failure is.

> The latency column is noisy (`combo_best` reported 0.12s while a near-identical config reported 0.93s) — a reminder to report cost with variance, not a single timing.

---

## 9. How to Run It

```bash
cd /Users/fullfocus/ares
pip install -r requirements.txt

# tiny end-to-end check (first run downloads Qwen2.5-1.5B + MiniLM embedder)
python3 scripts/smoke_test.py

# the self-improvement loop: ablations → synthesize → neighbour-search, then FINDINGS.md
python3 scripts/run_loop.py --minutes 95 --eval-n 40

# show the LangGraph agent's node-by-node trace on one question
python3 scripts/demo_langgraph.py
```

Results land in `runs/`: `leaderboard.csv`, `results.jsonl`, `progress.log`, and a generated `runs/FINDINGS.md` with Pareto / top-config plots.

**Point it at a hosted judge/model** (e.g., to use Claude as the LLM-as-judge backend):

```bash
export ARES_BACKEND=anthropic ANTHROPIC_API_KEY=...   # or ARES_BACKEND=openai OPENAI_API_KEY=...
# optional: export ARES_HOSTED_MODEL=claude-sonnet-4-6
```

**Use the new eval modules directly** (the GPU is touched only by the injected `answer_fn` / `llm.generate`):

```python
from ares import agent, llm_judge, trajectory, robustness
from ares.agent import Config

cfg = Config(name="cot_reflect", prompt_variant="cot", reflect=True, retrieval_k=4)
out = agent.answer(example, cfg)

# semantic judging (reference-free)
llm_judge.faithfulness(out["answer"], out["context"])
llm_judge.answer_relevancy(example["question"], out["answer"])
llm_judge.g_eval(example["question"], out["answer"], out["context"])

# trajectory / tool-use
trajectory.tool_use_summary(out["steps"])
trajectory.trajectory_score(out["steps"], expected_tools=["decompose", "generate"])

# adversarial robustness
robustness.robustness_eval(example, cfg, answer_fn=agent.answer)
```

> **macOS/MPS notes:** the scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` and keep the HF cache inside the project. No CUDA, no DeepSpeed — the stack is fp16-on-MPS by default and falls back to CPU automatically.

---

## 10. References (deduplicated)

**RAG metrics — RAGAS / ARES / retrieval / claim-level**
- RAGAS Faithfulness — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- RAGAS Context Precision — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
- RAGAS Answer Relevance — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
- RAGAS (paper) — https://arxiv.org/abs/2311.09476
- ARES (NAACL 2024) — https://aclanthology.org/2024.naacl-long.20/
- FaithBench / FaithJudge (2025) — https://arxiv.org/pdf/2505.04847
- nDCG@K in RAG pipelines — https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-part-3-dcgk-and-ndcgk/

**LLM-as-a-judge — G-Eval / MT-Bench / Arena**
- G-Eval (arXiv) — https://arxiv.org/abs/2303.16634
- G-Eval (EMNLP 2023) — https://aclanthology.org/2023.emnlp-main.153/  ·  PDF — https://aclanthology.org/2023.emnlp-main.153.pdf
- MT-Bench / Chatbot Arena (Zheng et al.) — https://arxiv.org/abs/2306.05685  ·  v4 HTML — https://arxiv.org/html/2306.05685v4
- LMSYS Bradley-Terry leaderboard — https://www.lmsys.org/blog/2023-12-07-leaderboard/
- LLM-judge survey — https://arxiv.org/pdf/2408.09235

**Agentic / trajectory / tool-use**
- τ-bench — https://arxiv.org/abs/2406.12045
- Berkeley Function-Calling Leaderboard (BFCL) — https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
- WebArena — https://arxiv.org/abs/2307.13854
- SWE-bench Verified — https://openai.com/index/introducing-swe-bench-verified/
- GAIA — https://arxiv.org/abs/2311.12983
- LangSmith trajectory evals — https://docs.langchain.com/langsmith/trajectory-evals
- AgentBench — https://arxiv.org/abs/2308.03688
- Agent-as-a-Judge / step-wise rubric rewards — https://arxiv.org/pdf/2506.07982

**Eval frameworks & tooling**
- DeepEval (G-Eval metric) — https://deepeval.com/docs/metrics-llm-evals
- TruLens RAG Triad — https://www.trulens.org/getting_started/core_concepts/rag_triad/
- OpenAI Evals (build-eval) — https://github.com/openai/evals/blob/main/docs/build-eval.md
- promptfoo configuration — https://www.promptfoo.dev/docs/configuration/guide/
- Braintrust autoevals — https://github.com/braintrustdata/autoevals
- Inspect (UK AISI) — https://inspect.aisi.org.uk/

**Eval rigor — hallucination / robustness / statistics**
- SelfCheckGPT (EMNLP 2023) — https://aclanthology.org/2023.emnlp-main.557.pdf
- Semantic Entropy (Nature 2024) — https://www.nature.com/articles/s41586-024-07421-0
- FActScore (EMNLP 2023) — https://aclanthology.org/2023.emnlp-main.741/
- PromptRobust / PromptBench — https://arxiv.org/abs/2306.04528
- Statistics for LLM evals — https://cameronrwolfe.substack.com/p/stats-llm-evals
- Contamination / membership testing — https://arxiv.org/html/2511.19794v1
- Claim-level entailment (CLATTER) — https://arxiv.org/html/2502.14425v2
```
