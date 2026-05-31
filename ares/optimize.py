"""The self-improvement loop. Phase 1 runs structured ablations; Phase 2 keeps
reiterating — synthesizing combined configs from the best factors, then searching
neighbours of the running best — until the time budget is spent. Everything is
checkpointed after every config so a partial run is still a usable result."""
from __future__ import annotations
import os, json, time, csv, random
from . import evaluate as ev, llm, corpus
from .agent import Config

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
LB_COLS = ["i", "name", "retrieval_k", "hybrid_alpha", "decompose", "reflect",
           "prompt_variant", "f1", "em", "recall", "grounding", "latency",
           "tokens", "n", "errors", "wall_s", "phase"]


def _base():
    return dict(retrieval_k=4, hybrid_alpha=0.5, decompose=False, reflect=False, prompt_variant="plain")


def ablation_grid():
    b = _base()
    cfgs = [Config(name="baseline", **b)]
    for k in (2, 6, 8):
        cfgs.append(Config(name=f"k{k}", **{**b, "retrieval_k": k}))
    for a in (0.0, 0.3, 0.7, 1.0):
        cfgs.append(Config(name=f"alpha{a}", **{**b, "hybrid_alpha": a}))
    for pv in ("cot", "strict"):
        cfgs.append(Config(name=f"prompt_{pv}", **{**b, "prompt_variant": pv}))
    cfgs.append(Config(name="decompose", **{**b, "decompose": True}))
    cfgs.append(Config(name="reflect", **{**b, "reflect": True}))
    cfgs.append(Config(name="decompose+reflect", **{**b, "decompose": True, "reflect": True}))
    return cfgs


def _best(board):
    return max(board, key=lambda r: (r["f1"], r["em"], -r["latency"])) if board else None


def _synthesize(board, tried):
    """Combine the winning value of each factor into new candidate configs."""
    if not board:
        return []
    def best_val(col, default):
        ranked = sorted(board, key=lambda r: (r["f1"], r["em"]), reverse=True)
        return ranked[0].get(col, default)
    b = _base()
    bk = int(best_val("retrieval_k", 4))
    ba = float(best_val("hybrid_alpha", 0.5))
    bp = best_val("prompt_variant", "plain")
    base_f1 = next((r["f1"] for r in board if r["name"] == "baseline"), 0.0)
    dec = next((r["f1"] for r in board if r["name"] == "decompose"), 0.0) > base_f1
    ref = next((r["f1"] for r in board if r["name"] == "reflect"), 0.0) > base_f1
    cands = [
        Config(name="combo_best", **{**b, "retrieval_k": bk, "hybrid_alpha": ba,
                                       "prompt_variant": bp, "decompose": dec, "reflect": ref}),
        Config(name="combo_kp", **{**b, "retrieval_k": bk, "hybrid_alpha": ba, "prompt_variant": bp}),
        Config(name="combo_dr", **{**b, "retrieval_k": bk, "hybrid_alpha": ba, "decompose": True, "reflect": True}),
    ]
    return [c for c in cands if c.key() not in tried]


def _neighbours(board, tried, rng):
    """Random neighbours of the running best — keeps the loop iterating to deadline."""
    best = _best(board)
    if not best:
        return []
    out = []
    for _ in range(4):
        k = max(2, min(8, int(best["retrieval_k"]) + rng.choice([-2, -1, 1, 2])))
        a = round(min(1.0, max(0.0, float(best["hybrid_alpha"]) + rng.choice([-0.2, -0.1, 0.1, 0.2]))), 2)
        pv = rng.choice(["plain", "cot", "strict"])
        dec = rng.random() < 0.4
        ref = rng.random() < 0.4
        c = Config(name=f"nbr_k{k}_a{a}_{pv}{'_d' if dec else ''}{'_r' if ref else ''}",
                   retrieval_k=k, hybrid_alpha=a, prompt_variant=pv, decompose=dec, reflect=ref)
        if c.key() not in tried:
            out.append(c)
    return out


def _write_board(path, board):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LB_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(board, key=lambda x: (x["f1"], x["em"]), reverse=True):
            w.writerow(r)


def run(minutes=110, eval_n=40, seed=0, out_dir=RUNS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    lb_path = os.path.join(out_dir, "leaderboard.csv")
    res_path = os.path.join(out_dir, "results.jsonl")
    log_path = os.path.join(out_dir, "progress.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    examples = corpus.load_examples(n=max(eval_n * 3, 120), seed=seed)
    random.Random(seed).shuffle(examples)
    eval_set = examples[:eval_n]
    log(f"eval_set={len(eval_set)} examples · budget={minutes} min · backend={os.environ.get('ARES_BACKEND','local')}")

    deadline = time.time() + minutes * 60
    board, tried = [], set()
    queue = ablation_grid()
    phase = "ablation"
    rng = random.Random(seed + 1)
    i = 0
    while time.time() < deadline:
        if not queue:
            if phase == "ablation":
                phase = "synthesis"
                queue = _synthesize(board, tried)
            if not queue:
                phase = "neighbour-search"
                queue = _neighbours(board, tried, rng)
            if not queue:
                queue = _neighbours(board, tried, rng) or [Config(name=f"baseline_rep{i}", **_base())]
        cfg = queue.pop(0)
        if cfg.key() in tried and not cfg.name.startswith("baseline_rep"):
            continue
        tried.add(cfg.key())
        i += 1
        t0 = time.time()
        agg, rows = ev.evaluate(eval_set, cfg)
        rec = {"i": i, "name": cfg.name, **cfg.as_dict(), **{k: round(v, 4) for k, v in agg.items()},
               "wall_s": round(time.time() - t0, 1), "phase": phase}
        board.append(rec)
        _write_board(lb_path, board)
        with open(res_path, "a") as f:
            f.write(json.dumps({"rec": rec, "samples": rows[:5], "cache": llm.stats()}) + "\n")
        best = _best(board)
        log(f"#{i} [{phase}] {cfg.name}: F1={agg['f1']:.3f} EM={agg['em']:.3f} "
            f"recall={agg['recall']:.3f} ground={agg['grounding']:.3f} lat={agg['latency']:.2f}s "
            f"| best={best['name']}({best['f1']:.3f}) | {int(deadline - time.time())}s left")
    log(f"DONE: {i} configs evaluated. best={_best(board)['name']} F1={_best(board)['f1']:.3f}")
    return board
