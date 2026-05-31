"""LLM backend. Local Qwen2.5 on MPS by default; pluggable to Anthropic/OpenAI
if a key is set. Content-addressed cache so the config search doesn't recompute
identical prompts (decomposition/reflection reuse)."""
from __future__ import annotations
import os, time, hashlib, json, threading

DEFAULT_MODEL = os.environ.get("ARES_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

_MODEL = None
_TOK = None
_LOCK = threading.Lock()
_CACHE: dict[str, dict] = {}
_STATS = {"calls": 0, "cache_hits": 0, "gen_tokens": 0, "gen_seconds": 0.0}


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load(model_name: str):
    global _MODEL, _TOK
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = device()
        _TOK = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if dev != "cpu" else torch.float32
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        m.to(dev)
        m.eval()
        _MODEL = m


def _key(parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def generate(prompt: str, system: str | None = None, max_new_tokens: int = 256,
             temperature: float = 0.0, model_name: str = DEFAULT_MODEL) -> dict:
    """Return {text, latency, tokens, cached}. Latency is wall-clock for fresh
    generations and ~0 for cache hits (so cost reflects marginal compute)."""
    k = _key([model_name, system, prompt, max_new_tokens, round(temperature, 3)])
    _STATS["calls"] += 1
    if k in _CACHE:
        _STATS["cache_hits"] += 1
        c = _CACHE[k]
        return {"text": c["text"], "latency": 0.0, "tokens": c["tokens"], "cached": True}

    # Optional hosted backends (used only if a key is present).
    if os.environ.get("ARES_BACKEND") == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        out = _anthropic(prompt, system, max_new_tokens, temperature)
    elif os.environ.get("ARES_BACKEND") == "openai" and os.environ.get("OPENAI_API_KEY"):
        out = _openai(prompt, system, max_new_tokens, temperature)
    else:
        out = _local(prompt, system, max_new_tokens, temperature, model_name)

    _CACHE[k] = {"text": out["text"], "tokens": out["tokens"]}
    _STATS["gen_tokens"] += out["tokens"]
    _STATS["gen_seconds"] += out["latency"]
    out["cached"] = False
    return out


def _local(prompt, system, max_new_tokens, temperature, model_name) -> dict:
    _load(model_name)
    import torch
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(text, return_tensors="pt").to(_MODEL.device)
    t0 = time.time()
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            pad_token_id=_TOK.eos_token_id,
        )
    dt = time.time() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    txt = _TOK.decode(gen, skip_special_tokens=True).strip()
    return {"text": txt, "latency": dt, "tokens": int(gen.shape[0])}


def _anthropic(prompt, system, max_new_tokens, temperature) -> dict:
    import anthropic
    cli = anthropic.Anthropic()
    t0 = time.time()
    r = cli.messages.create(
        model=os.environ.get("ARES_HOSTED_MODEL", "claude-sonnet-4-6"),
        max_tokens=max_new_tokens, temperature=temperature,
        system=system or "", messages=[{"role": "user", "content": prompt}],
    )
    dt = time.time() - t0
    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    return {"text": txt, "latency": dt, "tokens": r.usage.output_tokens}


def _openai(prompt, system, max_new_tokens, temperature) -> dict:
    from openai import OpenAI
    cli = OpenAI()
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    t0 = time.time()
    r = cli.chat.completions.create(
        model=os.environ.get("ARES_HOSTED_MODEL", "gpt-4o-mini"),
        max_tokens=max_new_tokens, temperature=temperature, messages=msgs,
    )
    dt = time.time() - t0
    return {"text": r.choices[0].message.content.strip(), "latency": dt,
            "tokens": r.usage.completion_tokens}


def stats() -> dict:
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0 if isinstance(_STATS[k], int) else 0.0
