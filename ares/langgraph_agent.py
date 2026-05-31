"""LangGraph showcase: the same agentic-RAG pipeline expressed as an explicit
StateGraph (decompose -> retrieve -> generate -> reflect -> [retry|end]). The
optimizer uses the lighter ares.agent path; this is the demonstrable graph."""
from __future__ import annotations
from typing import TypedDict, List
from . import llm, rag, agent


class State(TypedDict, total=False):
    question: str
    paragraphs: list
    k: int
    alpha: float
    prompt_variant: str
    do_decompose: bool
    do_reflect: bool
    queries: List[str]
    retrieved: list
    context: str
    answer: str
    grounded: bool
    trace: list


def _decompose(s: State) -> State:
    tr = s.get("trace", [])
    if s.get("do_decompose"):
        subs, _, _ = agent.decompose(s["question"])
        tr = tr + [{"node": "decompose", "subs": subs}]
        return {"queries": subs, "trace": tr}
    return {"queries": [s["question"]], "trace": tr + [{"node": "decompose", "subs": "skipped"}]}


def _retrieve(s: State) -> State:
    seen = {}
    for q in s["queries"]:
        top, _ = rag.hybrid_retrieve(q, s["paragraphs"], k=s.get("k", 4), alpha=s.get("alpha", 0.5))
        for p in top:
            seen[p["title"]] = p
    retrieved = list(seen.values())
    ctx = rag.format_context(retrieved)
    return {"retrieved": retrieved, "context": ctx,
            "trace": s.get("trace", []) + [{"node": "retrieve", "titles": [p["title"] for p in retrieved]}]}


def _generate(s: State) -> State:
    g = llm.generate(agent.PROMPTS[s.get("prompt_variant", "plain")].format(ctx=s["context"], q=s["question"]),
                     max_new_tokens=160)
    ans = agent._extract(g["text"], s.get("prompt_variant", "plain"))
    return {"answer": ans, "trace": s.get("trace", []) + [{"node": "generate", "answer": ans}]}


def _reflect(s: State) -> State:
    j = llm.generate(f"Context:\n{s['context']}\n\nQuestion: {s['question']}\nProposed answer: {s['answer']}\n\n"
                     "Is the answer fully supported by the context? Reply YES or NO.", max_new_tokens=10)
    grounded = j["text"].strip().upper().startswith("YES")
    return {"grounded": grounded, "trace": s.get("trace", []) + [{"node": "reflect", "grounded": grounded}]}


def build(checkpointer=None):
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    g.add_node("decompose", _decompose)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("reflect", _reflect)
    g.set_entry_point("decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "generate")

    def route_after_generate(s: State):
        return "reflect" if s.get("do_reflect") else END
    g.add_conditional_edges("generate", route_after_generate, {"reflect": "reflect", END: END})

    def route_after_reflect(s: State):
        # one retry if not grounded
        if not s.get("grounded") and not s.get("_retried"):
            return "retrieve"
        return END
    g.add_conditional_edges("reflect", route_after_reflect, {"retrieve": "retrieve", END: END})
    return g.compile(checkpointer=checkpointer)


def run_one(example, k=4, alpha=0.5, prompt_variant="plain", do_decompose=True, do_reflect=True):
    app = build()
    state = {"question": example["question"], "paragraphs": example["paragraphs"],
             "k": k, "alpha": alpha, "prompt_variant": prompt_variant,
             "do_decompose": do_decompose, "do_reflect": do_reflect, "trace": []}
    return app.invoke(state)
