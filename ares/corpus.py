"""Dataset loader. Uses HotpotQA (distractor) — each example is self-contained:
~10 paragraphs (2 gold + distractors), a question, a short gold answer, and the
gold supporting titles. Falls back to a small embedded sample if offline."""
from __future__ import annotations
import random

FALLBACK = [
    {"id": "fb1", "question": "What nationality is the director of the film that won the 1994 Palme d'Or?",
     "answer": "American",
     "paragraphs": [
         {"title": "Pulp Fiction", "text": "Pulp Fiction is a 1994 film directed by Quentin Tarantino. It won the Palme d'Or at the 1994 Cannes Film Festival."},
         {"title": "Quentin Tarantino", "text": "Quentin Tarantino is an American film director and screenwriter born in Knoxville, Tennessee."},
         {"title": "Cannes Film Festival", "text": "The Cannes Film Festival is an annual film festival held in Cannes, France."},
         {"title": "Palme d'Or", "text": "The Palme d'Or is the highest prize awarded at the Cannes Film Festival."},
         {"title": "Reservoir Dogs", "text": "Reservoir Dogs is a 1992 crime film, the directorial debut of Quentin Tarantino."}],
     "gold_titles": ["Pulp Fiction", "Quentin Tarantino"], "type": "bridge"},
    {"id": "fb2", "question": "The author of 'A Brief History of Time' held a professorship that was previously held by whom?",
     "answer": "Isaac Newton",
     "paragraphs": [
         {"title": "A Brief History of Time", "text": "A Brief History of Time is a popular-science book by Stephen Hawking published in 1988."},
         {"title": "Stephen Hawking", "text": "Stephen Hawking was Lucasian Professor of Mathematics at the University of Cambridge."},
         {"title": "Lucasian Professor of Mathematics", "text": "The Lucasian Chair of Mathematics is a position at Cambridge once held by Isaac Newton."},
         {"title": "Isaac Newton", "text": "Isaac Newton was an English mathematician and physicist."},
         {"title": "University of Cambridge", "text": "The University of Cambridge is a collegiate research university in England."}],
     "gold_titles": ["Stephen Hawking", "Lucasian Professor of Mathematics"], "type": "bridge"},
    {"id": "fb3", "question": "Which river flows through the capital of the country where the Eiffel Tower is located?",
     "answer": "Seine",
     "paragraphs": [
         {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."},
         {"title": "Paris", "text": "Paris is the capital of France. The Seine river flows through Paris."},
         {"title": "France", "text": "France is a country in Western Europe whose capital is Paris."},
         {"title": "Seine", "text": "The Seine is a river in northern France that flows through Paris."},
         {"title": "Thames", "text": "The Thames is a river that flows through London, England."}],
     "gold_titles": ["Eiffel Tower", "Paris"], "type": "bridge"},
    {"id": "fb4", "question": "Are the bands Radiohead and Coldplay both from the same country?",
     "answer": "yes",
     "paragraphs": [
         {"title": "Radiohead", "text": "Radiohead are an English rock band formed in Abingdon, Oxfordshire, England."},
         {"title": "Coldplay", "text": "Coldplay are a British rock band formed in London, England."},
         {"title": "Nirvana (band)", "text": "Nirvana was an American rock band formed in Aberdeen, Washington."},
         {"title": "England", "text": "England is a country that is part of the United Kingdom."}],
     "gold_titles": ["Radiohead", "Coldplay"], "type": "comparison"},
    {"id": "fb5", "question": "What is the elevation of the mountain that is the highest in the country hosting the 2016 Summer Olympics?",
     "answer": "2,994 metres",
     "paragraphs": [
         {"title": "2016 Summer Olympics", "text": "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil."},
         {"title": "Brazil", "text": "Brazil is the largest country in South America. Its highest mountain is Pico da Neblina."},
         {"title": "Pico da Neblina", "text": "Pico da Neblina is the highest mountain in Brazil with an elevation of 2,994 metres."},
         {"title": "Aconcagua", "text": "Aconcagua is the highest mountain in the Americas, located in Argentina."}],
     "gold_titles": ["2016 Summer Olympics", "Brazil"], "type": "bridge"},
    {"id": "fb6", "question": "The actor who played Iron Man also starred in a 2008 film directed by whom?",
     "answer": "Jon Favreau",
     "paragraphs": [
         {"title": "Iron Man (2008 film)", "text": "Iron Man is a 2008 superhero film directed by Jon Favreau, starring Robert Downey Jr. as Iron Man."},
         {"title": "Robert Downey Jr.", "text": "Robert Downey Jr. is an American actor who played Tony Stark / Iron Man in the Marvel films."},
         {"title": "Jon Favreau", "text": "Jon Favreau is an American actor and filmmaker who directed Iron Man (2008)."},
         {"title": "The Avengers (2012 film)", "text": "The Avengers is a 2012 film directed by Joss Whedon."}],
     "gold_titles": ["Robert Downey Jr.", "Iron Man (2008 film)"], "type": "bridge"},
]


def _from_hotpot(n: int):
    from datasets import load_dataset
    last = None
    for kwargs in ({}, {"trust_remote_code": True}):
        try:
            ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{n}]", **kwargs)
            out = []
            for ex in ds:
                ctx = ex["context"]
                paras = [{"title": t, "text": " ".join(s)} for t, s in zip(ctx["title"], ctx["sentences"])]
                out.append({
                    "id": ex["id"], "question": ex["question"], "answer": ex["answer"],
                    "paragraphs": paras, "gold_titles": sorted(set(ex["supporting_facts"]["title"])),
                    "type": ex.get("type", ""),
                })
            return out
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def load_examples(n: int = 200, seed: int = 0):
    try:
        ex = _from_hotpot(n)
        if ex:
            print(f"[corpus] loaded {len(ex)} HotpotQA examples")
            return ex
    except Exception as e:  # noqa: BLE001
        print(f"[corpus] HotpotQA unavailable ({type(e).__name__}: {e}); using embedded fallback ({len(FALLBACK)})")
    fb = list(FALLBACK)
    random.Random(seed).shuffle(fb)
    # repeat to reach n so the loop has volume even offline
    out = (fb * ((n // len(fb)) + 1))[:n]
    return out
