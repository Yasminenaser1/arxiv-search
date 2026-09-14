import json, random, re

SEED = 7
N = 50

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
random.seed(SEED)
picked = random.sample(docs, N)

def to_query(title):
    # drop subtitle after a colon — keeps the query from mirroring the title exactly
    t = title.split(":")[0]
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = " ".join(t.split()).lower()
    return t

cases = []
for d in picked:
    q = to_query(d["title"])
    if len(q.split()) < 3:        # too short to be a meaningful query
        continue
    cases.append({"query": q, "relevant_id": d["id"], "title": d["title"]})

with open("evals/easy.json", "w") as f:
    json.dump(cases, f, indent=2)

print(f"wrote {len(cases)} cases to evals/easy.json")
for c in cases[:5]:
    print(" ", c["query"])
