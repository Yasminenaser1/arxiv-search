import json, re
cases = json.load(open("evals/hard.json"))[:20]
lines = open("my_queries.txt").read().split("\n")
qs, cur = [], None
for ln in lines:
    if ln.startswith("#"):
        qs.append(cur); cur = None
    elif ln.strip() and cur is None:
        cur = " ".join(re.sub(r"[^\w\s\-]", " ", ln).split()).lower()
qs.append(cur); qs = qs[1:]
out = [{"query": q, "relevant_id": c["relevant_id"], "title": c["title"]}
       for q, c in zip(qs, cases) if q]
json.dump(out, open("evals/handwritten.json", "w"), indent=2)
print(f"loaded {len(out)} queries")
for o in out: print(" ", o["query"])
