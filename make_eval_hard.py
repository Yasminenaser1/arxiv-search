import json, random, re, subprocess

SEED = 13
N = 50
MODEL = "llama3.1:8b"

PROMPT = """You are helping build a search benchmark for a paper search engine.

Read this abstract and write ONE search query a researcher would type to find this paper.

Rules:
- Describe the problem or method in your own words
- Do NOT reuse distinctive words from the title
- 6 to 12 words
- No punctuation, no quotes, lowercase
- Output ONLY the query, nothing else

Title: {title}
Abstract: {abstract}

Query:"""

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
easy_ids = {c["relevant_id"] for c in json.load(open("evals/easy.json"))}
pool = [d for d in docs if d["id"] not in easy_ids]

random.seed(SEED)
picked = random.sample(pool, N)

def ask(prompt):
    r = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt, capture_output=True, text=True, timeout=120,
    )
    return r.stdout.strip()

cases = []
for i, d in enumerate(picked, 1):
    raw = ask(PROMPT.format(title=d["title"], abstract=d["abstract"][:1200]))
    q = " ".join(re.sub(r"[^\w\s\-]", " ", raw.split("\n")[0]).split()).lower()
    if not (4 <= len(q.split()) <= 20):
        print(f"  [{i}] rejected: {q[:60]!r}")
        continue
    cases.append({"query": q, "relevant_id": d["id"], "title": d["title"]})
    print(f"  [{i}] {q}")

with open("evals/hard.json", "w") as f:
    json.dump(cases, f, indent=2)
print(f"\nwrote {len(cases)} cases to evals/hard.json")
