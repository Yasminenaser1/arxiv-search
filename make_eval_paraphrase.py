import json, re, subprocess

MODEL = "llama3.1:8b"
PROMPT = """Rewrite this search query in plain everyday language.

Rules:
- Keep the same meaning
- Replace technical jargon with ordinary words wherever a natural alternative exists
- 8 to 16 words
- lowercase, no punctuation
- Output ONLY the rewritten query

Query: {q}

Rewritten:"""

src = json.load(open("evals/hard.json"))
out = []
for i, c in enumerate(src, 1):
    r = subprocess.run(["ollama", "run", MODEL], input=PROMPT.format(q=c["query"]),
                       capture_output=True, text=True, timeout=120)
    q = " ".join(re.sub(r"[^\w\s\-]", " ", r.stdout.strip().split("\n")[0]).split()).lower()
    if not (4 <= len(q.split()) <= 25):
        print(f"  [{i}] rejected: {q[:50]!r}")
        continue
    out.append({"query": q, "relevant_id": c["relevant_id"], "title": c["title"]})
    print(f"  [{i}] {c['query'][:45]}\n       → {q}")

json.dump(out, open("evals/paraphrase.json", "w"), indent=2)
print(f"\nwrote {len(out)} cases")
