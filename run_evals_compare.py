import json, pickle, re, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer

CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "evals/hard.json"

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
ids = [d["id"] for d in docs]
cases = json.load(open(CASE_FILE))
vecs = np.load("data/embeddings.npy")
bm25 = pickle.load(open("data/bm25.pkl", "rb"))

tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvecs = model.encode([c["query"] for c in cases], normalize_embeddings=True)

def metrics(rank_fn, name):
    h1 = h5 = h10 = 0; rr = 0.0; lat = []
    per_query = {}
    for case, qv in zip(cases, qvecs):
        t0 = time.perf_counter()
        order = rank_fn(case["query"], qv)[:10]
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = [ids[i] for i in order]
        gold = case["relevant_id"]
        if gold in ranked:
            r = ranked.index(gold) + 1
            rr += 1 / r; h10 += 1
            if r <= 5: h5 += 1
            if r == 1: h1 += 1
            per_query[case["query"]] = r
        else:
            per_query[case["query"]] = None
    n = len(cases)
    print(f"\n{name}")
    print(f"  recall@1 {h1/n:.3f}  recall@5 {h5/n:.3f}  recall@10 {h10/n:.3f}  MRR@10 {rr/n:.3f}")
    print(f"  p50 {np.percentile(lat,50):.2f} ms   p95 {np.percentile(lat,95):.2f} ms")
    return per_query

dense = metrics(lambda q, qv: np.argsort(-(vecs @ qv)), "DENSE (all-MiniLM-L6-v2)")
sparse = metrics(lambda q, qv: np.argsort(-bm25.get_scores(tokenize(q))), "BM25")

print("\n--- where they disagree most ---")
rows = []
for q in dense:
    d, s = dense[q], sparse[q]
    dv = d if d else 99
    sv = s if s else 99
    rows.append((abs(dv - sv), q, d, s))
for _, q, d, s in sorted(rows, reverse=True)[:8]:
    print(f"  dense {str(d):>4}  bm25 {str(s):>4}   {q[:60]}")
