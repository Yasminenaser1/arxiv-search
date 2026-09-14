import json, pickle, re, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer

CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "evals/hard.json"
RRF_K = 60          # standard default from the RRF paper
POOL = 100          # how deep each method contributes

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
ids = [d["id"] for d in docs]
cases = json.load(open(CASE_FILE))
vecs = np.load("data/embeddings.npy")
bm25 = pickle.load(open("data/bm25.pkl", "rb"))

tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvecs = model.encode([c["query"] for c in cases], normalize_embeddings=True)

def rrf(query, qv, k=RRF_K):
    dense_order = np.argsort(-(vecs @ qv))[:POOL]
    sparse_order = np.argsort(-bm25.get_scores(tokenize(query)))[:POOL]
    fused = {}
    for rank, i in enumerate(dense_order, 1):
        fused[i] = fused.get(i, 0) + 1 / (k + rank)
    for rank, i in enumerate(sparse_order, 1):
        fused[i] = fused.get(i, 0) + 1 / (k + rank)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]

def score(name, fn):
    h1 = h5 = h10 = 0; rr = 0.0; lat = []
    for case, qv in zip(cases, qvecs):
        t0 = time.perf_counter()
        order = fn(case["query"], qv)[:10]
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = [ids[i] for i in order]
        if case["relevant_id"] in ranked:
            r = ranked.index(case["relevant_id"]) + 1
            rr += 1 / r; h10 += 1
            if r <= 5: h5 += 1
            if r == 1: h1 += 1
    n = len(cases)
    print(f"{name:<12} recall@1 {h1/n:.3f}  recall@5 {h5/n:.3f}  "
          f"recall@10 {h10/n:.3f}  MRR@10 {rr/n:.3f}  "
          f"p50 {np.percentile(lat,50):.2f}ms  p95 {np.percentile(lat,95):.2f}ms")

print()
score("dense", lambda q, qv: np.argsort(-(vecs @ qv)))
score("bm25",  lambda q, qv: np.argsort(-bm25.get_scores(tokenize(q))))
score("hybrid", rrf)

print("\nRRF k sweep (MRR@10):")
for k in [10, 20, 40, 60, 100]:
    rr = 0.0
    for case, qv in zip(cases, qvecs):
        ranked = [ids[i] for i in rrf(case["query"], qv, k=k)[:10]]
        if case["relevant_id"] in ranked:
            rr += 1 / (ranked.index(case["relevant_id"]) + 1)
    print(f"  k={k:<4} {rr/len(cases):.3f}")
