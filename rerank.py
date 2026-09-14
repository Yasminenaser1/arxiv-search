import json, pickle, re, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "evals/hard.json"
RRF_K, POOL = 60, 100

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
ids = [d["id"] for d in docs]
cases = json.load(open(CASE_FILE))
vecs = np.load("data/embeddings.npy")
bm25 = pickle.load(open("data/bm25.pkl", "rb"))

tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
qvecs = bi.encode([c["query"] for c in cases], normalize_embeddings=True)

def rrf(query, qv):
    d = np.argsort(-(vecs @ qv))[:POOL]
    s = np.argsort(-bm25.get_scores(tokenize(query)))[:POOL]
    fused = {}
    for rank, i in enumerate(d, 1): fused[i] = fused.get(i, 0) + 1/(RRF_K+rank)
    for rank, i in enumerate(s, 1): fused[i] = fused.get(i, 0) + 1/(RRF_K+rank)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]

def reranked(query, qv, depth):
    cands = rrf(query, qv)[:depth]
    pairs = [[query, f"{docs[i]['title']}. {docs[i]['abstract'][:900]}"] for i in cands]
    scores = cross.predict(pairs, batch_size=32, show_progress_bar=False)
    return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]

def score(name, fn):
    h1 = h5 = h10 = 0; rr = 0.0; lat = []
    for case, qv in zip(cases, qvecs):
        t0 = time.perf_counter()
        order = fn(case["query"], qv)[:10]
        lat.append((time.perf_counter()-t0)*1000)
        ranked = [ids[i] for i in order]
        if case["relevant_id"] in ranked:
            r = ranked.index(case["relevant_id"]) + 1
            rr += 1/r; h10 += 1
            if r <= 5: h5 += 1
            if r == 1: h1 += 1
    n = len(cases)
    print(f"{name:<18} recall@1 {h1/n:.3f}  recall@5 {h5/n:.3f}  recall@10 {h10/n:.3f}  "
          f"MRR@10 {rr/n:.3f}  p50 {np.percentile(lat,50):.1f}ms  p95 {np.percentile(lat,95):.1f}ms")

print()
score("hybrid (no rerank)", rrf)
for depth in [20, 50, 100]:
    score(f"rerank top-{depth}", lambda q, qv, d=depth: reranked(q, qv, d))
