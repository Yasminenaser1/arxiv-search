import json, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer

CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "evals/easy.json"

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
vecs = np.load("data/embeddings.npy")
ids = [d["id"] for d in docs]
cases = json.load(open(CASE_FILE))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvecs = model.encode([c["query"] for c in cases], normalize_embeddings=True)

hits1 = hits5 = hits10 = 0
rr_total = 0.0
misses = []
latencies = []

for case, qv in zip(cases, qvecs):
    t0 = time.perf_counter()
    scores = vecs @ qv
    order = np.argsort(-scores)[:10]
    latencies.append((time.perf_counter() - t0) * 1000)

    ranked = [ids[i] for i in order]
    gold = case["relevant_id"]
    if gold in ranked:
        rank = ranked.index(gold) + 1
        rr_total += 1 / rank
        hits10 += 1
        if rank <= 5: hits5 += 1
        if rank == 1: hits1 += 1
    else:
        misses.append((case["query"], docs[order[0]]["title"]))

n = len(cases)
print(f"\n{CASE_FILE} — {n} queries")
print(f"  recall@1   {hits1/n:.3f}")
print(f"  recall@5   {hits5/n:.3f}")
print(f"  recall@10  {hits10/n:.3f}")
print(f"  MRR@10     {rr_total/n:.3f}")
print(f"  search p50 {np.percentile(latencies,50):.2f} ms")

if misses:
    print(f"\nmisses ({len(misses)}):")
    for q, got in misses[:8]:
        print(f"  q: {q}")
        print(f"     top hit: {got[:70]}")
