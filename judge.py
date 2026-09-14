import json, os, sys
import numpy as np
from sentence_transformers import SentenceTransformer

CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "evals/hard.json"
QRELS = CASE_FILE.replace(".json", "_qrels.json")
DEPTH = 3

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
by_id = {d["id"]: d for d in docs}
ids = [d["id"] for d in docs]
vecs = np.load("data/embeddings.npy")
cases = json.load(open(CASE_FILE))

qrels = json.load(open(QRELS)) if os.path.exists(QRELS) else {}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvecs = model.encode([c["query"] for c in cases], normalize_embeddings=True)

print("y = relevant   n = not   s = skip query   q = save and quit\n")

for ci, (case, qv) in enumerate(zip(cases, qvecs), 1):
    q = case["query"]
    judged = qrels.setdefault(q, {})
    judged[case["relevant_id"]] = 1          # gold is relevant by construction

    order = np.argsort(-(vecs @ qv))[:DEPTH]
    todo = [ids[i] for i in order if ids[i] not in judged]
    if not todo:
        continue

    print(f"\n=== [{ci}/{len(cases)}] {q}")
    print(f"    gold: {by_id[case['relevant_id']]['title'][:75]}")

    for did in todo:
        d = by_id[did]
        print(f"\n  {d['title'][:90]}")
        print(f"  {d['abstract'][:240]}...")
        ans = input("  relevant? [y/n/s/q] ").strip().lower()
        if ans == "q":
            json.dump(qrels, open(QRELS, "w"), indent=2)
            print(f"\nsaved {QRELS}"); sys.exit()
        if ans == "s":
            break
        judged[did] = 1 if ans == "y" else 0

json.dump(qrels, open(QRELS, "w"), indent=2)
total = sum(len(v) for v in qrels.values())
rel = sum(sum(v.values()) for v in qrels.values())
print(f"\nsaved {QRELS} — {total} judgments, {rel} relevant")
