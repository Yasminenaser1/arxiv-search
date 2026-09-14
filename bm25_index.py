import json, pickle, re, time
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
corpus = [tokenize(f"{d['title']}. {d['abstract']}") for d in docs]

t0 = time.time()
bm25 = BM25Okapi(corpus)
print(f"indexed {len(corpus)} docs in {time.time()-t0:.1f}s")

with open("data/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
print("saved data/bm25.pkl")
