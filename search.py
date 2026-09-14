import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
vecs = np.load("data/embeddings.npy")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, k=5):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = vecs @ q                      # normalized → dot product is cosine
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), docs[i]["title"]) for i in top]

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "transformers for time series forecasting"
    print(f"query: {query}\n")
    for score, title in search(query):
        print(f"{score:.3f}  {title}")
