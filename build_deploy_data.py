import json, pickle, random, re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

N, SEED = 3000, 42
docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
random.seed(SEED)
sub = random.sample(docs, N)

with open("deploy_data/abstracts.jsonl", "w") as f:
    for d in sub:
        f.write(json.dumps(d) + "\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vecs = model.encode([f"{d['title']}. {d['abstract']}" for d in sub],
                    batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.save("deploy_data/embeddings.npy", vecs.astype(np.float32))

tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
pickle.dump(BM25Okapi([tok(f"{d['title']}. {d['abstract']}") for d in sub]),
            open("deploy_data/bm25.pkl", "wb"))

print(f"built deploy_data/ — {N} docs")
