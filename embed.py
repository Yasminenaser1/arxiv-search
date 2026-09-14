import json, time
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

docs = [json.loads(l) for l in open("data/abstracts.jsonl")]
print(f"{len(docs)} docs")

# title + abstract in one string — whole-doc embedding is the baseline;
# chunking is a later experiment
texts = [f"{d['title']}. {d['abstract']}" for d in docs]

model = SentenceTransformer(MODEL)
t0 = time.time()
vecs = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # lets cosine sim be a plain dot product
)
print(f"embedded in {time.time()-t0:.1f}s | shape {vecs.shape}")

np.save("data/embeddings.npy", vecs.astype(np.float32))
print("saved data/embeddings.npy")
