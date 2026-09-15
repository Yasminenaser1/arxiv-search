import json, os, pickle, re
import numpy as np
import pytest

DATA = os.environ.get("DATA_DIR", "data")
from sentence_transformers import SentenceTransformer

@pytest.fixture(scope="module")
def index():
    docs = [json.loads(l) for l in open(f"{DATA}/abstracts.jsonl")]
    return {
        "docs": docs,
        "vecs": np.load(f"{DATA}/embeddings.npy"),
        "bm25": pickle.load(open(f"{DATA}/bm25.pkl", "rb")),
        "model": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"),
    }

tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())

def test_index_shapes_align(index):
    assert len(index["docs"]) == index["vecs"].shape[0]
    assert index["vecs"].shape[1] == 384

def test_embeddings_are_normalized(index):
    norms = np.linalg.norm(index["vecs"][:200], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

def test_dense_finds_obvious_match(index):
    qv = index["model"].encode(["adversarial examples fooling image classifiers"],
                               normalize_embeddings=True)[0]
    top = np.argsort(-(index["vecs"] @ qv))[:5]
    titles = " ".join(index["docs"][i]["title"].lower() for i in top)
    assert "adversarial" in titles

def test_bm25_exact_term_wins(index):
    scores = index["bm25"].get_scores(tok("federated learning"))
    top = index["docs"][int(np.argmax(scores))]
    assert "federated" in (top["title"] + top["abstract"]).lower()

@pytest.mark.skipif(DATA != "data", reason="eval gold docs are from the 50k local corpus")
def test_rrf_beats_either_alone(index):
    """The core claim of the architecture — guard it."""
    cases = json.load(open("evals/hard.json"))[:20]
    qvecs = index["model"].encode([c["query"] for c in cases], normalize_embeddings=True)
    ids = [d["id"] for d in index["docs"]]

    def mrr(rank_fn):
        total = 0.0
        for c, qv in zip(cases, qvecs):
            ranked = [ids[i] for i in rank_fn(c["query"], qv)[:10]]
            if c["relevant_id"] in ranked:
                total += 1 / (ranked.index(c["relevant_id"]) + 1)
        return total / len(cases)

    def rrf(q, qv, k=60):
        d = np.argsort(-(index["vecs"] @ qv))[:100]
        s = np.argsort(-index["bm25"].get_scores(tok(q)))[:100]
        f = {}
        for r, i in enumerate(d, 1): f[i] = f.get(i, 0) + 1/(k+r)
        for r, i in enumerate(s, 1): f[i] = f.get(i, 0) + 1/(k+r)
        return [i for i, _ in sorted(f.items(), key=lambda x: -x[1])]

    dense = mrr(lambda q, qv: np.argsort(-(index["vecs"] @ qv)))
    sparse = mrr(lambda q, qv: np.argsort(-index["bm25"].get_scores(tok(q))))
    hybrid = mrr(rrf)
    assert hybrid >= max(dense, sparse), f"hybrid {hybrid} < dense {dense} / bm25 {sparse}"
