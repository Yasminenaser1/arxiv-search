import torch
torch.set_num_threads(1)

import json, pickle, re, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

import os
DATA_DIR = os.environ.get("DATA_DIR", "data")
RERANK_AVAILABLE = os.environ.get("RERANK", "1") == "1"
RRF_K, POOL, RERANK_DEPTH = 60, 100, 50
S = {}

@asynccontextmanager
async def lifespan(app):
    t0 = time.time()
    S["docs"] = [json.loads(l) for l in open(f"{DATA_DIR}/abstracts.jsonl")]
    S["vecs"] = np.load(f"{DATA_DIR}/embeddings.npy")
    S["bm25"] = pickle.load(open(f"{DATA_DIR}/bm25.pkl", "rb"))
    S["bi"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    S["cross"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cpu") if RERANK_AVAILABLE else None
    print(f"loaded {len(S['docs'])} docs in {time.time()-t0:.1f}s")
    yield
    S.clear()

app = FastAPI(title="arxiv-search", lifespan=lifespan)
tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())

from functools import lru_cache

CACHE_STATS = {"hits": 0, "misses": 0}
from collections import deque
HISTORY = deque(maxlen=200)

def retrieve(query):
    qv = S["bi"].encode([query], normalize_embeddings=True)[0]
    dense = np.argsort(-(S["vecs"] @ qv))[:POOL]
    sparse = np.argsort(-S["bm25"].get_scores(tokenize(query)))[:POOL]
    fused = {}
    for rank, i in enumerate(dense, 1):  fused[i] = fused.get(i, 0) + 1/(RRF_K+rank)
    for rank, i in enumerate(sparse, 1): fused[i] = fused.get(i, 0) + 1/(RRF_K+rank)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]

def rerank(query, cands):
    docs = S["docs"]
    pairs = [[query, f"{docs[i]['title']}. {docs[i]['abstract'][:900]}"] for i in cands]
    scores = S["cross"].predict(pairs, batch_size=32, show_progress_bar=False)
    return sorted(zip(scores, cands), key=lambda x: -x[0])

def _search_uncached(q, k, rerank_on):
    t0 = time.perf_counter()
    cands = retrieve(q)
    t_retrieve = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    scored = rerank(q, cands[:RERANK_DEPTH])[:k] if rerank_on else [(None, i) for i in cands[:k]]
    t_rerank = (time.perf_counter() - t1) * 1000
    docs = S["docs"]
    return {
        "query": q,
        "timing_ms": {"retrieve": round(t_retrieve, 1), "rerank": round(t_rerank, 1)},
        "results": [
            {"id": docs[i]["id"], "title": docs[i]["title"],
             "abstract": docs[i]["abstract"][:400],
             "score": float(s) if s is not None else None}
            for s, i in scored
        ],
    }

@lru_cache(maxsize=512)
def _cached_search(q, k, rerank_on):
    return _search_uncached(q, k, rerank_on)

@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 10, rerank_on: bool = True):
    rerank_on = rerank_on and RERANK_AVAILABLE
    key = q.strip().lower()
    t0 = time.perf_counter()
    before = _cached_search.cache_info().hits
    out = dict(_cached_search(key, k, rerank_on))
    hit = _cached_search.cache_info().hits > before
    CACHE_STATS["hits" if hit else "misses"] += 1
    out["cached"] = hit
    total = round((time.perf_counter()-t0)*1000, 1)
    out["timing_ms"] = {"total": total} if hit else {**out["timing_ms"], "total": total}
    HISTORY.append({"q": key, "total": total, "cached": hit,
                    "rerank": rerank_on, "at": time.time()})
    return out

@app.post("/cache/clear")
def cache_clear():
    _cached_search.cache_clear()
    CACHE_STATS["hits"] = 0
    CACHE_STATS["misses"] = 0
    return {"cleared": True}

@app.get("/stats")
def stats():
    info = _cached_search.cache_info()
    total = CACHE_STATS["hits"] + CACHE_STATS["misses"]
    return {**CACHE_STATS,
            "hit_rate": round(CACHE_STATS["hits"]/total, 3) if total else 0.0,
            "cache_size": info.currsize, "maxsize": info.maxsize}

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(S["docs"]), "rerank_available": RERANK_AVAILABLE}


@app.get("/metrics")
def metrics():
    import numpy as _np
    rows = list(HISTORY)
    cold = [r["total"] for r in rows if not r["cached"]]
    warm = [r["total"] for r in rows if r["cached"]]
    def pct(v, p):
        return round(float(_np.percentile(v, p)), 1) if v else None
    total = CACHE_STATS["hits"] + CACHE_STATS["misses"]
    return {
        "requests": total,
        "cache": {**CACHE_STATS,
                  "hit_rate": round(CACHE_STATS["hits"]/total, 3) if total else 0.0,
                  "size": _cached_search.cache_info().currsize},
        "cold": {"n": len(cold), "p50": pct(cold, 50), "p95": pct(cold, 95)},
        "warm": {"n": len(warm), "p50": pct(warm, 50), "p95": pct(warm, 95)},
        "recent": rows[-25:],
    }

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
