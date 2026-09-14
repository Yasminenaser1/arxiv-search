import asyncio, json, statistics, time, urllib.parse, urllib.request

BASE = "http://127.0.0.1:8000"
CONCURRENCY = [1, 2, 4, 8]
REQUESTS = 40
QUERIES = [c["query"] for c in json.load(open("evals/hard.json"))]

def clear():
    req = urllib.request.Request(f"{BASE}/cache/clear", method="POST")
    urllib.request.urlopen(req).read()

def fetch(q, rerank_on):
    u = f"{BASE}/search?q={urllib.parse.quote(q)}&k=10&rerank_on={str(rerank_on).lower()}"
    t0 = time.perf_counter()
    with urllib.request.urlopen(u, timeout=120) as r:
        r.read()
    return (time.perf_counter() - t0) * 1000

async def run(conc, rerank_on, queries):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(conc)
    async def one(q):
        async with sem:
            return await loop.run_in_executor(None, fetch, q, rerank_on)
    t0 = time.perf_counter()
    lat = await asyncio.gather(*[one(q) for q in queries])
    return lat, time.perf_counter() - t0

def report(label, lat, wall):
    s = sorted(lat)
    print(f"  {label:<18} p50 {statistics.median(s):7.1f}ms  p95 {s[int(len(s)*0.95)-1]:7.1f}ms  "
          f"max {s[-1]:7.1f}ms  throughput {len(s)/wall:5.1f} req/s")

async def main():
    # warm up lazy init so it doesn't pollute the first measurement
    clear(); fetch(QUERIES[0], True); clear()

    for rerank_on in [True, False]:
        print(f"\n=== rerank={'on' if rerank_on else 'off'} | cold cache, unique queries ===")
        for c in CONCURRENCY:
            clear()
            qs = [f"{QUERIES[i % len(QUERIES)]} v{c}" for i in range(REQUESTS)]
            lat, wall = await run(c, rerank_on, qs)
            report(f"concurrency {c}", lat, wall)

    print("\n=== warm cache (same query repeated) ===")
    clear()
    fetch(QUERIES[0], True)
    for c in CONCURRENCY:
        lat, wall = await run(c, True, [QUERIES[0]] * REQUESTS)
        report(f"concurrency {c}", lat, wall)

asyncio.run(main())
