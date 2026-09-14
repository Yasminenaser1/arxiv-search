import json, random
from datasets import load_dataset

OUT = "data/abstracts.jsonl"
TARGET = 50000
SEED = 42

ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
print("available:", len(ds))

# dataset is ordered by arXiv ID (chronological), so a head slice is
# the oldest papers only — sample across the full range instead
random.seed(SEED)
idx = sorted(random.sample(range(len(ds)), min(TARGET * 2, len(ds))))

n = 0
with open(OUT, "w") as f:
    for i in idx:
        row = ds[i]
        title = " ".join((row.get("title") or "").split())
        abstract = " ".join((row.get("abstract") or "").split())
        if not title or not abstract:
            continue
        f.write(json.dumps({
            "id": f"arxiv-{i}",       # keeps position → rough chronology
            "source_index": i,
            "title": title,
            "abstract": abstract,
        }) + "\n")
        n += 1
        if n >= TARGET:
            break

print(f"wrote {n} records (seed {SEED})")
