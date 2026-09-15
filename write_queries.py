import json

cases = json.load(open("evals/hard.json"))[:20]
docs = {json.loads(l)["id"]: json.loads(l) for l in open("data/abstracts.jsonl")}

print("Describe each paper in your own words. Enter to skip, Ctrl-C to stop.\n")

out = []
for i, c in enumerate(cases, 1):
    print(f"[{i}/20] {docs[c['relevant_id']]['title']}")
    try:
        q = input("  > ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print(); break
    if q:
        out.append({"query": q, "relevant_id": c["relevant_id"], "title": c["title"]})

json.dump(out, open("evals/handwritten.json", "w"), indent=2)
print(f"\nwrote {len(out)} queries to evals/handwritten.json")
