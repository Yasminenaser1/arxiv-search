import requests, feedparser, time

QUERY = "cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.AI"
UA = {"User-Agent": "arxiv-search/0.1 (student project; contact yasminenaser1@github)"}
url = ("https://export.arxiv.org/api/query"
       f"?search_query={requests.utils.quote(QUERY)}"
       "&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending")

for attempt in range(4):
    r = requests.get(url, headers=UA, timeout=60)
    print(f"attempt {attempt+1}: status {r.status_code} | {len(r.text)} bytes")
    if r.status_code == 200:
        break
    time.sleep(5 * (attempt + 1))

feed = feedparser.parse(r.text)
print("entries:", len(feed.entries))
if feed.entries:
    print("first title:", feed.entries[0].title[:80])
else:
    print(r.text[:300])
