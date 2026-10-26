"""Optional Elasticsearch baseline (off by default).

The same reviews are indexed with ES out of the box: the default `standard`
analyzer (lowercase only - no stop words, no stemming) and default BM25
(k1=1.2, b=0.75). Only settings that do not change analysis or scoring are set:
  number_of_shards=1   df and avgdl are global, as in our index
  _id = doc_id         results can be matched against our engine
  _source disabled     review text is read from docs/, like our engine does

  docker compose up -d                                      # start ES (compose.yaml)
  uv run python -m app.search.es index                      # load all reviews (--limit N to test)
  uv run python -m app.search.es search "authentic carbonara"
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter
from multiprocessing import Pool

from app.search.indexer import read_body
from app.search.ranker import PHRASE_RE, WORD_RE, SearchEngine, print_results

DEFAULT_URL = "http://127.0.0.1:9200"
INDEX_NAME = "yelp-reviews"
FIELD = "text"

SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "-1",            # restored after bulk loading
    "translog": {"durability": "async"},  # restored after bulk loading
}
MAPPINGS = {
    "_source": {"enabled": False},
    "dynamic": "strict",
    "properties": {FIELD: {"type": "text"}},
}


def connect(url):
    from elasticsearch import Elasticsearch
    return Elasticsearch(url, request_timeout=300)


# ---------------------------------------------------------------- indexing

def load_chunk(args):
    """Bulk-load doc_ids in [lo, hi); returns the number of documents indexed."""
    from elasticsearch import helpers
    url, index, catalog, docs_root, lo, hi = args
    db = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    rows = db.execute("SELECT doc_id, path FROM reviews WHERE doc_id >= ? AND doc_id < ?"
                      " ORDER BY doc_id", (lo, hi)).fetchall()
    db.close()
    actions = ({"_index": index, "_id": str(doc_id),
                FIELD: read_body(os.path.join(docs_root, path)).strip()}
               for doc_id, path in rows)
    ok, failed = helpers.bulk(connect(url), actions, chunk_size=2000, stats_only=True)
    if failed:
        raise RuntimeError(f"{failed} documents failed in [{lo}, {hi})")
    return ok


def build(args):
    es = connect(args.url)
    es.info()
    if es.indices.exists(index=args.index_name):
        if not args.force:
            sys.exit(f"index {args.index_name} exists; use --force to replace it")
        es.indices.delete(index=args.index_name)
    es.indices.create(index=args.index_name, settings=SETTINGS, mappings=MAPPINGS)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
    db = sqlite3.connect(f"file:{args.catalog}?mode=ro", uri=True)
    docs_root = db.execute("SELECT value FROM meta WHERE key = 'docs_root'").fetchone()[0]
    max_doc = db.execute("SELECT MAX(doc_id) FROM reviews").fetchone()[0]
    db.close()
    if args.limit:
        max_doc = min(max_doc, args.limit)
    catalog = os.path.abspath(args.catalog)
    tasks = [(args.url, args.index_name, catalog, docs_root, lo, min(lo + args.chunk_docs, max_doc + 1))
             for lo in range(1, max_doc + 1, args.chunk_docs)]
    log(f"loading {max_doc:,} docs into {args.index_name} in {len(tasks)} chunks, {args.workers} workers")

    done = 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(load_chunk, tasks), 1):
            done += n
            if i % 10 == 0 or i == len(tasks):
                log(f"{i}/{len(tasks)} chunks, {done:,} docs")

    es.indices.put_settings(index=args.index_name,
                            settings={"refresh_interval": "1s", "translog": {"durability": "request"}})
    es.indices.refresh(index=args.index_name)
    if args.forcemerge:
        log("force-merging to one segment")
        es.indices.forcemerge(index=args.index_name, max_num_segments=1, request_timeout=3600)
    count = es.count(index=args.index_name)["count"]
    log(f"done: {count:,} docs searchable")
    if count != max_doc:
        sys.exit(f"expected {max_doc:,} docs, ES has {count:,}")


# ---------------------------------------------------------------- searching

def parse_explanation(expl):
    """Per-term rows from a Lucene BM25 explanation tree."""
    rows, dl = {}, None

    def find(node, prefix):
        if node.get("description", "").startswith(prefix):
            return node
        for child in node.get("details", []):
            hit = find(child, prefix)
            if hit:
                return hit
        return None

    def walk(node):
        nonlocal dl
        desc = node.get("description", "")
        if desc.startswith(f"weight({FIELD}:"):
            term = desc[len(f"weight({FIELD}:"):].split(" in ")[0]
            idf, freq = find(node, "idf"), find(node, "freq")
            n, length = find(node, "n, number of documents"), find(node, "dl")
            dl = int(length["value"]) if length else dl
            row = rows.setdefault(term, {"term": term, "tf": int(freq["value"]) if freq else 0,
                                         "df": int(n["value"]) if n else 0,
                                         "idf": round(idf["value"], 4) if idf else 0.0,
                                         "contribution": 0.0})
            row["contribution"] = round(row["contribution"] + node["value"], 4)
            return
        for child in node.get("details", []):
            walk(child)

    walk(expl)
    return list(rows.values()), dl


class ESSearcher:
    def __init__(self, engine, url=DEFAULT_URL, index=INDEX_NAME):
        self.engine, self.index = engine, index
        self.es = connect(url)
        self.es.info()
        if not self.es.indices.exists(index=index):
            raise RuntimeError(f"ES index {index} not found; run: python -m app.search.es index")
        self.n_docs = self.es.count(index=index)["count"]

    def analyze(self, text):
        """Tokens produced by the index's analyzer (ES `standard`)."""
        resp = self.es.indices.analyze(index=self.index, field=FIELD, text=text)
        return [t["token"] for t in resp["tokens"]]

    def doc_freqs(self, terms):
        if not terms:
            return {}
        searches = []
        for t in terms:
            searches += [{"index": self.index},
                         {"size": 0, "track_total_hits": True, "query": {"term": {FIELD: t}}}]
        resp = self.es.msearch(searches=searches)
        return {t: r["hits"]["total"]["value"] for t, r in zip(terms, resp["responses"])}

    def search(self, query, k=10, offset=0):
        phrases = [p for p in PHRASE_RE.findall(query) if WORD_RE.search(p)]
        text = query.replace('"', " ")
        body = {"bool": {"should": [{"match": {FIELD: {"query": text}}}],
                         "filter": [{"match_phrase": {FIELD: p}} for p in phrases],
                         "minimum_should_match": 1}}
        t0 = time.time()
        resp = self.es.search(index=self.index, query=body, size=k, from_=offset,
                              explain=True, track_total_hits=True, source=False)
        took = time.time() - t0

        qterms = Counter(self.analyze(text))
        hits = resp["hits"]["hits"]
        docs = self.engine.fetch([int(h["_id"]) for h in hits])
        results = []
        for rank, h in enumerate(hits, offset + 1):
            d = docs[int(h["_id"])]
            explain, dl = parse_explanation(h["_explanation"])
            d.update(rank=rank, score=round(h["_score"], 4), doc_len=dl,
                     highlight=sorted({w.lower() for w in WORD_RE.findall(d["text"])
                                       if w.lower() in qterms}),
                     explain=explain)
            results.append(d)
        df = self.doc_freqs(list(qterms))
        return {
            "query": query,
            "model": "es",
            "params": {"k1": 1.2, "b": 0.75},
            "terms": [{"term": t, "df": df[t], "qtf": n} for t, n in qterms.items()],
            "unknown_terms": [t for t in qterms if df[t] == 0],
            "phrases": phrases,
            "total_hits": resp["hits"]["total"]["value"],
            "took_ms": round(took * 1000, 1),
            "results": results,
        }


def main():
    ap = argparse.ArgumentParser(description="Elasticsearch baseline.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--index-name", default=INDEX_NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("index", help="create the ES index and bulk-load reviews")
    ip.add_argument("--catalog", default="catalog.sqlite")
    ip.add_argument("--workers", type=int, default=8)
    ip.add_argument("--chunk-docs", type=int, default=50_000)
    ip.add_argument("--limit", type=int, help="load only doc_id <= LIMIT (for testing)")
    ip.add_argument("--forcemerge", action="store_true", help="merge into one segment afterwards")
    ip.add_argument("--force", action="store_true", help="replace an existing ES index")
    sp = sub.add_parser("search", help="query the ES index")
    sp.add_argument("query", nargs="+")
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--index", default="index", help="our index dir (for review metadata)")
    sp.add_argument("--width", type=int, default=160)
    args = ap.parse_args()

    if args.cmd == "index":
        build(args)
    else:
        searcher = ESSearcher(SearchEngine(args.index), args.url, args.index_name)
        print_results(searcher.search(" ".join(args.query), args.k), args.width)


if __name__ == "__main__":
    main()
