"""FastAPI service and web GUI for the review search engine.

The index is loaded once at startup; every request reuses it. The Elasticsearch
baseline is off by default; pass --es to add it (needs ES running and indexed,
see app/search/es.py).

  uv run python -m app.search.service            # then open http://<host>:8000
  uv run python -m app.search.service --es       # also compare with ES at 127.0.0.1:9200

Endpoints:
  GET /                 web GUI (app/web/index.html)
  GET /api/stats        index statistics, whether ES is enabled
  GET /api/search       q, model=bm25|tfidf|es, k, offset, k1, b
  GET /api/compare      q, k, k1, b -> all models side by side, with pairwise overlap
"""

import argparse
import os
from contextlib import asynccontextmanager
from itertools import combinations
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.search.ranker import SearchEngine

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

engine = None
es = None


@asynccontextmanager
async def lifespan(app):
    global engine, es
    engine = SearchEngine(os.environ.get("SEARCH_INDEX", "index"),
                          os.environ.get("SEARCH_CATALOG") or None)
    print(f"index loaded in {engine.load_seconds:.1f}s", flush=True)
    es_url = os.environ.get("SEARCH_ES_URL")
    if es_url:
        from app.search.es import ESSearcher
        try:
            es = ESSearcher(engine, es_url, os.environ.get("SEARCH_ES_INDEX") or "yelp-reviews")
            print(f"Elasticsearch enabled: {es_url} ({es.n_docs:,} docs)", flush=True)
        except Exception as e:
            print(f"Elasticsearch disabled: {e}", flush=True)
    yield


app = FastAPI(title="Yelp Review Search", lifespan=lifespan)

QueryText = Query(..., min_length=1, max_length=500)
TopK = Query(10, ge=1, le=100)
K1 = Query(1.2, ge=0, le=10)
B = Query(0.75, ge=0, le=1)


def run_model(model, q, k, offset, k1, b):
    if model == "es":
        if es is None:
            raise HTTPException(400, "Elasticsearch is not enabled (start the service with --es)")
        return es.search(q, k, offset)
    return engine.search(q, model, k, offset, k1, b)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/stats")
def stats():
    return {**engine.stats(), "es_enabled": es is not None}


@app.get("/api/search")
def search(q: str = QueryText, model: Literal["bm25", "tfidf", "es"] = "bm25", k: int = TopK,
           offset: int = Query(0, ge=0, le=1000), k1: float = K1, b: float = B):
    return run_model(model, q, k, offset, k1, b)


@app.get("/api/compare")
def compare(q: str = QueryText, k: int = TopK, k1: float = K1, b: float = B):
    models = ["bm25", "tfidf"] + (["es"] if es is not None else [])
    res = {m: run_model(m, q, k, 0, k1, b) for m in models}
    pairs = []
    for x, y in combinations(models, 2):
        a = {r["doc_id"] for r in res[x]["results"]}
        c = {r["doc_id"] for r in res[y]["results"]}
        pairs.append({"a": x, "b": y, "overlap": len(a & c),
                      "jaccard": round(len(a & c) / len(a | c), 3) if a | c else 0.0})
    return {"query": q, "models": models, "pairs": pairs, **res}


def main():
    ap = argparse.ArgumentParser(description="Serve the search engine with a web GUI.")
    ap.add_argument("--index", default="index")
    ap.add_argument("--catalog", help="defaults to the catalog recorded in the index")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--es", nargs="?", const="http://127.0.0.1:9200", metavar="URL",
                    help="enable the Elasticsearch baseline (default URL http://127.0.0.1:9200)")
    ap.add_argument("--es-index", default="yelp-reviews")
    args = ap.parse_args()
    os.environ["SEARCH_INDEX"] = args.index
    if args.catalog:
        os.environ["SEARCH_CATALOG"] = args.catalog
    if args.es:
        os.environ["SEARCH_ES_URL"] = args.es
        os.environ["SEARCH_ES_INDEX"] = args.es_index
    else:
        os.environ.pop("SEARCH_ES_URL", None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
