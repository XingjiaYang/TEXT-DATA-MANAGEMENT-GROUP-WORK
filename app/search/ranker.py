"""Ranking functions implemented from scratch, and the search engine around them.

Vector space model (SMART ltc.ltc):
  w(t,d) = (1 + ln tf) * ln(N / df)            same weighting for the query
  score  = sum_t w(t,q) * w(t,d) / (|q| * |d|)
  |d| is over *all* terms of d; it is computed once from the postings and
  cached as index/tfidf_norm.npy (the indexer recreates the dir, so it never
  goes stale).

Okapi BM25 (Lucene's non-negative IDF, so very common terms like "not",
df > N/2, do not get a negative weight):
  idf    = ln(1 + (N - df + 0.5) / (df + 0.5))
  score  = sum_t qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
  avgdl comes from the index (computed globally at indexing time).

Query semantics: OR over all query terms, like a web search box. Text in
"double quotes" is a phrase that a result must contain (positional match).

Command line:
  uv run python -m app.search.ranker "best carbonara" --model both -k 5
  uv run python -m app.search.ranker            # interactive prompt
  uv run python -m app.search.ranker "best carbonara" --es --model all   # + Elasticsearch
"""

import argparse
import os
import re
import shlex
import sqlite3
import sys
import textwrap
import time
from collections import Counter

import numpy as np

from app.search.index import InvertedIndex
from app.search.indexer import read_body
from app.search.tokenizer import analyze, normalize_token, tokenize

MODELS = ("bm25", "tfidf")
PHRASE_RE = re.compile(r'"([^"]+)"')
WORD_RE = re.compile(r"[^\W_]+")
NORM_BLOCK = 50_000_000  # postings per block when computing document norms


def bm25_idf(n_docs, df):
    return np.log1p((n_docs - df + 0.5) / (df + 0.5))


def tfidf_idf(n_docs, df):
    return np.log(n_docs / df)


def tfidf_doc_norms(index, block=NORM_BLOCK):
    """|d| = sqrt(sum_t ((1 + ln tf) * idf_t)^2) over every term in every doc."""
    df = np.diff(index.term_ptr)
    idf = tfidf_idf(index.n_docs, df.astype(np.float64)).astype(np.float32)
    norm2 = np.zeros(len(index.doc_len), dtype=np.float64)
    # Walk term ranges holding about `block` postings each.
    cuts = np.searchsorted(index.term_ptr, np.arange(0, index.term_ptr[-1], block))
    cuts = np.unique(np.append(cuts, len(df)))
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        s, e = index.term_ptr[t0], index.term_ptr[t1]
        w = (1 + np.log(index.post_tf[s:e].astype(np.float32))) * np.repeat(idf[t0:t1], df[t0:t1])
        norm2 += np.bincount(index.post_docs[s:e], weights=w * w, minlength=len(norm2))
    return np.sqrt(norm2).astype(np.float32)


def load_tfidf_norms(index, index_dir):
    path = os.path.join(index_dir, "tfidf_norm.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    norms = tfidf_doc_norms(index)
    try:
        np.save(path, norms)
    except OSError:
        pass
    return norms


def parse_query(query):
    """Split a query into scored terms (with counts) and required phrases."""
    phrases = [p for p in PHRASE_RE.findall(query) if tokenize(p)]
    return Counter(tokenize(query.replace('"', " "))), phrases


def tf_in_doc(index, tid, doc_id):
    docs, tfs = index.postings(tid)
    i = np.searchsorted(docs, doc_id)
    return int(tfs[i]) if i < len(docs) and docs[i] == doc_id else 0


class SearchEngine:
    def __init__(self, index_dir="index", catalog=None):
        t0 = time.time()
        self.index = InvertedIndex(index_dir)
        self.catalog = catalog or self.index.meta["catalog"]
        self.docs_root = self.index.meta["docs_root"]
        self.tfidf_norm = load_tfidf_norms(self.index, index_dir)
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------ scoring

    def _query_terms(self, qterms):
        """[(term, tid, qtf, df)] for query terms present in the vocabulary."""
        out = []
        for term, qtf in qterms.items():
            tid = self.index.term_id(term)
            if tid is not None:
                out.append((term, tid, qtf, self.index.df(tid)))
        return out

    def score_bm25(self, terms, k1, b):
        idx = self.index
        scores = np.zeros(len(idx.doc_len), dtype=np.float32)
        for _, tid, qtf, df in terms:
            docs, tf = idx.postings(tid)
            tf = tf.astype(np.float32)
            norm = k1 * (1 - b + b * idx.doc_len[docs] / idx.avgdl)
            scores[docs] += qtf * bm25_idf(idx.n_docs, df) * tf * (k1 + 1) / (tf + norm)
        return scores

    def score_tfidf(self, terms):
        idx = self.index
        scores = np.zeros(len(idx.doc_len), dtype=np.float32)
        wq = {tid: (1 + np.log(qtf)) * tfidf_idf(idx.n_docs, df) for _, tid, qtf, df in terms}
        q_norm = np.sqrt(sum(w * w for w in wq.values()))
        if q_norm == 0:
            return scores
        for _, tid, _, df in terms:
            docs, tf = idx.postings(tid)
            wd = (1 + np.log(tf.astype(np.float32))) * tfidf_idf(idx.n_docs, df)
            scores[docs] += wq[tid] * wd
        nz = np.flatnonzero(scores)
        scores[nz] /= q_norm * self.tfidf_norm[nz]
        return scores

    def explain(self, model, terms, doc_id, k1, b):
        """Per-term contribution to one document's score."""
        idx = self.index
        dl = int(idx.doc_len[doc_id])
        rows = []
        if model == "tfidf":
            wq = {tid: (1 + np.log(qtf)) * tfidf_idf(idx.n_docs, df) for _, tid, qtf, df in terms}
            denom = np.sqrt(sum(w * w for w in wq.values())) * float(self.tfidf_norm[doc_id])
        for term, tid, qtf, df in terms:
            tf = tf_in_doc(idx, tid, doc_id)
            if model == "bm25":
                idf = bm25_idf(idx.n_docs, df)
                c = qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / idx.avgdl)) if tf else 0.0
            else:
                idf = tfidf_idf(idx.n_docs, df)
                c = wq[tid] * (1 + np.log(tf)) * idf / denom if tf and denom else 0.0
            rows.append({"term": term, "tf": tf, "df": df, "idf": round(float(idf), 4),
                         "contribution": round(float(c), 4)})
        return rows

    # ------------------------------------------------------------ search

    def search(self, query, model="bm25", k=10, offset=0, k1=1.2, b=0.75):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        t0 = time.time()
        qterms, phrases = parse_query(query)
        terms = self._query_terms(qterms)
        scores = self.score_bm25(terms, k1, b) if model == "bm25" else self.score_tfidf(terms)

        hits = np.flatnonzero(scores)
        for phrase in phrases:
            pdocs, _ = self.index.phrase_match(*analyze(phrase))
            hits = np.intersect1d(hits, pdocs, assume_unique=True)

        # Top (offset + k) by score, ties broken by doc_id for stable output.
        n_top = min(offset + k, len(hits))
        top = hits[np.argpartition(-scores[hits], n_top - 1)[:n_top]] if n_top else hits[:0]
        top = top[np.lexsort((top, -scores[top]))][offset:offset + k]
        took = time.time() - t0

        docs = self.fetch(top.tolist())
        results = []
        for rank, doc_id in enumerate(top.tolist(), offset + 1):
            d = docs[doc_id]
            d.update(rank=rank, score=round(float(scores[doc_id]), 4),
                     doc_len=int(self.index.doc_len[doc_id]),
                     highlight=self.highlight_words(d["text"], qterms),
                     explain=self.explain(model, terms, doc_id, k1, b))
            results.append(d)
        return {
            "query": query,
            "model": model,
            "params": {"k1": k1, "b": b} if model == "bm25" else {},
            "terms": [{"term": t, "df": df, "qtf": q} for t, _, q, df in terms],
            "unknown_terms": [t for t in qterms if self.index.term_id(t) is None],
            "phrases": phrases,
            "total_hits": int(len(hits)),
            "took_ms": round(took * 1000, 1),
            "results": results,
        }

    def fetch(self, doc_ids):
        """Catalog metadata and review text for the given doc_ids."""
        if not doc_ids:
            return {}
        db = sqlite3.connect(f"file:{self.catalog}?mode=ro", uri=True)
        rows = db.execute(
            "SELECT r.doc_id, r.review_id, r.stars, r.date, r.path, r.business_id,"
            " b.name, b.city, b.state, b.categories, b.stars"
            " FROM reviews r JOIN businesses b ON b.business_id = r.business_id"
            f" WHERE r.doc_id IN ({','.join('?' * len(doc_ids))})", doc_ids).fetchall()
        db.close()
        out = {}
        for (doc_id, review_id, stars, date, path, business_id,
             name, city, state, categories, business_stars) in rows:
            out[doc_id] = {
                "doc_id": doc_id, "review_id": review_id, "stars": stars, "date": date,
                "business_id": business_id, "business_name": name, "city": city,
                "state": state, "categories": categories, "business_stars": business_stars,
                "text": read_body(os.path.join(self.docs_root, path)).strip(),
            }
        return out

    @staticmethod
    def highlight_words(text, qterms):
        """Surface words in text whose analysed form is a query term."""
        words = set()
        for w in WORD_RE.findall(text):
            res = normalize_token(w)
            if res and not res[1] and res[0] in qterms:
                words.add(w.lower())
        return sorted(words)

    def stats(self):
        m = self.index.meta
        return {"n_docs": m["n_docs"], "n_terms": m["n_terms"], "n_postings": m["n_postings"],
                "avgdl": round(m["avgdl"], 2), "built_at": m["built_at"]}


# ---------------------------------------------------------------- CLI

def print_results(res, width):
    p = "".join(f" {k}={v}" for k, v in res["params"].items())
    print(f"\n== {res['model'].upper()}{p} | {res['total_hits']:,} hits | {res['took_ms']} ms")
    if res["unknown_terms"]:
        print(f"   (not in index: {', '.join(res['unknown_terms'])})")
    for r in res["results"]:
        print(f"{r['rank']:>3}. {r['score']:8.4f}  doc {r['doc_id']}  {'★' * r['stars']:<5} "
              f"{r['date'][:10]}  {r['business_name']} ({r['city']}, {r['state']})")
        snippet = " ".join(r["text"].split())
        print(textwrap.indent(textwrap.shorten(snippet, width, placeholder=" …"), "      "))


CLI_MODELS = MODELS + ("es", "both", "all")  # both = bm25 + tfidf, all = + es


def run(engine, es, query, opts):
    models = {"both": MODELS, "all": MODELS + ("es",)}.get(opts.model, (opts.model,))
    for m in models:
        if m == "es":
            if es is None:
                print("\n(Elasticsearch not enabled; pass --es)")
                continue
            print_results(es.search(query, opts.k), opts.width)
        else:
            print_results(engine.search(query, m, opts.k, 0, opts.k1, opts.b), opts.width)


def repl(engine, es, opts):
    print("Type a query (\"quotes\" = phrase). Commands: :model bm25|tfidf|es|both|all, "
          ":k1 X, :b X, :k N, :q")
    while True:
        try:
            line = input(f"[{opts.model} k1={opts.k1} b={opts.b} k={opts.k}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            cmd, *arg = shlex.split(line[1:]) or [""]
            if cmd in ("q", "quit", "exit"):
                return
            try:
                if cmd == "model" and arg and arg[0] in CLI_MODELS:
                    opts.model = arg[0]
                elif cmd in ("k1", "b") and arg:
                    setattr(opts, cmd, float(arg[0]))
                elif cmd == "k" and arg:
                    opts.k = int(arg[0])
                else:
                    print("unknown command")
            except ValueError:
                print("bad value")
            continue
        run(engine, es, line, opts)


def main():
    ap = argparse.ArgumentParser(description="Search Yelp reviews with BM25 / TF-IDF.")
    ap.add_argument("query", nargs="*", help="query; omit for an interactive prompt")
    ap.add_argument("--model", choices=CLI_MODELS, default="bm25",
                    help="both = bm25 + tfidf; all = bm25 + tfidf + es")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("-k", type=int, default=10, help="results per model")
    ap.add_argument("--index", default="index")
    ap.add_argument("--catalog", help="defaults to the catalog recorded in the index")
    ap.add_argument("--width", type=int, default=160, help="snippet width")
    ap.add_argument("--es", nargs="?", const="http://127.0.0.1:9200", metavar="URL",
                    help="enable the Elasticsearch baseline (off by default)")
    opts = ap.parse_args()

    engine = SearchEngine(opts.index, opts.catalog)
    print(f"loaded index in {engine.load_seconds:.1f}s", file=sys.stderr)
    es = None
    if opts.es:
        from app.search.es import ESSearcher
        es = ESSearcher(engine, opts.es)
    if opts.query:
        run(engine, es, " ".join(opts.query), opts)
    else:
        repl(engine, es, opts)


if __name__ == "__main__":
    main()
