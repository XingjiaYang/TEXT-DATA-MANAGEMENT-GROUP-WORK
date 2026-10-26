"""Build the positional inverted index from the extracted Markdown reviews.

Map:   worker processes take doc_id ranges, analyse each review body and build a
       local sorted index for the chunk (local vocab, postings, positions).
Merge: the parent unions the vocabularies, remaps local term ids to global ones
       and stable-sorts all postings by term id; chunks are in doc_id order, so
       doc_ids stay ascending within each postings list. Positions are gathered
       into the new order in one vectorised step.

Output (numpy arrays, loaded with mmap by app.search.index.InvertedIndex):
  vocab.txt         term per line; line number = term id
  term_ptr.npy      int64[V+1]  postings of term t: [term_ptr[t], term_ptr[t+1])
  post_docs.npy     int32[P]    doc ids, ascending within a term
  post_tf.npy       uint16[P]   term frequency
  term_pos_ptr.npy  int64[V+1]  positions of term t start at term_pos_ptr[t];
                                within a term, posting i owns tf[i] positions
  positions.npy     uint16[L]   token positions, ascending within a posting
  doc_len.npy       int32[N+1]  number of index terms per doc (slot 0 unused)
  meta.json         N, avgdl (computed globally here), sizes, settings

Usage:
  uv run python -m app.search.indexer --catalog catalog.sqlite --out index
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

from app.search.tokenizer import analyze

MAX_POSITION = np.iinfo(np.uint16).max


def read_body(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n---\n", 1)[1]


def index_chunk(args):
    """Build a local index for doc_ids in [lo, hi)."""
    db_path, docs_root, lo, hi = args
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute("SELECT doc_id, path FROM reviews WHERE doc_id >= ? AND doc_id < ?"
                      " ORDER BY doc_id", (lo, hi)).fetchall()
    db.close()

    vocab = {}
    tok_term, tok_doc, tok_pos = [], [], []
    doc_ids, doc_lens = [], []
    for doc_id, path in rows:
        terms, positions = analyze(read_body(os.path.join(docs_root, path)))
        tok_term.extend(vocab.setdefault(t, len(vocab)) for t in terms)
        tok_pos.extend(positions)
        tok_doc.extend([doc_id] * len(terms))
        doc_ids.append(doc_id)
        doc_lens.append(len(terms))

    term = np.array(tok_term, dtype=np.int32)
    doc = np.array(tok_doc, dtype=np.int32)
    pos = np.array(tok_pos, dtype=np.int64)
    if len(pos) and pos.max() > MAX_POSITION:
        raise ValueError(f"position {pos.max()} does not fit in uint16")
    order = np.lexsort((pos, doc, term))
    term, doc, pos = term[order], doc[order], pos[order].astype(np.uint16)

    # One posting per run of equal (term, doc).
    new = np.ones(len(term), dtype=bool)
    new[1:] = (term[1:] != term[:-1]) | (doc[1:] != doc[:-1])
    starts = np.flatnonzero(new)
    tf = np.diff(np.append(starts, len(term)))
    return (list(vocab), term[starts], doc[starts], tf.astype(np.uint16), pos,
            np.array(doc_ids, dtype=np.int32), np.array(doc_lens, dtype=np.int32))


def merge(chunks, n_slots):
    """Merge chunk-local indexes into global CSR arrays."""
    vocab = sorted(set().union(*(c[0] for c in chunks)))
    gid = {t: i for i, t in enumerate(vocab)}
    p_term = np.concatenate([np.array([gid[t] for t in c[0]], dtype=np.int32)[c[1]]
                             for c in chunks])
    p_doc = np.concatenate([c[2] for c in chunks])
    p_tf = np.concatenate([c[3] for c in chunks])
    positions = np.concatenate([c[4] for c in chunks])

    doc_len = np.zeros(n_slots, dtype=np.int32)
    for c in chunks:
        doc_len[c[5]] = c[6]

    perm = np.argsort(p_term, kind="stable")
    tf = p_tf.astype(np.int64)
    old_start = np.cumsum(tf) - tf
    tf_new = tf[perm]
    new_start = np.cumsum(tf_new) - tf_new
    gather = np.repeat(old_start[perm] - new_start, tf_new)
    gather += np.arange(len(gather))
    positions = positions[gather]
    del gather, old_start, new_start

    p_term, p_doc, p_tf = p_term[perm], p_doc[perm], p_tf[perm]
    df = np.bincount(p_term, minlength=len(vocab))
    term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(df, out=term_ptr[1:])
    pos_cum = np.zeros(len(p_tf) + 1, dtype=np.int64)
    np.cumsum(tf_new, out=pos_cum[1:])
    term_pos_ptr = pos_cum[term_ptr]
    return vocab, term_ptr, p_doc, p_tf, term_pos_ptr, positions, doc_len


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default="catalog.sqlite")
    ap.add_argument("--out", default="index")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-docs", type=int, default=50_000)
    ap.add_argument("--limit", type=int, help="index only doc_id <= LIMIT (for testing)")
    ap.add_argument("--force", action="store_true", help="replace an existing index dir")
    args = ap.parse_args()

    if os.path.exists(args.out):
        if not args.force:
            sys.exit(f"{args.out} exists; use --force to replace it")
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    db = sqlite3.connect(f"file:{args.catalog}?mode=ro", uri=True)
    docs_root = db.execute("SELECT value FROM meta WHERE key = 'docs_root'").fetchone()[0]
    max_doc = db.execute("SELECT MAX(doc_id) FROM reviews").fetchone()[0]
    db.close()
    if args.limit:
        max_doc = min(max_doc, args.limit)
    tasks = [(os.path.abspath(args.catalog), docs_root, lo, min(lo + args.chunk_docs, max_doc + 1))
             for lo in range(1, max_doc + 1, args.chunk_docs)]
    log(f"{max_doc:,} docs in {len(tasks)} chunks, {args.workers} workers")

    chunks = []
    with Pool(args.workers) as pool:
        for i, c in enumerate(pool.imap(index_chunk, tasks), 1):
            chunks.append(c)
            if i % 10 == 0 or i == len(tasks):
                log(f"map: {i}/{len(tasks)} chunks")

    vocab, term_ptr, post_docs, post_tf, term_pos_ptr, positions, doc_len = merge(chunks, max_doc + 1)
    del chunks
    log(f"merge: {len(vocab):,} terms, {len(post_docs):,} postings, {len(positions):,} positions")

    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    for name, arr in [("term_ptr", term_ptr), ("post_docs", post_docs), ("post_tf", post_tf),
                      ("term_pos_ptr", term_pos_ptr), ("positions", positions),
                      ("doc_len", doc_len)]:
        np.save(os.path.join(args.out, f"{name}.npy"), arr)

    n_docs = int(max_doc)  # doc_ids are contiguous 1..max_doc
    meta = {
        "n_docs": n_docs,
        "avgdl": float(doc_len[1:].sum() / n_docs),
        "n_terms": len(vocab),
        "n_postings": int(len(post_docs)),
        "n_tokens": int(len(positions)),
        "max_doc_id": int(max_doc),
        "catalog": os.path.abspath(args.catalog),
        "docs_root": docs_root,
        "tokenizer": "app.search.tokenizer.analyze",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done: avgdl {meta['avgdl']:.2f}, written to {args.out}")


if __name__ == "__main__":
    main()
