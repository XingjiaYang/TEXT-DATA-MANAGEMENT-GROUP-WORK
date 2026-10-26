"""Extract Yelp reviews into one Markdown file per review, plus a SQLite catalog.

Pipeline (both passes run in parallel over byte-range chunks of review.json):
  pass 1  clean each review text and hash it -> find exact duplicates globally
          (whitespace-normalised text; keep the earliest review, ties by file order)
  pass 2  write docs/<business_id>_<name>/<review_id>.md with YAML front matter,
          and collect catalog rows; the parent process inserts them into SQLite

The Markdown body holds the review text only; metadata lives in the front
matter and in the catalog, so indexing and document length use the text alone.

Usage:
  python3 extract_reviews.py --reviews data/yelp/yelp_academic_dataset_review.json \
      --business data/yelp/yelp_academic_dataset_business.json \
      --out docs --db catalog.sqlite
"""

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SLUG_RE = re.compile(r"\W+")
MAX_SLUG_BYTES = 150  # keeps <business_id>_<slug> well under the 255-byte filename limit

# Set in each worker by init_worker().
_BIZ = None      # business_id -> (dirname, front-matter lines for the business)
_DROPPED = None  # source offset -> kept offset (-1 when dropped for being empty)
_DOCS = None     # docs root directory


def clean_text(text):
    """Unescape HTML entities, normalise Unicode and newlines, drop control chars."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return CTRL_RE.sub("", text).strip()


def dedup_key(text):
    """64-bit hash of the whitespace-normalised text."""
    norm = " ".join(text.split()).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(norm, digest_size=8).digest(), "little")


def date_key(date):
    """'2018-07-07 22:09:11' -> 20180707220911, sortable as an integer."""
    return int(re.sub(r"\D", "", date))


def slugify(name):
    s = SLUG_RE.sub("_", unicodedata.normalize("NFKC", name or "")).strip("_")
    s = s.encode("utf-8")[:MAX_SLUG_BYTES].decode("utf-8", "ignore").rstrip("_")
    return s or "unnamed"


def yq(value):
    """YAML scalar: JSON strings/numbers/lists are valid YAML flow values."""
    return json.dumps(value, ensure_ascii=False)


def make_chunks(path, size):
    """Split a JSON-lines file into (start, end) byte ranges aligned to line starts."""
    total = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        while bounds[-1] < total:
            nxt = bounds[-1] + size
            if nxt >= total:
                bounds.append(total)
                break
            f.seek(nxt)
            f.readline()
            bounds.append(f.tell())
    return list(zip(bounds[:-1], bounds[1:]))


def iter_lines(path, start, end):
    """Yield (byte offset, raw line) for lines starting in [start, end)."""
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < end:
            line = f.readline()
            if not line:
                break
            yield pos, line
            pos += len(line)


def load_businesses(path):
    biz = {}
    with open(path, "rb") as f:
        for line in f:
            b = json.loads(line)
            cats = [c.strip() for c in (b.get("categories") or "").split(",") if c.strip()]
            biz[b["business_id"]] = {
                "business_id": b["business_id"],
                "name": b.get("name") or "",
                "city": b.get("city"),
                "state": b.get("state"),
                "categories": cats,
                "stars": b.get("stars"),
                "review_count": b.get("review_count"),
                "is_open": b.get("is_open"),
                "dirname": f"{b['business_id']}_{slugify(b.get('name'))}",
            }
    return biz


def business_front_matter(b):
    return (
        f"business_id: {yq(b['business_id'])}\n"
        f"business_name: {yq(b['name'])}\n"
        f"city: {yq(b['city'])}\n"
        f"state: {yq(b['state'])}\n"
        f"categories: {yq(b['categories'])}\n"
        f"business_stars: {yq(b['stars'])}\n"
        f"business_review_count: {yq(b['review_count'])}\n"
    )


# ---------------------------------------------------------------- pass 1

def hash_chunk(args):
    path, start, end = args
    hashes, dates, offsets, empty = [], [], [], []
    for off, line in iter_lines(path, start, end):
        r = json.loads(line)
        text = clean_text(r["text"])
        if not text:
            empty.append(off)
            continue
        hashes.append(dedup_key(text))
        dates.append(date_key(r["date"]))
        offsets.append(off)
    return (np.array(hashes, dtype=np.uint64), np.array(dates, dtype=np.int64),
            np.array(offsets, dtype=np.int64), empty)


def find_duplicates(results):
    """Return {dropped offset: kept offset}; the earliest review of each group is kept."""
    h = np.concatenate([r[0] for r in results])
    d = np.concatenate([r[1] for r in results])
    o = np.concatenate([r[2] for r in results])
    order = np.lexsort((o, d, h))  # by hash, then date, then file position
    h, o = h[order], o[order]
    first = np.ones(len(h), dtype=bool)
    first[1:] = h[1:] != h[:-1]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(h)), 0))
    dup = np.nonzero(~first)[0]
    dropped = dict(zip(o[dup].tolist(), o[group_start[dup]].tolist()))
    for r in results:
        dropped.update((off, -1) for off in r[3])
    return dropped


# ---------------------------------------------------------------- pass 2

def init_worker(biz_fm, dropped, docs_root):
    global _BIZ, _DROPPED, _DOCS
    _BIZ, _DROPPED, _DOCS = biz_fm, dropped, docs_root


def write_chunk(args):
    path, start, end = args
    rows, dropped, made = [], [], set()
    for off, line in iter_lines(path, start, end):
        r = json.loads(line)
        if off in _DROPPED:
            dropped.append((off, r["review_id"], r["business_id"]))
            continue
        text = clean_text(r["text"])
        dirname, biz_fm = _BIZ[r["business_id"]]
        if dirname not in made:
            os.makedirs(os.path.join(_DOCS, dirname), exist_ok=True)
            made.add(dirname)
        relpath = f"{dirname}/{r['review_id']}.md"
        stars = int(r["stars"])
        with open(os.path.join(_DOCS, relpath), "w", encoding="utf-8", newline="\n") as f:
            f.write(
                "---\n"
                f"review_id: {yq(r['review_id'])}\n"
                f"{biz_fm}"
                f"user_id: {yq(r['user_id'])}\n"
                f"stars: {stars}\n"
                f"date: {yq(r['date'])}\n"
                f"useful: {r['useful']}\n"
                f"funny: {r['funny']}\n"
                f"cool: {r['cool']}\n"
                "---\n\n"
                f"{text}\n"
            )
        rows.append((r["review_id"], r["business_id"], r["user_id"], stars, r["date"],
                     int(r["date"][:4]), r["useful"], r["funny"], r["cool"],
                     len(text), relpath, off))
    return rows, dropped


# ---------------------------------------------------------------- catalog

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE businesses (
    business_id TEXT PRIMARY KEY, name TEXT, city TEXT, state TEXT,
    categories TEXT, stars REAL, review_count INTEGER, is_open INTEGER, dirname TEXT
);
CREATE TABLE reviews (
    doc_id INTEGER PRIMARY KEY,          -- 1..N in source-file order; used by the index
    review_id TEXT UNIQUE NOT NULL, business_id TEXT NOT NULL, user_id TEXT,
    stars INTEGER, date TEXT, year INTEGER, useful INTEGER, funny INTEGER, cool INTEGER,
    n_chars INTEGER,                     -- cleaned text length in characters
    path TEXT NOT NULL,                  -- relative to the docs root (see meta)
    src_offset INTEGER NOT NULL          -- byte offset of the line in review.json
);
CREATE TABLE dropped (
    review_id TEXT PRIMARY KEY, business_id TEXT, reason TEXT,
    kept_review_id TEXT, same_business INTEGER
);
"""

INDEXES = """
CREATE INDEX idx_reviews_business ON reviews(business_id);
CREATE INDEX idx_reviews_year ON reviews(year);
CREATE INDEX idx_reviews_stars ON reviews(stars);
CREATE UNIQUE INDEX idx_reviews_offset ON reviews(src_offset);
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--business", required=True)
    ap.add_argument("--out", default="docs", help="docs root directory")
    ap.add_argument("--db", default="catalog.sqlite")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                    help="write into a non-empty docs dir and replace an existing db")
    args = ap.parse_args()

    if not args.force and (os.path.exists(args.db) or
                           (os.path.isdir(args.out) and os.listdir(args.out))):
        sys.exit(f"{args.out} is not empty or {args.db} exists; use --force to overwrite")
    if os.path.exists(args.db):
        os.remove(args.db)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    biz = load_businesses(args.business)
    log(f"loaded {len(biz):,} businesses")
    chunks = [(args.reviews, s, e) for s, e in make_chunks(args.reviews, args.chunk_mb << 20)]
    log(f"{len(chunks)} chunks, {args.workers} workers")

    with Pool(args.workers) as pool:
        hashed = pool.map(hash_chunk, chunks)
    n_total = sum(len(r[2]) + len(r[3]) for r in hashed)
    dropped = find_duplicates(hashed)
    del hashed
    n_empty = sum(1 for v in dropped.values() if v == -1)
    log(f"pass 1: {n_total:,} reviews, {len(dropped) - n_empty:,} duplicates, {n_empty:,} empty")

    db = sqlite3.connect(args.db)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + SCHEMA)
    db.executemany(
        "INSERT INTO businesses VALUES (?,?,?,?,?,?,?,?,?)",
        [(b["business_id"], b["name"], b["city"], b["state"], ", ".join(b["categories"]),
          b["stars"], b["review_count"], b["is_open"], b["dirname"]) for b in biz.values()])

    biz_fm = {k: (b["dirname"], business_front_matter(b)) for k, b in biz.items()}
    dropped_rows = []
    n_written = 0
    with Pool(args.workers, initializer=init_worker,
              initargs=(biz_fm, dropped, os.path.abspath(args.out))) as pool:
        # imap keeps chunk order, so doc_id follows source-file order.
        for i, (rows, drop) in enumerate(pool.imap(write_chunk, chunks), 1):
            db.executemany(
                "INSERT INTO reviews (review_id, business_id, user_id, stars, date, year,"
                " useful, funny, cool, n_chars, path, src_offset)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            dropped_rows.extend(drop)
            n_written += len(rows)
            if i % 10 == 0 or i == len(chunks):
                log(f"pass 2: {i}/{len(chunks)} chunks, {n_written:,} files written")
    db.executescript(INDEXES)

    for off, rid, bid in dropped_rows:
        kept = dropped[off]
        if kept == -1:
            db.execute("INSERT INTO dropped VALUES (?,?,?,?,?)", (rid, bid, "empty", None, None))
            continue
        kept_rid, kept_bid = db.execute(
            "SELECT review_id, business_id FROM reviews WHERE src_offset = ?", (kept,)).fetchone()
        db.execute("INSERT INTO dropped VALUES (?,?,?,?,?)",
                   (rid, bid, "duplicate", kept_rid, int(bid == kept_bid)))

    meta = {
        "source_reviews": os.path.abspath(args.reviews),
        "source_business": os.path.abspath(args.business),
        "docs_root": os.path.abspath(args.out),
        "n_source_reviews": n_total,
        "n_docs": n_written,
        "n_dropped_duplicate": len(dropped) - n_empty,
        "n_dropped_empty": n_empty,
        "cleaning": "html.unescape; NFC; CRLF->LF; strip control chars; strip",
        "dedup": "exact match on whitespace-normalised cleaned text; keep earliest date, then file order",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    db.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    db.commit()
    db.close()
    log(f"done: {n_written:,} docs in {args.out}, catalog {args.db}")


if __name__ == "__main__":
    main()
