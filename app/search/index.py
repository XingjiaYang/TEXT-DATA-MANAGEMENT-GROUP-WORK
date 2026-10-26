"""Read-only access to the positional inverted index built by app.search.indexer.

Arrays are memory-mapped, so loading is fast and only touched pages are read.
"""

import json
import os

import numpy as np

POS_BITS = 16  # positions are uint16, so (doc << 16) | position is a unique int64 key


class InvertedIndex:
    def __init__(self, index_dir):
        with open(os.path.join(index_dir, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(index_dir, "vocab.txt"), encoding="utf-8") as f:
            self.terms = f.read().split("\n")[:-1]
        self.term_ids = {t: i for i, t in enumerate(self.terms)}
        load = lambda name: np.load(os.path.join(index_dir, f"{name}.npy"), mmap_mode="r")
        self.term_ptr = load("term_ptr")
        self.post_docs = load("post_docs")
        self.post_tf = load("post_tf")
        self.term_pos_ptr = load("term_pos_ptr")
        self.positions = load("positions")
        self.doc_len = load("doc_len")
        self.n_docs = self.meta["n_docs"]
        self.avgdl = self.meta["avgdl"]

    def term_id(self, term):
        return self.term_ids.get(term)

    def df(self, tid):
        return int(self.term_ptr[tid + 1] - self.term_ptr[tid])

    def postings(self, tid):
        """(doc_ids, tfs) of a term, doc_ids ascending."""
        s, e = self.term_ptr[tid], self.term_ptr[tid + 1]
        return self.post_docs[s:e], self.post_tf[s:e]

    def term_positions(self, tid):
        """(doc per position, position) for every occurrence of a term."""
        docs, tfs = self.postings(tid)
        s = self.term_pos_ptr[tid]
        pos = self.positions[s:s + int(tfs.sum(dtype=np.int64))]
        return np.repeat(docs, tfs), pos

    def doc_positions(self, tid, doc_id):
        """Positions of a term in one document (empty if absent)."""
        docs, tfs = self.postings(tid)
        i = np.searchsorted(docs, doc_id)
        if i == len(docs) or docs[i] != doc_id:
            return np.empty(0, dtype=np.uint16)
        s = self.term_pos_ptr[tid] + int(tfs[:i].sum(dtype=np.int64))
        return self.positions[s:s + int(tfs[i])]

    def phrase_match(self, terms, positions):
        """Docs containing the analysed phrase, with how often it occurs.

        terms/positions come from tokenizer.analyze() on the query, so stop-word
        gaps must line up exactly. Each occurrence is keyed as (doc, start) where
        start = token position - offset in the phrase; a phrase occurrence is a
        key present for every query term.
        """
        tids = [self.term_id(t) for t in terms]
        if not tids or None in tids:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
        # Restrict to docs containing all terms, rarest term first.
        order = sorted(range(len(tids)), key=lambda i: self.df(tids[i]))
        cand = np.asarray(self.postings(tids[order[0]])[0])
        for i in order[1:]:
            cand = np.intersect1d(cand, self.postings(tids[i])[0], assume_unique=True)
        keys = None
        for i in order:
            docs, pos = self.term_positions(tids[i])
            mask = np.isin(docs, cand, assume_unique=False)
            start = pos[mask].astype(np.int64) - (positions[i] - positions[0])
            ok = start >= 0
            k = (docs[mask][ok].astype(np.int64) << POS_BITS) | start[ok]
            keys = k if keys is None else np.intersect1d(keys, k, assume_unique=True)
            if not len(keys):
                break
        docs, counts = np.unique(keys >> POS_BITS, return_counts=True)
        return docs.astype(np.int32), counts
