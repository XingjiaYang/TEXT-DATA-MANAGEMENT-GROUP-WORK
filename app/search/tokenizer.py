"""Text analysis for the search engine, shared by indexing and querying.

Pipeline, chosen to mirror Elasticsearch's built-in `english` analyzer so that
our BM25 can be compared with ES on equal footing:

  normalise    curly quotes, ` and ´ -> ASCII; drop URLs; break on dot runs,
               zero-width chars and CJK/full-width punctuation ("good..but"),
               and missing spaces after a period ("carbonara.First");
               split words joined by - / \\ _ = (the-best-food)
  tokenise     NLTK TreebankWordTokenizer
  clean        lowercase, strip edge punctuation, drop tokens without letters/digits
               (punctuation, emoji); drop clitics 's 're ... and map n't -> not
  stop words   Lucene's English stop set minus "no"/"not" (negation is kept)
  stem         NLTK Porter stemmer, original algorithm (as ES `porter_stem`)

Known differences from ES `english`: we keep "no"/"not" (configure ES with the
same stop list), and Treebank splits "don't" into "do" + "not", whereas the ES
standard tokenizer keeps "don't" as one term.

Usage:
  uv run python -m app.search.tokenizer "I don't like the-best-food!!"
"""

import re
import sys
from functools import lru_cache

from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET minus the negations "no" and
# "not", which carry meaning in reviews ("not good"); the ES comparison index
# must use this same list in a custom analyzer.
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it of on or such that
the their then there these they this to was will with
""".split())

CLITICS = frozenset({"'s", "'re", "'ll", "'ve", "'d", "'m"})

QUOTES = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'",
                        "`": "'", "´": "'",
                        "“": '"', "”": '"', "‟": '"', "″": '"'})
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Word breaks Treebank misses: dot runs and ellipsis ("good..but"), zero-width
# characters, and CJK / full-width punctuation ("good，but"); full-width
# letters and digits are left alone.
BREAK_RE = re.compile(r"\.{2,}|[…​-‍⁠﻿　-〿"
                      r"！-／：-＠［-｀｛-･]")
# A period with 2+ letters before and a letter after is a missing space
# ("carbonara.First", "st.louis"); single-letter abbreviations (a.m., u.s.a)
# and decimals (3.5) are kept.
SENTENCE_JOIN_RE = re.compile(r"(?<=[^\W\d_]{2})\.(?=[^\W\d_])")
JOINER_RE = re.compile(r"(?<=\w)[-‐-―/\\_=]+(?=\w)")
EDGE_PUNCT_RE = re.compile(r"^[^\w']+|[^\w']+$|^'+|'+$")
ALNUM_RE = re.compile(r"[^\W_]")

_treebank = TreebankWordTokenizer()
_stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)


@lru_cache(maxsize=1 << 20)
def normalize_token(token):
    """Map one raw token to (term, is_stop), or None if it is not a word.

    Stop words are reported rather than dropped so that analyze() can leave a
    position gap for them, as Lucene does.
    """
    tok = token.lower()
    if tok in CLITICS:
        return None
    if tok == "n't":
        tok = "not"
    tok = EDGE_PUNCT_RE.sub("", tok)
    if tok.endswith("'s"):
        tok = tok[:-2]
    if not ALNUM_RE.search(tok):
        return None
    if tok in STOPWORDS:
        return tok, True
    return _stemmer.stem(tok, to_lowercase=False), False


def raw_tokens(text):
    """Treebank tokens after normalisation, before cleaning/stopping/stemming."""
    text = URL_RE.sub(" ", text.translate(QUOTES))
    text = SENTENCE_JOIN_RE.sub(" ", BREAK_RE.sub(" ", text))
    text = JOINER_RE.sub(" ", text)
    return _treebank.tokenize(text)


def analyze(text):
    """Analyse text into (terms, positions).

    Every word takes one position; stop words are removed but keep their slot,
    so "food was good" gives food@0, good@2 and phrase queries stay exact.
    """
    terms, positions = [], []
    pos = 0
    for tok in raw_tokens(text):
        res = normalize_token(tok)
        if res is None:
            continue
        term, is_stop = res
        if not is_stop:
            terms.append(term)
            positions.append(pos)
        pos += 1
    return terms, positions


def tokenize(text):
    """Analyse text into a list of index terms (order and duplicates kept)."""
    return analyze(text)[0]


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    terms, positions = analyze(text)
    print("raw:  ", raw_tokens(text))
    print("terms:", [f"{t}@{p}" for t, p in zip(terms, positions)])
