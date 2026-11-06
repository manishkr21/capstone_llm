"""Lexical retrieval (BM25 / TF-IDF) over word n-grams: the non-neural baseline and first-stage ranker."""
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

TAG = re.compile(r"<(NAME|ORG|CITATION)>")


def clean(t):
    return TAG.sub(" ", t).lower()


def bm25(queries, docs, ngram=(1, 1), k1=1.2, b=0.75, min_df=2, max_df=0.5, q_binary=False):
    """BM25 score matrix (n_queries, n_docs); vocabulary/IDF from the docs (the candidate pool)."""
    cv = CountVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df, token_pattern=r"(?u)\b[a-z][a-z0-9]+\b",
                         preprocessor=clean, dtype=np.float32)
    D = cv.fit_transform(docs).tocsr()
    Q = cv.transform(queries).tocsr()
    n = D.shape[0]
    df = np.bincount(D.indices, minlength=D.shape[1])
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(D.sum(1)).ravel()
    D = D.tocoo()
    denom = D.data + k1 * (1 - b + b * dl[D.row] / dl.mean())
    W = sp.csr_matrix((D.data * (k1 + 1) / denom * idf[D.col], (D.row, D.col)), shape=D.shape)
    if q_binary:
        Q.data[:] = 1
    return np.asarray((Q @ W.T).todense())


def windows(text, w):
    """Citation contexts: +-w words around every <CITATION> marker (the whole text if there is none)."""
    ws = text.split()
    idx = [i for i, x in enumerate(ws) if "<CITATION>" in x]
    return [" ".join(ws[max(0, i - w):i + w + 1]) for i in idx] or [text]


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def per_citation(qtexts, docs, w=60, agg="max", **kw):
    """Every citation window is its own BM25 query; per-window z-scores aggregated per query."""
    W, owner = [], []
    for k, t in enumerate(qtexts):
        ws = windows(t, w)
        W += ws
        owner += [k] * len(ws)
    SW, owner = zscore(bm25(W, docs, **kw)), np.array(owner)
    A = np.zeros((len(qtexts), len(docs)), dtype=np.float32)
    for k in range(len(qtexts)):
        M = SW[owner == k]
        A[k] = M.max(0) if agg == "max" else np.log(np.exp(M).sum(0))
    return A


def per_citation_wincol(qids, qtexts, docs, cids, w=60, gammas=(1.0, 2.0), **kw):
    """Per-citation BM25 with window-level pool normalisation: every window's scores are z-scored over the
    cases (row), then each case's column over all windows of the split (a case is usually cited from only
    one or two windows); per query, the max over its windows of row + gamma * column. A query's own case is
    masked. Returns {gamma: (n_q, n_c)}."""
    W, owner = [], []
    for k, t in enumerate(qtexts):
        ws = windows(t, w)
        W += ws
        owner += [k] * len(ws)
    owner = np.array(owner)
    SW = bm25(W, docs, **kw).astype(np.float64)
    col = {c: j for j, c in enumerate(cids)}
    lo = SW.min()
    for k, q in enumerate(qids):
        if q in col:
            SW[owner == k, col[q]] = lo
    R = zscore(SW)
    C = zscore(R.T).T
    out = {}
    for g in gammas:
        M = R + g * C
        out[g] = np.stack([M[owner == k].max(0) for k in range(len(qids))]).astype(np.float32)
    return out
