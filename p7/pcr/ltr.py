"""Learning-to-rank over the top candidates of the per-citation BM25 ranking (LightGBM lambdarank).

Features per (query, case): z-scores and ranks of every lexical score matrix, any neural score matrices,
the same scores normalised per case over the split's queries (a pool is essentially the union of its queries'
gold cases, so "which query does this case belong to" is informative), whether the case is itself a query of the
split, date relation, case length, number of citation windows in the query.
"""
import numpy as np

from .lexical import zscore

LEX_KEYS = ("ctx", "pc40_max", "pc40_lse", "pc60_max", "pc60_lse", "pc100_max", "pc100_lse",
            "wc60_g1", "wc60_g2", "wc100_g1", "wc100_g2")


def features(lx, neural, ntok, nwin, depth=100, first="pc60_max", first_col=True):
    """lx: lexical cache dict; neural: list of (n_q, n_c) matrices aligned with it; ntok: case lengths;
    nwin: windows per query. The candidates of a query: its top-``depth`` cases by the ``first`` lexical score
    (row z-score + column z-score when ``first_col``). Returns X (rows), group sizes, (qi, cj) index pairs."""
    qids, cids = list(lx["qids"]), list(lx["cids"])
    col = {c: j for j, c in enumerate(cids)}
    own = [(i, col[q]) for i, q in enumerate(qids) if q in col]
    mats = [row_col(lx[k], own) for k in LEX_KEYS] + [row_col(S, own) for S in neural]
    mats = [m for pair in mats for m in pair]
    ranks = [np.argsort(np.argsort(-M, 1), 1) for M in mats[0::2]] + \
            [np.argsort(np.argsort(-M, 0), 0) for M in mats[1::2]]
    isq = np.array([c in set(qids) for c in cids], dtype=np.float32)
    dy = lx["qyear"][:, None].astype(float) - lx["cyear"][None, :].astype(float)
    X, groups, pairs = [], [], []
    R, C = row_col(lx[first], own)
    F = R + C if first_col else R
    for i, q in enumerate(qids):
        s = F[i].copy()
        if q in col:
            s[col[q]] = -np.inf
        top = np.argsort(-s)[:depth]
        for j in top:
            d = dy[i, j] if lx["cyear"][j] and lx["qyear"][i] < 9999 else np.nan
            X.append([M[i, j] for M in mats] + [np.log1p(r[i, j]) for r in ranks] +
                     [d, float(d < 0) if d == d else np.nan, np.log(ntok[j]), np.log(nwin[i]), isq[j]])
            pairs.append((i, j))
        groups.append(len(top))
    return np.array(X, dtype=np.float32), np.array(groups), pairs


def row_col(S, own):
    """Row z-score (over cases) and column z-score of that (over queries), a query's own case masked."""
    S = S.astype(np.float64).copy()
    lo = S.min()
    for i, j in own:
        S[i, j] = lo
    R = zscore(S)
    return R, zscore(R.T).T


def labels(lx, queries, pairs):
    qids, cids = list(lx["qids"]), list(lx["cids"])
    return np.array([cids[j] in queries[qids[i]]["rel"] for i, j in pairs], dtype=np.int32)


def rank_from(pred, lx, pairs, k=7):
    qids, cids = list(lx["qids"]), list(lx["cids"])
    best = {}
    for p, (i, j) in zip(pred, pairs):
        best.setdefault(i, []).append((p, cids[j]))
    return {qids[i]: [c for _, c in sorted(v, key=lambda x: -x[0])[:k]] for i, v in best.items()}
