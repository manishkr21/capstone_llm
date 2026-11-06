"""Local proxy of the Codabench scoring: BERTScore F1 (primary), ROUGE-1 F1 and ROUGE-L F1.

The competition does not say which encoder its "legal BERTScore" uses; locally we use the bert_score
default for English (roberta-large), which is an evaluation tool only (never used to train or decode).
Both truncate texts at 512 encoder tokens, so only roughly the first 400 words of a summary are scored by
BERTScore, while ROUGE sees the whole text.
"""
from __future__ import annotations

import numpy as np


def score(hyps, refs, bertscore=True, device=None):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r = [sc.score(ref, hyp) for hyp, ref in zip(hyps, refs)]
    out = {"rouge1_f1": float(np.mean([x["rouge1"].fmeasure for x in r])),
           "rougeL_f1": float(np.mean([x["rougeL"].fmeasure for x in r])),
           "words": float(np.mean([len(h.split()) for h in hyps])),
           "distinct_trigram": distinct(hyps, 3)}
    if bertscore:
        import transformers
        transformers.logging.set_verbosity_error()
        from bert_score import score as bs
        P, R, F = bs(hyps, refs, lang="en", batch_size=32, verbose=False, device=device)
        out.update(bertscore_f1=float(F.mean()), bertscore_p=float(P.mean()), bertscore_r=float(R.mean()))
    return out


def distinct(hyps, n):
    """Mean share of distinct word n-grams per summary (1 = no repetition)."""
    vals = []
    for h in hyps:
        w = h.split()
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        vals.append(len(set(grams)) / max(1, len(grams)))
    return float(np.mean(vals))
