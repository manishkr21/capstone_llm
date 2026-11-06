"""Corpus audit (problems A-I in slides/data_audit_retrain.tex and the Phase-1 report).

    python scripts/11_audit_corpus.py data/corpus        # the first corpus
    python scripts/11_audit_corpus.py data_clean/corpus  # the cleaned corpus

Reports exact duplicates, documents sharing their first 300 characters, val documents copied
in train, val documents with >50% word-13-gram overlap against a 30k-document train sample,
and per-source noise counters (non-ASCII, Devanagari, U+FFFD/mojibake, one-letter-word OCR,
low letter share, repeated words, long dot/dash runs, URLs, short documents, OCR stutter, RTF).
"""
import collections
import hashlib
import json
import random
import re
import statistics as st
import sys


def shingles(text, n=13, step=7):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(0, max(1, len(w) - n), step)}


def main(path):
    docs = [json.loads(line) for line in open(f"{path}/train.jsonl")]
    val = [json.loads(line) for line in open(f"{path}/val.jsonl")]
    print("train docs", len(docs), "val docs", len(val))
    h = lambda t: hashlib.md5(t.encode()).hexdigest()
    hs = collections.Counter(h(d["text"]) for d in docs)
    dup = sum(c - 1 for c in hs.values() if c > 1)
    print(f"exact duplicate train docs {dup} ({dup / len(docs):.1%})")
    prefix = collections.Counter(d["text"][:300] for d in docs)
    print("extra docs sharing their first 300 chars", sum(c - 1 for c in prefix.values() if c > 1))
    print("val docs exactly in train", sum(1 for d in val if h(d["text"]) in hs))
    random.seed(0)
    train_sh = set()
    for d in random.sample(docs, min(30000, len(docs))):
        train_sh |= shingles(d["text"])
    sample = random.sample(val, min(500, len(val)))
    ov = [len(shingles(d["text"]) & train_sh) / max(1, len(shingles(d["text"]))) for d in sample]
    print(f"val docs with >50% 13-gram overlap vs a 30k-train sample: {sum(o > .5 for o in ov)}/{len(sample)}")

    deva, stut = re.compile(r"[\u0900-\u097F]"), re.compile(r"\b([A-Za-z]{3,})\b(?: \1\b){2}")
    rtf = re.compile(r"\\(par|pard|tab|rquote|lquote|f\d|fs\d+|qj|sl\d+)\b")
    per = collections.defaultdict(collections.Counter)
    lens = collections.defaultdict(list)
    for d in docs:
        t, s, c = d["text"], d["source"], per[d["source"]]
        c["n"] += 1
        n, w = len(t), t.split()
        lens[s].append(len(w))
        if sum(ord(ch) > 127 for ch in t) / n > 0.2:
            c[">20% non-ascii"] += 1
        if deva.search(t):
            c["has devanagari"] += 1
        if "\ufffd" in t or "Ã" in t or "â€" in t:
            c["mojibake/U+FFFD"] += 1
        if sum(len(x) == 1 for x in w) / len(w) > 0.3:
            c[">30% 1-char words"] += 1
        if sum(ch.isalpha() for ch in t) / n < 0.6:
            c["<60% letters"] += 1
        if re.search(r"(\b\S+\b)( \1){9,}", t[:20000]):
            c["word repeated 10x"] += 1
        if re.search(r"[-_=.*]{20,}", t):
            c["long ---/... runs"] += 1
        if re.search(r"https?://|www\.", t):
            c["urls"] += 1
        if len(w) < 200:
            c["<200 words"] += 1
        if len(stut.findall(t)) >= 5:
            c["OCR stutter >=5"] += 1
            c["stutter words"] += len(w)
        if len(rtf.findall(t)) >= 5:
            c["RTF >=5"] += 1
    for s, c in per.items():
        print(s, dict(c))
    for s, l in lens.items():
        print(s, "median words", st.median(l), "p10", sorted(l)[len(l) // 10])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/corpus")
