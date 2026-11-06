"""Share of each Phase-3 document's word 13-grams (1 in 8 sampled) found in each Phase-1 training corpus."""
import json, sys, multiprocessing as mp
from pathlib import Path
import numpy as np
P1 = Path(__file__).resolve().parents[2] / "capstone_llm"
sys.path.insert(0, str(P1))
from legal_lm.data.clean import _bounded_imap, _ngrams_work
K, MOD = 13, 8
def ref(path, pool):
    def texts():
        with open(path) as f:
            for ln in f:
                yield (json.loads(ln)["text"], K, MOD)
    return np.unique(np.concatenate(list(_bounded_imap(pool, _ngrams_work, texts(), 4096, 64))))
D = P1 / "data/hf/Phase_3_RR"
docs = {}
for f in ["Train/CL_train.json", "Train/CL_dev.json", "Train/IT_train.json", "Train/IT_dev.json"]:
    for d in json.load(open(D / f)):
        docs[f.split("/")[1][:2] + "-train:" + d["id"]] = " ".join(d["text"])
for f in ["Test/CL_test.json", "Test/IT_test.json"]:
    for k, v in json.load(open(D / f)).items():
        docs[f.split("/")[1][:2] + "-test:" + k] = " ".join(v)
with mp.get_context("fork").Pool(12) as pool:
    g = pool.map(_ngrams_work, [(t, K, MOD) for t in docs.values()])
    for name, path in [("first corpus", P1 / "data/corpus/train.jsonl"), ("clean corpus", P1 / "data_clean/corpus/train.jsonl")]:
        r = ref(path, pool)
        ov = np.array([float((r[np.minimum(np.searchsorted(r, x), len(r) - 1)] == x).mean()) if len(x) else 0 for x in g])
        for grp in ["CL-train", "IT-train", "CL-test", "IT-test"]:
            m = np.array([k.startswith(grp) for k in docs])
            print(f"{name:13s} {grp:9s} n={m.sum():2d} mean overlap {ov[m].mean():.3f}  >50%: {(ov[m] > .5).sum()}  >90%: {(ov[m] > .9).sum()}", flush=True)
