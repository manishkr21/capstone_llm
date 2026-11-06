"""Figures, tables and number macros for the Phase-7 (prior case retrieval) report.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p7/runs/<run>/{summary.json, metrics.jsonl, config.yaml}, p7/runs/stack/*.{json,npz}, the lexical
score cache (p7/cache/lex_{split}.npz), the raw Phase-7 files and p7/submissions/.  The lexical ladder and the
zero-shot (untrained) Phase-1 retrieval scores are computed here once and cached in report/cache.json.
No test relevance label exists or is read.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
P7 = HERE.parent
sys.path.insert(0, str(P7))
from pcr.data import K, load, micro_f1, topk  # noqa: E402
from pcr.lexical import bm25, windows, zscore  # noqa: E402
from pcr.ltr import rank_from, row_col  # noqa: E402

FIGS, TABS = HERE / "figs", HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)
CACHE = HERE / "cache.json"

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
loadj = lambda p: json.load(open(p))
macros = {}
DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
MAIN = ["E1_a3_lorma", "E2_a1_full", "E3_a4_partial", "E4_a2_qlora", "E5_a3_ia3", "E6_a1_prefix", "E7_a4_prompt",
        "K1_kd_a1_4l", "K0_a1_4l_nokd"]
EXTRA = ["E2b_a1_full_seed1", "E2c_a1_full_seed2", "E2d_a1_full_hard", "E1b_a3_lorma_seed1", "E3b_a4_partial_seed1"]
NICE = {"full": "full fine-tuning", "lorma": "LoRMA", "partial": "partial (top 4 blocks)", "qlora": "QLoRA (NF4)",
        "ia3": "IA$^3$", "prefix": "prefix tuning", "prompt": "prompt tuning"}


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def summ(r):
    return loadj(P7 / "runs" / r / "summary.json")


def cfg(r):
    return yaml.safe_load(open(P7 / "runs" / r / "config.yaml"))


def arch(r):
    return cfg(r)["model"]["phase1_ckpt"].split("/")[1].split("_")[0]


def method(r):
    c = cfg(r)
    if c.get("distill"):
        return f"KD student, {len(c['model']['keep_layers'])} blocks"
    if c["model"].get("keep_layers"):
        return f"{len(c['model']['keep_layers'])} blocks, no teacher"
    m = NICE[c["peft"]["method"]]
    if c["train"].get("n_hard", 3) != 3:
        m += ", 6 hard negatives"
    return m


def short(r):
    return r.split("_")[0]


def lex(split):
    return dict(np.load(P7 / "cache" / f"lex_{split}.npz"))


SPL = {s: load(s) for s in ("train", "dev", "test")}


# ------------------------------------------------------------------ data
def data():
    rows = []
    for s in ("train", "dev", "test"):
        q, pool = SPL[s]
        qw = [len(v["text"].split()) for v in q.values()]
        cw = [len(t.split()) for t in pool.values()]
        nc = [v["text"].count("<CITATION>") for v in q.values()]
        if s != "test":
            r = np.array([len(v["rel"]) for v in q.values()])
            rel = f"{r.mean():.2f} & {int(np.median(r))} & {r.min()}--{r.max()}"
            gold = {c for v in q.values() for c in v["rel"]}
            nonq = [c for c in pool if c not in q]
            M(f"nonqgold{s}", f"{100 * np.mean([c in gold for c in nonq]):.1f}")
            M(f"qgold{s}", f"{100 * np.mean([c in gold for c in q]):.1f}")
            M(f"corrcit{s}", f"{np.corrcoef(nc, r)[0, 1]:.2f}")
            hit = np.minimum(r, K).sum()
            P, R = hit / (K * len(r)), hit / r.sum()
            M(f"oracle{s}", f"{2 * P * R / (P + R):.3f}")
        else:
            rel = "-- & -- & --"
        rows.append(f"{s} & {len(q):,} & {len(pool):,} & {sum(k in pool for k in q)} & {rel} & {np.median(nc):.0f} & "
                    f"{np.median(qw):,.0f} & {np.median(cw):,.0f} \\\\")
        M(f"nq{s}", f"{len(q):,}")
        M(f"npool{s}", f"{len(pool):,}")
    (TABS / "data.tex").write_text("\n".join(rows) + "\n")
    tr, dv, te = (set(SPL[s][1]) for s in ("train", "dev", "test"))
    M("testpooltrain", f"{len(te & tr):,}")
    lx = lex("dev")
    viol = [lx["cyear"][list(lx["cids"]).index(c)] > lx["qyear"][i] for i, q in enumerate(lx["qids"])
            for c in SPL["dev"][0][q]["rel"] if lx["cyear"][list(lx["cids"]).index(c)] and lx["qyear"][i] < 9999]
    M("relnewer", f"{100 * np.mean(viol):.0f}")


# ------------------------------------------------------------------ cached computations
def f1(S, lx, q):
    return micro_f1(topk(S, list(lx["qids"]), list(lx["cids"])), q)["f1"]


def own(lx):
    col = {c: j for j, c in enumerate(lx["cids"])}
    return [(i, col[x]) for i, x in enumerate(lx["qids"]) if x in col]


def cached():
    if CACHE.exists():
        return loadj(CACHE)
    out = {"ladder": {}, "zeroshot": {}}
    for s in ("dev", "train"):
        q, pool = SPL[s]
        lx = lex(s)
        qi, ci = list(lx["qids"]), list(lx["cids"])
        viol = (lx["cyear"][None] > lx["qyear"][:, None]).astype(float)
        full = bm25([q[i]["text"] for i in qi], [pool[c] for c in ci], ngram=(1, 3))
        o = own(lx)
        steps = [("BM25 on the whole query (word 1--3-grams)", full),
                 ("BM25 on the citation contexts ($\\pm$80 words)", lx["ctx"]),
                 ("per-citation BM25 ($\\pm$60 words), max over windows", lx["pc60_max"]),
                 ("\\quad + penalty for cases dated after the query", zscore(lx["pc60_max"]) - 2 * viol),
                 ("\\quad + case-wise (column) normalisation", sum(row_col(lx["pc60_max"], o)) - 2 * viol),
                 ("window-level normalisation (wc60, $\\gamma=2$) + column + date", sum(row_col(lx["wc60_g2"], o)) - 2 * viol)]
        out["ladder"][s] = [(name, f1(S, lx, q)) for name, S in steps]
        print(s, out["ladder"][s], flush=True)
    import torch
    _s = __import__("importlib.util").util.spec_from_file_location("p7_train", P7 / "train.py")
    t = __import__("importlib.util").util.module_from_spec(_s)
    _s.loader.exec_module(t)
    from legal_lm.config import load_config
    from pcr import tok as T
    from pcr.model import build_model
    toks = T.get()
    for a in ("A1_gpt", "A2_encdec", "A3_prefixlm", "A4_efficient"):
        c = load_config([P7 / "configs" / "base.yaml"], [f"model.phase1_ckpt=runs/{a}_clean_1B/best.pt", "peft.method=full"])
        model, _, _ = build_model(c)
        model.head = torch.nn.Identity()
        model.cuda()
        S, qi, ci = t.score_split(model, "dev", toks, c, torch.device("cuda"))
        out["zeroshot"][a.split("_")[0]] = micro_f1(topk(S, qi, ci), SPL["dev"][0])["f1"]
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def ladder(cache):
    rows = [f"{name} & {d:.4f} & {t:.4f} \\\\" for (name, d), (_, t) in zip(cache["ladder"]["dev"], cache["ladder"]["train"])]
    (TABS / "ladder.tex").write_text("\n".join(rows) + "\n")
    for k, (_, v) in enumerate(cache["ladder"]["dev"]):
        M(f"lad{k}", f"{v:.3f}")
    rows = [f"{a} & {v:.4f} \\\\" for a, v in sorted(cache["zeroshot"].items())]
    (TABS / "zeroshot.tex").write_text("\n".join(rows) + "\n")
    M("zsmin", f"{min(cache['zeroshot'].values()):.3f}")
    M("zsmax", f"{max(cache['zeroshot'].values()):.3f}")

    # first-stage recall@k on dev
    q, _ = SPL["dev"]
    lx = lex("dev")
    o = own(lx)
    viol = (lx["cyear"][None] > lx["qyear"][:, None]).astype(float)
    qi, ci = list(lx["qids"]), np.array(lx["cids"])
    ks = [7, 10, 20, 30, 50, 100, 200]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for n, (lab, S) in enumerate((("per-citation BM25 (row)", zscore(lx["pc60_max"]) - 2 * viol),
                                  ("+ column normalisation", sum(row_col(lx["pc60_max"], o)) - 2 * viol),
                                  ("window-level (wc60)", sum(row_col(lx["wc60_g2"], o)) - 2 * viol))):
        S = S.copy()
        for i, j in o:
            S[i, j] = -1e9
        order = np.argsort(-S, 1)
        rec = [np.sum([len(set(ci[order[i, :k]]) & q[x]["rel"]) for i, x in enumerate(qi)]) /
               sum(len(q[x]["rel"]) for x in qi) for k in ks]
        ax.plot(ks, rec, "o-", color=C[n], label=lab, ms=4)
        if n == 2:
            M("recallhundred", f"{100 * rec[ks.index(100)]:.0f}")
            M("recallseven", f"{100 * rec[0]:.0f}")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set(xlabel="k (cases kept per query)", ylabel="dev recall@k (gold cases)")
    ax.legend(fontsize=7.5)
    fig.savefig(FIGS / "recall.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ experiments
def results():
    rows = []
    for r in MAIN + ["\\midrule"] + EXTRA:
        if r.startswith("\\"):
            rows.append(r)
            continue
        s = summ(r)
        n, fu, p = s["dev_neural"], s["dev_fused"], s["params"]
        rows.append(f"{tex(short(r))} & {arch(r)} & {method(r)} & {p['total_params'] / 1e6:.1f}M & "
                    f"{p['trainable_params'] / 1e6:.2f}M & {p['trainable_pct']:.1f} & {n['f1']:.4f} & {n['p']:.3f} & "
                    f"{n['r']:.3f} & {fu['f1']:.4f} \\\\")
        M("nf" + short(r), f"{n['f1']:.3f}")
        M("ff" + short(r), f"{fu['f1']:.3f}")
    (TABS / "results.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for r in MAIN:
        s = summ(r)
        rows.append(f"{tex(short(r))} & {arch(r)} & {method(r)} & {s['params']['total_params'] / 1e6:.1f}M & "
                    f"{s['train_min']:.1f} & {s['dev_encode_sec']:.1f} & {s['peak_mem_gb']:.1f} & {s['ckpt_mb']:.1f} \\\\")
    (TABS / "efficiency.tex").write_text("\n".join(rows) + "\n")
    st, t = summ("K1_kd_a1_4l"), summ("E2_a1_full")
    M("kdparams", f"{st['params']['total_params'] / 1e6:.1f}M")
    M("teacherparams", f"{t['params']['total_params'] / 1e6:.1f}M")
    M("kdenc", f"{st['dev_encode_sec']:.1f}")
    M("teacherenc", f"{t['dev_encode_sec']:.1f}")

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for r in MAIN:
        s = summ(r)
        x, y = s["params"]["trainable_params"] / 1e6, s["dev_neural"]["f1"]
        ax.plot(x, y, "o", ms=6, color=C[1] if r.startswith("K") else C[0], markeredgecolor="white")
        ax.annotate(f"{short(r)} {arch(r)} {method(r)}".replace("$^3$", "3"), (x, y), xytext=(4, -3),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="dev micro-F1@7 (neural only)")
    fig.savefig(FIGS / "f1_vs_params.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for k, r in enumerate(MAIN[:7]):
        recs = [json.loads(l) for l in open(P7 / "runs" / r / "metrics.jsonl") if "dev_f1" in l]
        ax.plot([x["epoch"] for x in recs], [x["dev_f1"] for x in recs], "o-", ms=3, lw=1.2, color=C[k],
                label=f"{short(r)} {arch(r)} {method(r)}")
    ax.set(xlabel="epoch", ylabel="dev micro-F1@7 (neural only)")
    ax.legend(fontsize=6.5, ncol=2)
    fig.savefig(FIGS / "curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ stacks + cross-encoder
def stacks():
    names = [("STK0_lex", "lexical features only"), ("STK1_e2", "+ E2 out-of-fold scores as features"),
             ("STK2_lex_blend4", "lexical LTR + 4 bi-encoders blended (row z)"),
             ("STK4_rc_blend4", "row+column features and first stage, + 4 bi-encoders (row+column)"),
             ("STK5_all13_w075", "as STK4, 13 bi-encoders"),
             ("STK6_wc_all13", "+ window-level lexical features and first stage, 13 bi-encoders")]
    rows = []
    for n, desc in names:
        m = loadj(P7 / "runs" / "stack" / f"{n}.json")
        d = m["dev"]
        rows.append(f"{tex(n)} & {desc} & {len(m.get('blend', []))} & {m.get('w', 0) if m.get('blend') else '--'} & "
                    f"{d['f1']:.4f} & {d['p']:.3f} & {d['r']:.3f} \\\\")
        M("stk" + n.split("_")[0][3:], f"{d['f1']:.3f}")
    c = summ("C1_a1_cross")
    rows.append(f"C1 cross-encoder & STK4 top-30 re-ordered by z(STK4) + {c['w']} z(cross) & -- & -- & {c['dev_best']:.4f} & -- & -- \\\\")
    (TABS / "stacks.tex").write_text("\n".join(rows) + "\n")
    M("crossbest", f"{c['dev_best']:.3f}")
    M("crossbase", f"{c['dev_stack']:.3f}")
    recs = [json.loads(l) for l in open(P7 / "runs" / "C1_a1_cross" / "metrics.jsonl") if "dev_cross_only" in l]
    M("crossonly", f"{max(x['dev_cross_only'] for x in recs):.3f}")


# ------------------------------------------------------------------ errors of the final system on dev
def errors():
    q, pool = SPL["dev"]
    lx = lex("dev")
    d = np.load(P7 / "runs" / "stack" / "STK6_wc_all13_dev.npz")
    ranked = rank_from(d["pred"], lx, [tuple(x) for x in d["pairs"]])
    m = micro_f1(ranked, q)
    M("finalf", f"{m['f1']:.3f}")
    M("finalp", f"{m['p']:.3f}")
    M("finalr", f"{m['r']:.3f}")
    ng = np.array([len(q[x]["rel"]) for x in ranked])
    hits = np.array([len(set(ranked[x]) & q[x]["rel"]) for x in ranked])
    M("zerohit", int((hits == 0).sum()))
    bins = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 1000)]
    rows = []
    for lo, hi in bins:
        sel = (ng >= lo) & (ng <= hi)
        if not sel.any():
            continue
        P, R = hits[sel].sum() / (K * sel.sum()), hits[sel].sum() / ng[sel].sum()
        rows.append(f"{lo}--{hi if hi < 1000 else '+'} & {sel.sum()} & {hits[sel].mean():.2f} & {P:.3f} & {R:.3f} & "
                    f"{2 * P * R / max(1e-9, P + R):.3f} & {(hits[sel] == 0).sum()} \\\\".replace("--+", "+"))
    (TABS / "errors.tex").write_text("\n".join(rows) + "\n")
    pos = np.zeros(K)
    for x in ranked:
        for k, c in enumerate(ranked[x]):
            pos[k] += c in q[x]["rel"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axes[0].bar(range(1, K + 1), 100 * pos / len(ranked), color=C[0])
    axes[0].set(xlabel="rank", ylabel="% of queries with a gold case")
    axes[0].grid(axis="x", visible=False)
    axes[1].scatter(ng + np.random.default_rng(0).uniform(-0.2, 0.2, len(ng)), hits, s=10, color=C[1], edgecolor="white", linewidth=0.3)
    xs = np.arange(1, ng.max() + 1)
    axes[1].plot(xs, np.minimum(xs, K), color=INK2, ls="--", lw=0.8)  # the most a 7-item list can find
    axes[1].set_xscale("log")
    axes[1].set(xlabel="gold cases of the query (log)", ylabel="gold cases in the top 7")
    fig.tight_layout()
    fig.savefig(FIGS / "errors.pdf")
    plt.close(fig)


def submissions():
    zips = sorted(p.name for p in (P7 / "submissions").glob("*.zip"))
    (TABS / "submissions.tex").write_text("\n".join(f"\\code{{{tex(z)}}} \\\\" for z in zips) + "\n")
    M("nzips", len(zips))


if __name__ == "__main__":
    cache = cached()
    data()
    ladder(cache)
    results()
    stacks()
    errors()
    submissions()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
