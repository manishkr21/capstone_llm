"""Figures, tables and number macros for the Phase-5 (legal statute identification) report.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p5/runs/<run>/{summary.json, metrics.jsonl, val_probs.npy, config.yaml}, p5/runs/submissions_meta/*.json
(cross-fitted validation scores of every submission), the token cache (p5/cache/{train,val}.npz: labels and
lengths), the raw Phase-5 files (split integrity) and p5/submissions/.  No test label is read anywhere.
STACK10 / STACK11 have no metadata file; their cross-fitted scores are copied from the run log (SCORES_LOG).
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
P5 = HERE.parent
META = P5 / "runs" / "submissions_meta"
DATA = P5.parent / "capstone_llm" / "data" / "hf" / "Phase_5_LSI"
FIGS, TABS = HERE / "figs", HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
load = lambda p: json.load(open(p))
macros = {}
DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
FIRST = ["E1_a4_lorma", "E2_a1_full", "E3_a3_adapter", "E4_a2_partial", "E5_a4_partial", "E6_a1_qlora",
         "E7_a3_adalora", "E8_a4_prefix", "E9_a1_prompt", "E10_kd_a4_small"]
LATER = ["V1_e5_evidence", "V2_e5_ep5", "V3_e2_ep5", "W1_a4_partial_ep8", "W2_a4_full_ep5", "W3_a3_adapter_ep5",
         "W4_a4_lorma_ep5", "W5_a1_qlora_ep5", "W6_kd2_a4_small", "X1_a1_qlora_ep8", "X2_a4_full_ep8",
         "X3_kd3_a4_small", "Y1_kd4_a4", "Y2_kd4_a4_alpha03", "Y3_kd4_a1", "Y4_a3_full_ep5", "Y5_a2_full_ep5"]
LEXICAL = ["T1_tfidf_word", "T2_tfidf_char", "T3_tfidf_svm", "T5_tfidf_lr13", "T6_tfidf_cv", "M1_lexmlp"]
TEACHERS = {"TEACHER": ["E2_a1_full", "E3_a3_adapter", "E5_a4_partial"],
            "TEACHER2": ["E1_a4_lorma", "E3_a3_adapter", "E6_a1_qlora", "V2_e5_ep5", "V3_e2_ep5"],
            "TEACHER3": ["W1_a4_partial_ep8", "W2_a4_full_ep5", "W3_a3_adapter_ep5", "W5_a1_qlora_ep5", "W6_kd2_a4_small"],
            "TEACHER4": ["W1_a4_partial_ep8", "W2_a4_full_ep5", "W3_a3_adapter_ep5", "W4_a4_lorma_ep5", "W5_a1_qlora_ep5",
                         "W6_kd2_a4_small", "X1_a1_qlora_ep8", "X3_kd3_a4_small"]}
# cross-fitted validation scores printed when the two final zips were written (no metadata file was saved)
SCORES_LOG = {"STACK10_seedavg": {"macro_f1": 0.4602, "macro_p": 0.4747, "macro_r": 0.4604, "micro_f1": 0.556, "subset_acc": 0.084},
              "STACK11_full": {"macro_f1": 0.4603, "macro_p": 0.4747, "macro_r": 0.4589, "micro_f1": 0.5583, "subset_acc": 0.0856}}
NICE = {"full": "full fine-tuning", "lorma": "LoRMA", "adapter": "Houlsby adapters", "partial": "partial (top 4 blocks)",
        "qlora": "QLoRA (NF4)", "adalora": "AdaLoRA", "prefix": "prefix tuning", "prompt": "prompt tuning"}


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def summ(r):
    return load(P5 / "runs" / r / "summary.json")


def meta(r):
    return load(META / f"{r}.json")


def cfg(r):
    return yaml.safe_load(open(P5 / "runs" / r / "config.yaml"))


def arch(r):
    return cfg(r)["model"]["phase1_ckpt"].split("/")[1].split("_")[0]


def method(r):
    c = cfg(r)
    if c.get("distill"):
        t = Path(c["distill"]["teacher_logits"]).parent.name.replace("_ens", "")
        return f"KD student ({len(c['model']['keep_layers'])} blocks, {t})"
    return NICE[c["peft"]["method"]]


def epochs(r):
    return cfg(r)["train"]["epochs"]


def short(r):
    return r.split("_")[0]


YV = np.load(P5 / "cache" / "val.npz")["labels"].astype(bool)
YT = np.load(P5 / "cache" / "train.npz")["labels"].astype(bool)
FREQ = YT.sum(0)


# ------------------------------------------------------------------ data
def data():
    h = lambda t: hashlib.md5(" ".join(t.split()).encode()).hexdigest()
    tr, dv, te = (load(DATA / "Train" / "train.json"), load(DATA / "Train" / "dev.json"), load(DATA / "Test" / "test.json"))
    htr, hte = Counter(h(r["text"]) for r in tr), {h(r["text"]) for r in te}
    M("ntrainraw", f"{len(tr):,}")
    M("ntraindup", f"{len(tr) - len(htr):,}")
    M("ndevraw", f"{len(dv):,}")
    M("ndevintest", f"{sum(h(r['text']) in hte for r in dv):,}")
    M("ndevintrain", f"{sum(h(r['text']) in htr for r in dv):,}")
    rows = []
    for name in ("train", "val", "test"):
        z = np.load(P5 / "cache" / f"{name}.npz")
        L = np.diff(z["offs"])
        lab = z["labels"]
        lpd = f"{lab.sum(1).mean():.2f}" if lab.any() else "--"
        rows.append(f"{name} & {len(L):,} & {lpd} & {np.median(L):,.0f} & {np.percentile(L, 90):,.0f} & {L.sum() / 1e6:.1f}M \\\\")
        M(f"n{name}", f"{len(L):,}")
        if name == "val":
            M("shortereight", f"{100 * (L <= 4096).mean():.0f}")
    (TABS / "data.tex").write_text("\n".join(rows) + "\n")
    M("nlabels", YT.shape[1])
    M("rarecount", int((FREQ < 500).sum()))
    M("topshare", f"{100 * np.sort(FREQ)[::-1][:10].sum() / FREQ.sum():.0f}")
    M("freqmax", f"{FREQ.max():,}")
    M("freqmin", f"{FREQ.min():,}")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axes[0].bar(range(len(FREQ)), np.sort(FREQ)[::-1], color=C[0], width=1.0)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="statute (sorted by frequency)", ylabel="training documents")
    axes[0].grid(axis="x", visible=False)
    k = YT.sum(1)
    axes[1].hist(k, bins=np.arange(0.5, 16.5), color=C[1], rwidth=0.85)
    axes[1].set(xlabel="statutes per training document", ylabel="documents")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "labels.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ experiments
def row(r):
    s, m = summ(r), meta(r)
    cf = m["val_crossfit"]
    return (f"{tex(short(r))} & {arch(r)} & {tex(method(r))} & {epochs(r)} & {s['trainable_params'] / 1e6:.2f}M & "
            f"{s['trainable_pct']:.1f} & {s['val']['macro_f1']:.4f} & {cf['macro_f1']:.4f} & {cf['macro_p']:.3f} & "
            f"{cf['macro_r']:.3f} & {cf['micro_f1']:.3f} & {cf['subset_acc']:.3f} \\\\")


def results():
    rows = [row(r) for r in FIRST] + ["\\midrule"] + [row(r) for r in LATER]
    (TABS / "results.tex").write_text("\n".join(rows) + "\n")
    for r in FIRST + LATER:
        M("cf" + short(r), f"{meta(r)['val_crossfit']['macro_f1']:.3f}")
    rows = []
    for r in LEXICAL:
        s = summ(r)
        v = s.get("val_crossfit") or meta(r)["val_crossfit"] if (META / f"{r}.json").exists() else s.get("val_crossfit")
        kind = "linear SVM" if "svm" in r else s.get("kind", "lexical").split(" (")[0]
        args = s.get("args", {})
        desc = f"{kind}, {args.get('analyzer', '')} {tuple(args.get('ngram', [])) if args.get('ngram') else ''}".strip(" ,")
        rows.append(f"{tex(r)} & {tex(desc)} & {v['macro_f1']:.4f} & {v['micro_f1']:.3f} & {s['train_min']:.1f} \\\\")
        M("cf" + r.split("_")[0], f"{v['macro_f1']:.3f}")
    (TABS / "lexical.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for r in FIRST + ["W4_a4_lorma_ep5", "W2_a4_full_ep5", "Y2_kd4_a4_alpha03", "Y3_kd4_a1"]:
        s = summ(r)
        rows.append(f"{tex(short(r))} & {arch(r)} & {tex(method(r))} & {s['total_params'] / 1e6:.1f}M & "
                    f"{s['trainable_params'] / 1e6:.2f}M & {s['train_min']:.0f} & {s['test_infer_s']:.0f} & "
                    f"{s['peak_gpu_gb']:.1f} & {s['ckpt_mb']:.1f} \\\\")
    (TABS / "efficiency.tex").write_text("\n".join(rows) + "\n")
    st, tf = summ("Y2_kd4_a4_alpha03"), summ("W2_a4_full_ep5")
    M("kdparams", f"{st['total_params'] / 1e6:.1f}M")
    M("fullparams", f"{tf['total_params'] / 1e6:.1f}M")
    M("kdinfer", f"{st['test_infer_s']:.0f}")

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for k, (runs, lab) in enumerate(((FIRST, "first round (3 epochs)"), (LATER, "longer training / KD rounds"))):
        x = [summ(r)["trainable_params"] / 1e6 for r in runs]
        y = [meta(r)["val_crossfit"]["macro_f1"] for r in runs]
        ax.plot(x, y, "o", ms=6, color=C[k], markeredgecolor="white", label=lab)
    for r in FIRST:
        ax.annotate(f"{short(r)} {arch(r)}", (summ(r)["trainable_params"] / 1e6, meta(r)["val_crossfit"]["macro_f1"]),
                    xytext=(4, -3), textcoords="offset points", fontsize=6.5, color=INK2)
    for r in ("Y2_kd4_a4_alpha03", "W4_a4_lorma_ep5", "W2_a4_full_ep5"):
        ax.annotate(f"{short(r)} {arch(r)}", (summ(r)["trainable_params"] / 1e6, meta(r)["val_crossfit"]["macro_f1"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=C[1])
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="cross-fitted val macro-F1")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(FIGS / "f1_vs_params.pdf")
    plt.close(fig)


def curves():
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.4), sharey=True)
    for ax, r in zip(axes.flat, FIRST):
        recs = [json.loads(l) for l in open(P5 / "runs" / r / "metrics.jsonl") if "sel_macro_f1_tuned" in l]
        st = [x["step"] / 1000 for x in recs]
        ax.plot(st, [x["sel_macro_f1_tuned"] for x in recs], color=C[1], lw=1.3, label="tuned global threshold")
        ax.plot(st, [x["sel_macro_f1_at_0.5"] for x in recs], color=C[0], lw=1.1, label="threshold 0.5")
        ax.set_title(f"{short(r)} {arch(r)}", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("step (k)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("selection macro-F1", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ ensembles and stacking
def stacks():
    names = ["ENS1_5models", "ENS2_greedy4", "ENS7_greedy4", "STACK4_all", "STACK6_all", "STACK7_hybrid",
             "STACK8_hybrid", "STACK9_gbm_mlp", "STACK10_seedavg", "STACK11_full"]
    rows, ladder = [], []
    for n in names:
        if n in SCORES_LOG:
            v, kind, members = SCORES_LOG[n], "MLP + GBM stackers, seed-averaged", "all runs + lexical"
        else:
            m = meta(n)
            if "val_crossfit" in m:
                v, kind = m["val_crossfit"], "mean of probabilities"
            elif "blend_crossfit" in m:
                v, kind = m["blend_crossfit"], "per-statute GBM + MLP stacker"
            else:
                v, kind = m["stacked_crossfit"], "MLP stacker (5-fold on val)"
            rs = [x.split("/")[-1] for x in m.get("runs", [])]
            lex = sum(x.startswith(("T", "M1")) for x in rs)
            members = f"{len(rs)} runs" + (f" ({lex} lexical)" if lex else " (Phase-1 only)")
        rows.append(f"{tex(n)} & {tex(kind)} & {tex(members)} & {v['macro_f1']:.4f} & {v['macro_p']:.3f} & {v['macro_r']:.3f} & {v['micro_f1']:.3f} \\\\")
        M("cf" + n.split("_")[0].replace("STACK", "stack").replace("ENS", "ens"), f"{v['macro_f1']:.3f}")
    (TABS / "stacks.tex").write_text("\n".join(rows) + "\n")
    M("meanensall", f"{meta('STACK6_all')['mean_ensemble_crossfit']['macro_f1']:.3f}")

    steps = [("best first-round model (E10 KD)", meta("E10_kd_a4_small")["val_crossfit"]["macro_f1"]),
             ("longer training (W4 LoRMA, 5 ep)", meta("W4_a4_lorma_ep5")["val_crossfit"]["macro_f1"]),
             ("KD from larger teachers (Y2)", meta("Y2_kd4_a4_alpha03")["val_crossfit"]["macro_f1"]),
             ("greedy mean ensemble (ENS7)", meta("ENS7_greedy4")["val_crossfit"]["macro_f1"]),
             ("MLP stacker, Phase-1 only (STACK6)", meta("STACK6_all")["stacked_crossfit"]["macro_f1"]),
             ("+ TF-IDF inputs (STACK8, hybrid)", meta("STACK8_hybrid")["stacked_crossfit"]["macro_f1"]),
             ("+ GBM stacker blend (STACK11)", SCORES_LOG["STACK11_full"]["macro_f1"])]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.barh(range(len(steps)), [v for _, v in steps], color=[C[0]] * 5 + [C[1]] * 2, height=0.62)
    for i, (_, v) in enumerate(steps):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5)
    ax.set_yticks(range(len(steps)), [s for s, _ in steps], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.33, 0.48)
    ax.set_xlabel("cross-fitted validation macro-F1 (orange: uses lexical TF-IDF models)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGS / "ladder.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ per-statute behaviour of the best single model
def per_label():
    r = "Y2_kd4_a4_alpha03"
    p = np.load(P5 / "runs" / r / "val_probs.npy")
    thr = np.array(meta(r)["thresholds"])
    P = p >= thr[None, :]
    tp, fp, fn = (P & YV).sum(0), (P & ~YV).sum(0), (~P & YV).sum(0)
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), np.nan)
    q = np.quantile(FREQ, [0, .2, .4, .6, .8, 1])
    rows = []
    for k in range(5):  # frequency quintiles of the 100 statutes
        m = (FREQ >= q[k]) & ((FREQ <= q[k + 1]) if k == 4 else (FREQ < q[k + 1]))
        rows.append(f"Q{k + 1} & {int(q[k]):,}--{int(q[k + 1]):,} & {m.sum()} & {np.nanmean(f1[m]):.3f} & "
                    f"{np.nanmin(f1[m]):.3f} & {np.nanmax(f1[m]):.3f} \\\\")
    (TABS / "perlabel.tex").write_text("\n".join(rows) + "\n")
    M("corrfreq", f"{np.corrcoef(np.log(FREQ), np.nan_to_num(f1))[0, 1]:.2f}")

    # most frequent (gold statute missed -> wrong statute predicted in the same document) pairs
    pairs = Counter()
    for i in range(len(YV)):
        miss, wrong = np.flatnonzero(YV[i] & ~P[i]), np.flatnonzero(P[i] & ~YV[i])
        for a in miss:
            for b in wrong:
                pairs[(a, b)] += 1
    co = (YT.T.astype(np.int32) @ YT.astype(np.int32))
    rows = []
    for (a, b), n in pairs.most_common(10):
        rows.append(f"{a} & {b} & {n} & {FREQ[a]:,} & {FREQ[b]:,} & {co[a, b]:,} \\\\")
    (TABS / "confusions.tex").write_text("\n".join(rows) + "\n")
    worst = [j for j in np.argsort(np.nan_to_num(f1, nan=1.0)) if FREQ[j] >= np.median(FREQ)][:8]  # frequent but weak
    (TABS / "worst.tex").write_text("\n".join(f"{j} & {FREQ[j]:,} & {YV[:, j].sum():,} & {f1[j]:.3f} & {tp[j]} & {fp[j]} & {fn[j]} \\\\" for j in worst) + "\n")

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.scatter(FREQ, f1, s=12, color=C[0], edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set(xlabel="training documents with the statute (log scale)", ylabel="val F1 of the statute (Y2)")
    fig.savefig(FIGS / "perlabel.pdf")
    plt.close(fig)

    # threshold strategies on the same model
    def macro(Pm):
        tp_, fp_, fn_ = (Pm & YV).sum(0), (Pm & ~YV).sum(0), (~Pm & YV).sum(0)
        present = (tp_ + fp_ + fn_) > 0
        return float(np.mean((2 * tp_ / np.maximum(1, 2 * tp_ + fp_ + fn_))[present]))
    s = summ(r)
    rows = [f"0.5 for every statute & {macro(p >= 0.5):.4f} \\\\",
            f"one global threshold ({s['global_threshold']:.2f}) & {macro(p >= s['global_threshold']):.4f} \\\\",
            f"per-statute thresholds, fitted on all of val (in-sample) & {meta(r)['val_in_sample']['macro_f1']:.4f} \\\\",
            f"per-statute thresholds, 2-fold cross-fitted (honest) & {meta(r)['val_crossfit']['macro_f1']:.4f} \\\\"]
    (TABS / "thresholds.tex").write_text("\n".join(rows) + "\n")


def submissions():
    zips = sorted(p.name for p in (P5 / "submissions").glob("*.zip"))
    (TABS / "submissions.tex").write_text("\n".join(f"\\code{{{tex(z)}}} \\\\" for z in zips) + "\n")
    M("nzips", len(zips))
    for t, members in TEACHERS.items():
        M("members" + t.lower(), ", ".join(tex(short(x)) for x in members))


if __name__ == "__main__":
    data()
    results()
    curves()
    stacks()
    per_label()
    submissions()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
