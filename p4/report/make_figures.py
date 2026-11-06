"""Figures, tables and number macros for the Phase-4 (court judgment prediction) report.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p4/runs/<run>/{summary.json, metrics.jsonl, val_probs.npy, config.yaml}, the token cache
(p4/cache/*.npz: labels and document lengths of the clean validation split), the raw Phase-4 files
(for the data-integrity checks) and p4/submissions/.  No test label is read anywhere.
"""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
P4 = HERE.parent
DATA = P4.parent / "capstone_llm" / "data" / "hf" / "Phase_4_CJPE"
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
MAIN = ["E1_a1_full", "E2_a4_lorma", "E3_a3_adapter", "E4_a2_bitfit", "E5_a1_adalora", "E6_a4_partial",
        "E7_a3_gradual", "E8_kd_a4_small"]
ABL = ["E6_a4_partial", "A_e6_first_last", "A_e6_spread", "A_e6_first"]
NICE = {"full": "full fine-tuning", "lorma": "LoRMA+", "adapter": "Houlsby adapters", "bitfit": "BitFit",
        "adalora": "AdaLoRA", "partial": "partial (top 4 blocks)", "gradual": "gradual unfreezing"}


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def summ(r):
    return load(P4 / "runs" / r / "summary.json")


def cfg(r):
    return yaml.safe_load(open(P4 / "runs" / r / "config.yaml"))


def arch(r):
    return cfg(r)["model"]["phase1_ckpt"].split("/")[1].split("_")[0]


def method(r):
    c = cfg(r)
    return "KD student (full FT)" if c.get("distill") else NICE[c["peft"]["method"]]


def macro_f1(y, pred):
    out = []
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        p, g = int((pred == c).sum()), int((y == c).sum())
        pr, rc = (tp / p if p else 0.0), (tp / g if g else 0.0)
        out.append((pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0))
    return (out[0][2] + out[1][2]) / 2, (out[0][0] + out[1][0]) / 2, (out[0][1] + out[1][1]) / 2


def best_thr(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    s = [macro_f1(y, (p >= t).astype(int))[0] for t in grid]
    return float(grid[int(np.argmax(s))])


VAL = np.load(P4 / "cache" / "val.npz")
YV = VAL["labels"]
LENV = np.diff(VAL["offs"])


# ------------------------------------------------------------------ data and integrity
def data():
    tr, dv, te = (load(DATA / "Train" / "train.json"), load(DATA / "Train" / "dev.json"), load(DATA / "Test" / "test.json"))
    h = lambda t: hashlib.md5(t.encode()).hexdigest()
    hte = {h(r["text"]) for r in te}
    tr_h = [h(r["text"]) for r in tr]
    M("ntrainraw", f"{len(tr):,}")
    M("ntrainuniq", f"{len(set(tr_h)):,}")
    M("ntraindup", f"{len(tr) - len(set(tr_h)):,}")
    M("ndev", f"{len(dv):,}")
    M("ntest", f"{len(te):,}")
    M("ntestindev", f"{sum(h(r['text']) in {h(x['text']) for x in dv} for r in te):,}")
    M("testhaslabel", "yes" if all("label" in r for r in te) else "no")
    tc = {k: np.load(P4 / "cache" / f"{k}.npz") for k in ("train", "val", "test")}
    rows = []
    for name, z in tc.items():
        n = len(z["offs"]) - 1
        L = np.diff(z["offs"])
        lab = z["labels"]
        bal = f"{(lab == 0).sum():,} / {(lab == 1).sum():,}" if (lab >= 0).all() else "-- (not read)"
        rows.append(f"{name} & {n:,} & {bal} & {np.median(L):,.0f} & {np.percentile(L, 90):,.0f} & {L.sum() / 1e6:.1f}M \\\\")
        M(f"n{name}", f"{n:,}")
        M(f"med{name}", f"{np.median(L):,.0f}")
    (TABS / "data.tex").write_text("\n".join(rows) + "\n")
    trl = tc["train"]["labels"]
    M("trainzero", f"{100 * (trl == 0).mean():.1f}")

    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    for k, (name, z) in enumerate((("train", tc["train"]), ("clean validation", tc["val"]))):
        L = np.diff(z["offs"])
        ax.hist(np.log10(np.maximum(L, 1)), bins=50, density=True, histtype="step", lw=1.6, color=C[k], label=name)
    for x, lab in ((np.log10(512), "1 chunk"), (np.log10(4096), "8 chunks")):
        ax.axvline(x, color=INK2, ls="--", lw=0.8)
        ax.annotate(lab, (x, ax.get_ylim()[1] * 0.9), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xticks([2, 3, 4, 5], ["100", "1k", "10k", "100k"])
    ax.set(xlabel="document length (BPE tokens, log scale)", ylabel="density")
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "lengths.pdf")
    plt.close(fig)
    M("shortereight", f"{100 * (LENV <= 4096).mean():.0f}")


# ------------------------------------------------------------------ main results
def results():
    rows = []
    for r in MAIN + ["A_e6_first_last", "A_e6_spread", "A_e6_first"]:
        s = summ(r)
        v = s["val"]
        if r == "A_e6_first_last":
            rows.append("\\midrule")
        rows.append(f"{tex(r)} & {arch(r)} & {tex(method(r))} & {tex(s['data']['strategy'])} & "
                    f"{s['trainable_params'] / 1e6:.2f}M & {s['trainable_pct']:.1f} & {v['macro_f1']:.4f} & "
                    f"{s['val_f1_at_0.5']:.4f} & {s['threshold']:.2f} & {v['macro_p']:.3f} & {v['macro_r']:.3f} \\\\")
        M("fone" + r.split("_")[0].replace("A", "abl") + ("" if not r.startswith("A_") else r.split("_", 2)[2].replace("_", "")),
          f"{v['macro_f1']:.3f}")
    (TABS / "results.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for r in MAIN:
        s = summ(r)
        rows.append(f"{tex(r.split('_')[0])} & {arch(r)} & {tex(method(r))} & {s['total_params'] / 1e6:.1f}M & "
                    f"{s['trainable_params'] / 1e6:.2f}M & {s['train_min']:.1f} & {s['test_infer_s']:.1f} & "
                    f"{s['peak_gpu_gb']:.1f} & {s['ckpt_mb']:.1f} \\\\")
    (TABS / "efficiency.tex").write_text("\n".join(rows) + "\n")
    t, st = summ("E6_a4_partial"), summ("E8_kd_a4_small")
    M("kdparams", f"{st['total_params'] / 1e6:.1f}M")
    M("teacherparams", f"{t['total_params'] / 1e6:.1f}M")
    M("kdretain", f"{100 * st['val']['macro_f1'] / t['val']['macro_f1']:.1f}")
    M("kdinfer", f"{st['test_infer_s']:.0f}")
    M("teacherinfer", f"{t['test_infer_s']:.0f}")

    # F1 vs trainable parameters
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for r in MAIN:
        s = summ(r)
        ax.plot(s["trainable_params"] / 1e6, s["val"]["macro_f1"], "o", ms=7, color=C[0], markeredgecolor="white")
        off = {"E2": (6, -9), "E3": (-8, 6), "E1": (6, -9), "E7": (6, 3), "E8": (-10, -10), "E6": (-8, 5)}.get(r.split("_")[0], (6, 2))
        ax.annotate(f"{r.split('_')[0]} {arch(r)} {method(r)}", (s["trainable_params"] / 1e6, s["val"]["macro_f1"]),
                    xytext=off, textcoords="offset points", fontsize=7, color=INK,
                    ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="clean-validation macro-F1", xlim=(0.4, 400))
    fig.savefig(FIGS / "f1_vs_params.pdf")
    plt.close(fig)

    # threshold: F1 at 0.5 vs tuned (dumbbell)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for i, r in enumerate(MAIN):
        s = summ(r)
        ax.plot([s["val_f1_at_0.5"], s["val"]["macro_f1"]], [i, i], color=GRID, lw=2, zorder=0)
        ax.plot(s["val_f1_at_0.5"], i, "o", ms=6, color=C[0], label="threshold 0.5" if i == 0 else None)
        ax.plot(s["val"]["macro_f1"], i, "s", ms=6, color=C[1], label="validation-tuned threshold" if i == 0 else None)
        ax.annotate(f"t={s['threshold']:.2f}", (s["val"]["macro_f1"], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(MAIN)), [f"{r.split('_')[0]} {arch(r)} {method(r)}" for r in MAIN], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("clean-validation macro-F1")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(FIGS / "threshold.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ training curves
def curves():
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 3.6), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, MAIN):
        recs = [json.loads(l) for l in open(P4 / "runs" / r / "metrics.jsonl")]
        st = [x["step"] for x in recs]
        ax.plot(st, [x["val_f1_tuned"] for x in recs], color=C[1], lw=1.4, label="tuned threshold")
        ax.plot(st, [x["val_f1_at_0.5"] for x in recs], color=C[0], lw=1.2, label="threshold 0.5")
        ax.set_title(f"{r.split('_')[0]} {arch(r)}", fontsize=8)
        ax.set_ylim(0.3, 0.82)
    for ax in axes[1]:
        ax.set_xlabel("step", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("val macro-F1", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ section ablation
def ablation():
    lab = {"E6_a4_partial": "last 8 chunks", "A_e6_first_last": "first 4 + last 4", "A_e6_spread": "8 spread evenly",
           "A_e6_first": "first 8 chunks"}
    vals = [summ(r)["val"]["macro_f1"] for r in ABL]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.barh(range(len(ABL)), vals, color=C[0], height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(ABL)), [lab[r] for r in ABL], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.80)
    ax.set_xlabel("clean-validation macro-F1 (E6 recipe, 8 x 512 tokens)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGS / "ablation.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ errors of the best single model
def errors():
    r = "E6_a4_partial"
    p = np.load(P4 / "runs" / r / "val_probs.npy")
    thr = summ(r)["threshold"]
    pred = (p >= thr).astype(int)
    fp, fn = ((pred == 1) & (YV == 0)), ((pred == 0) & (YV == 1))
    M("nfp", int(fp.sum()))
    M("nfn", int(fn.sum()))
    q = np.quantile(LENV, [0, .25, .5, .75, 1])
    rows = []
    for k in range(4):
        m = (LENV >= q[k]) & (LENV <= q[k + 1]) if k == 3 else (LENV >= q[k]) & (LENV < q[k + 1])
        f1 = macro_f1(YV[m], pred[m])[0]
        rows.append(f"Q{k + 1} & {int(q[k]):,}--{int(q[k + 1]):,} & {m.sum()} & {f1:.3f} & {(fp & m).sum()} & {(fn & m).sum()} \\\\")
    (TABS / "errors_len.tex").write_text("\n".join(rows) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
    for k, (name, m) in enumerate((("label 0 (dismissed)", YV == 0), ("label 1 (allowed)", YV == 1))):
        axes[0].hist(p[m], bins=30, range=(0, 1), histtype="step", lw=1.6, color=C[k], label=name)
    axes[0].axvline(thr, color=INK2, ls="--", lw=0.9)
    axes[0].annotate(f"threshold {thr:.2f}", (thr, axes[0].get_ylim()[1] * 0.55), xytext=(3, 0), textcoords="offset points",
                     fontsize=7, color=INK2)
    axes[0].set(xlabel="predicted P(label 1)", ylabel="documents")
    axes[0].legend(fontsize=7, loc="upper right")
    labels = []
    for k in range(4):
        m = (LENV >= q[k]) & (LENV <= q[k + 1]) if k == 3 else (LENV >= q[k]) & (LENV < q[k + 1])
        axes[1].bar(k - 0.2, 100 * fp[m].sum() / m.sum(), 0.38, color=C[0], label="false positive" if k == 0 else None)
        axes[1].bar(k + 0.2, 100 * fn[m].sum() / m.sum(), 0.38, color=C[1], label="false negative" if k == 0 else None)
        labels.append(f"Q{k + 1}\n{int(q[k] / 1000)}k-{int(q[k + 1] / 1000)}k")
    axes[1].set_xticks(range(4), labels, fontsize=7)
    axes[1].set(ylabel="% of documents in quartile", xlabel="document length quartile (tokens)")
    axes[1].grid(axis="x", visible=False)
    axes[1].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "errors.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ ensembles + submissions
def ensembles():
    rows = []
    for name, runs in (("ENS\\_top3", ["E6_a4_partial", "E7_a3_gradual", "E1_a1_full"]),
                       ("ENS\\_top5", ["E6_a4_partial", "E7_a3_gradual", "E1_a1_full", "E3_a3_adapter", "E2_a4_lorma"])):
        p = np.mean([np.load(P4 / "runs" / r / "val_probs.npy") for r in runs], 0)
        t = best_thr(YV, p)
        f1, pr, rc = macro_f1(YV, (p >= t).astype(int))
        rows.append(f"{name} & {', '.join(r.split('_')[0] for r in runs)} & {f1:.4f} & {t:.2f} & {pr:.3f} & {rc:.3f} \\\\")
        M("ens" + ("three" if "3" in name else "five"), f"{f1:.3f}")
    (TABS / "ensembles.tex").write_text("\n".join(rows) + "\n")
    zips = sorted(p.name for p in (P4 / "submissions").glob("*.zip"))
    (TABS / "submissions.tex").write_text("\n".join(f"\\code{{{tex(z)}}} \\\\" for z in zips) + "\n")
    M("nzips", len(zips))


if __name__ == "__main__":
    data()
    results()
    curves()
    ablation()
    errors()
    ensembles()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
