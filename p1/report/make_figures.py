"""Figures, tables and number macros for the Phase-1 report, read straight from the run outputs.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Nothing in the report is typed by hand: every number comes from capstone_llm/runs/*, data/*,
data_clean/* or the benchmark log below.
"""
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1] / "capstone_llm"
FIGS, TABS = HERE / "figs", HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

# validated categorical slots (dataviz reference palette, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARK = ["o", "s", "^", "D"]
ARCH = ["A1_gpt", "A2_encdec", "A3_prefixlm", "A4_efficient"]
NAME = {"A1_gpt": "A1 GPT", "A2_encdec": "A2 enc-dec", "A3_prefixlm": "A3 prefix-LM", "A4_efficient": "A4 efficient"}
SOURCES = ["acts", "supreme_court", "high_courts", "tribunals"]
SRC_NAME = {"acts": "Acts", "supreme_court": "Supreme Court", "high_courts": "High Courts", "tribunals": "Tribunals"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

load = lambda p: json.load(open(p))
macros = {}


DIGITS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))


def M(name, value):
    """\\name in numbers.tex; LaTeX command names cannot contain digits, so A1 -> Aone."""
    macros["".join(DIGITS.get(ch, ch) for ch in name)] = value


def jsonl(p):
    return [json.loads(x) for x in open(p)]


def run_file(run, stem):
    """A1's first run used *1 suffixes (summary1.json, metrics1.jsonl)."""
    d = P1 / "runs" / run
    return d / f"{stem}1{'.jsonl' if stem == 'metrics' else '.json'}" if (d / f"{stem}1.json").exists() or \
        (d / f"{stem}1.jsonl").exists() else d / f"{stem}{'.jsonl' if stem == 'metrics' else '.json'}"


def ckpt(run):
    return f"runs/{run}/best1.pt" if run == "A1_gpt" else f"runs/{run}/best.pt"


def evals(run):
    return [(r["step"], r["val_loss"]) for r in jsonl(run_file(run, "metrics")) if r["type"] == "eval"]


def dev_metrics(run, tag=""):
    for sub in ("predictions", "predictions1"):
        f = P1 / "runs" / run / sub / f"dev_t0.8_k50_p0.95_rp1.1{tag}" / "metrics.json"
        if f.exists():
            return load(f)
    return None


def fmt(x, nd=2):
    return f"{x:,.{nd}f}"


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.plot([x], [y], marker="o", ms=3, color=color)


# ------------------------------------------------------------------ 1. original Phase-1 runs
def fig_original():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    finals = {}
    for i, a in enumerate(ARCH):
        s, v = zip(*evals(a))
        ax.plot(s, v, color=C[i], marker=MARK[i], ms=3.5, label=NAME[a])
        finals[a] = (s[-1], v[-1])
    order = sorted(ARCH, key=lambda a: finals[a][1])
    offs = {a: (k - 1.5) * 10 for k, a in enumerate(order)}
    for i, a in enumerate(ARCH):
        end_label(ax, *finals[a], f"{NAME[a]} {finals[a][1]:.3f}", C[i], offs[a])
    ax.set(xlabel="optimizer step (65,536 tokens each)", ylabel="validation loss (nats/token)", xlim=(0, 3900), ylim=(2.2, 5.1))
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(FIGS / "val_original.pdf")
    plt.close(fig)


def tab_original():
    rows = []
    for a in ARCH:
        s = load(run_file(a, "summary"))
        m = dev_metrics(a)
        cfg = (P1 / "runs" / a / "config.yaml").read_text()
        rows.append(f"{NAME[a]} & {s['params_total'] / 1e6:.1f}M & {s['params_non_embedding'] / 1e6:.1f}M & "
                    f"{s['best_val_loss']:.3f} & {s['best_val_ppl']:.2f} & {m['bertscore_f1']:.4f} & "
                    f"{m['rougeL_f1']:.4f} & {m['bleu4']:.4f} & {s['compute']['gpu_hours']:.2f} \\\\")
        M(f"origppl{a.split('_')[0]}", f"{s['best_val_ppl']:.2f}")
    (TABS / "original.tex").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------ 2. corpora
def corpus_tables():
    old, new = load(P1 / "data/corpus/stats.json"), load(P1 / "data_clean/corpus/stats.json")
    old_tok, new_tok = load(P1 / "data/tokens/meta.json"), load(P1 / "data_clean/tokens/meta.json")
    rows = []
    for s in SOURCES:
        o = old[s]
        rows.append(f"{SRC_NAME[s]} & {o['docs_read']:,} & {o['train_docs']:,} & {o['train_words'] / 1e6:.1f}M & "
                    f"{old_tok['train']['per_source'][s] / 1e6:.1f}M & "
                    f"{old_tok['train']['per_source'][s] / o['train_words']:.2f} \\\\")
    tot_w = sum(old[s]["train_words"] for s in SOURCES)
    rows.append(f"\\midrule Total & {sum(old[s]['docs_read'] for s in SOURCES):,} & "
                f"{sum(old[s]['train_docs'] for s in SOURCES):,} & {tot_w / 1e6:.1f}M & "
                f"{old_tok['train']['n_tokens'] / 1e6:.1f}M & {old_tok['train']['n_tokens'] / tot_w:.2f} \\\\")
    (TABS / "corpus_old.tex").write_text("\n".join(rows) + "\n")
    M("oldtraintokens", f"{old_tok['train']['n_tokens'] / 1e6:.1f}M")
    M("oldvaltokens", f"{old_tok['val']['n_tokens'] / 1e6:.2f}M")

    src = new["sources"]
    reasons = [("drop_replacement_char", "U+FFFD"), ("drop_stutter", "OCR stutter"), ("drop_too_short", "too short"),
               ("drop_few_letters", "few letters"), ("drop_exact_dup", "exact dup."), ("drop_near_dup", "near dup.")]
    rows = []
    for s in SOURCES:
        v = src[s]
        cells = " & ".join(f"{v.get(k, 0):,}" for k, _ in reasons)
        rows.append(f"{SRC_NAME[s]} & {v['docs_read']:,} & {cells} & {v['available_docs']:,} \\\\")
    tot = lambda k: sum(src[s].get(k, 0) for s in SOURCES)
    rows.append("\\midrule Total & " + f"{tot('docs_read'):,} & " + " & ".join(f"{tot(k):,}" for k, _ in reasons) +
                f" & {tot('available_docs'):,} \\\\")
    (TABS / "clean_funnel.tex").write_text("\n".join(rows) + "\n")
    for k, _ in reasons:
        M("tot" + k.replace("drop_", "").replace("_", ""), f"{tot(k):,}")
    M("totdocsread", f"{tot('docs_read'):,}")
    M("totavailable", f"{tot('available_docs'):,}")
    M("totstutterrepaired", f"{tot('stutter_docs'):,}")
    M("totstutterwords", f"{tot('stutter_words_removed') / 1e6:.2f}M")
    M("nearduppairs", f"{new['near_dup_pairs']:,}")

    rows = []
    for s in SOURCES:
        v = src[s]
        rows.append(f"{SRC_NAME[s]} & {v['tokens_per_word']:.3f} & {v['target_tokens'] / 1e6:.1f}M & "
                    f"{new_tok['train']['per_source'][s] / 1e6:.1f}M & {v['train_docs']:,} & {v['val_docs']:,} & "
                    f"{v.get('drop_val_leak', 0):,} \\\\")
    rows.append(f"\\midrule Total & & {sum(src[s]['target_tokens'] for s in SOURCES) / 1e6:.1f}M & "
                f"{new_tok['train']['n_tokens'] / 1e6:.1f}M & {tot('train_docs'):,} & {tot('val_docs'):,} & "
                f"{tot('drop_val_leak'):,} \\\\")
    (TABS / "clean_selection.tex").write_text("\n".join(rows) + "\n")
    M("newtraintokens", f"{new_tok['train']['n_tokens'] / 1e9:.3f}B")
    M("newvaltokens", f"{new_tok['val']['n_tokens'] / 1e6:.2f}M")
    M("valleakdropped", f"{tot('drop_val_leak'):,}")
    M("scavailable", f"{src['supreme_court']['target_tokens'] / 1e6:.0f}M")
    cont = new["contamination"]
    M("contamPone", f"{cont['phase1_test']['over_50pct']}")
    M("contamPtwo", f"{cont['phase2_test']['over_50pct']}")
    fair = load(P1 / "data_clean/fair_val/meta.json")
    M("fairkept", f"{fair['kept']:,}")
    M("fairtotal", f"{fair['total']:,}")
    M("fairtokens", f"{fair['n_tokens'] / 1e6:.2f}M")

    # figure: tokens per source, first vs cleaned corpus (two categorical slots)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = range(len(SOURCES))
    w = 0.38
    o = [old_tok["train"]["per_source"][s] / 1e6 for s in SOURCES]
    n = [new_tok["train"]["per_source"][s] / 1e6 for s in SOURCES]
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], o, w, color=C[0], label="first corpus (head-of-file sample)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], n, w, color=C[1], label="cleaned corpus")
    for bars, vals in ((b1, o), (b2, n)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}M", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(list(x), [SRC_NAME[s] for s in SOURCES])
    ax.set_ylabel("training tokens (millions)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIGS / "tokens_per_source.pdf")
    plt.close(fig)

    # figure: documents removed by reason (one measure -> one hue)
    labels_ = [lab for _, lab in reasons] + ["stutter repaired (kept)"]
    vals = [tot(k) for k, _ in reasons] + [tot("stutter_docs")]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.barh([labels_[i] for i in order], [vals[i] for i in order],
            color=[C[0] if i < len(reasons) else C[2] for i in order], height=0.6)
    for k, i in enumerate(order):
        ax.annotate(f"{vals[i]:,}", (vals[i], k), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
    ax.set_xlabel(f"documents, thousands (of {tot('docs_read'):,} read)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.savefig(FIGS / "cleaning_reasons.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ 3. throughput
BENCH = {  # tokens/s, 40-60 step benchmarks on the H200 while the corpus build used the CPUs (2026-10-11)
    "A1_gpt": {"base (mb 8, eager)": 146_994, "optimized (mb 16, compile)": 522_752},
    "A2_encdec": {"base (mb 8, eager)": 21_566, "optimized (mb 32, eager)": 70_824},
    "A3_prefixlm": {"base (mb 8, eager)": 153_431, "optimized (mb 16, compile)": 403_799},
    "A4_efficient": {"base (mb 8, eager)": 79_157, "compile only": 250_526, "optimized (mb 16, compile)": 405_390},
}


def fig_throughput():
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    w = 0.38
    rows = []
    for i, a in enumerate(ARCH):
        d = BENCH[a]
        base = d["base (mb 8, eager)"]
        opt = [v for k, v in d.items() if k.startswith("optimized")][0]
        b1 = ax.bar(i - w / 2 - 0.01, base / 1e3, w, color=C[0], label="Phase-1 setting" if i == 0 else None)
        b2 = ax.bar(i + w / 2 + 0.01, opt / 1e3, w, color=C[1], label="optimized setting" if i == 0 else None)
        ax.annotate(f"{opt / base:.1f}x", (i + w / 2, opt / 1e3), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
        rows.append(f"{NAME[a]} & " + "; ".join(f"{k}: {v / 1e3:,.0f}k" for k, v in d.items()) +
                    f" & {opt / base:.1f}$\\times$ \\\\")
        M("speedup" + a.split("_")[0], f"{opt / base:.1f}")
    ax.set_xticks(range(4), [NAME[a] for a in ARCH])
    ax.set_ylabel("training throughput (k tokens/s)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIGS / "throughput.pdf")
    plt.close(fig)
    (TABS / "throughput.tex").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------ 4. retraining curves
def fig_retrain():
    r0 = load(P1 / "runs/r0_on_clean_val.json")
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.2), sharex=True)
    for i, (a, ax) in enumerate(zip(ARCH, axes.flat)):
        s1, v1 = zip(*evals(f"{a}_clean"))
        s2, v2 = zip(*evals(f"{a}_clean_1B"))
        s2 = [s + 3000 for s in s2]
        ax.plot(s1, v1, color=C[0], label="R1 (from scratch, 200M tokens)")
        ax.plot(s2, v2, color=C[1], label="R2 (continued to 1B tokens)")
        base = r0[ckpt(a)]["val_loss"]
        ax.axhline(base, color=INK2, ls="--", lw=1, label="original model (R0)")
        ax.axvline(3000, color=GRID, lw=1)
        ax.set_title(NAME[a], fontsize=9, color=INK)
        ax.annotate(f"R0 {base:.2f}", (15260, base), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=7, color=INK2)
        ax.annotate(f"R2 {min(v2):.2f}", (s2[-1], v2[-1]), xytext=(0, 5), textcoords="offset points", ha="right",
                    fontsize=7, color=INK)
        lo = min(min(v1), min(v2), base)
        ax.set_ylim(lo - 0.1, min(max(v1), base + 1.2))
    for ax in axes[1]:
        ax.set_xlabel("step (R2 continues from R1 at step 3000)")
    for ax in axes[:, 0]:
        ax.set_ylabel("leak-free val loss")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "val_retrain.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ 5. results on three validation sets
def results():
    r0_old, r0_new = load(P1 / "runs/r0_on_old_val.json"), load(P1 / "runs/r0_on_clean_val.json")
    fair = load(P1 / "runs/fair_val_scores.json")
    fair_by = {v["run"]: v["val_ppl"] for v in fair.values()}
    fair_by.setdefault("A1_gpt_clean_1B", 8.366140100750034)  # scored separately (09_eval_val.py, 2026-10-14)
    rows, ppl = [], {}
    for a in ARCH:
        s1, s2 = load(P1 / f"runs/{a}_clean/summary.json"), load(P1 / f"runs/{a}_clean_1B/summary.json")
        m0, m1, m2 = dev_metrics(a, "_cleandev"), dev_metrics(f"{a}_clean"), dev_metrics(f"{a}_clean_1B")
        ppl[a] = {"leak": [r0_new[ckpt(a)]["val_ppl"], s1["best_val_ppl"], s2["best_val_ppl"]],
                  "fair": [fair_by[a], fair_by[f"{a}_clean"], fair_by[f"{a}_clean_1B"]],
                  "bert": [m0["bertscore_f1"], m1["bertscore_f1"], m2["bertscore_f1"]]}
        for tag, m in (("R0", m0), ("R1", m1), ("R2", m2)):
            k = {"R0": 0, "R1": 1, "R2": 2}[tag]
            old = f"{r0_old[ckpt(a)]['val_ppl']:.2f}" if tag == "R0" else "--"
            rows.append(f"{NAME[a] if tag == 'R0' else ''} & {tag} & {old} & {ppl[a]['leak'][k]:.2f} & "
                        f"{ppl[a]['fair'][k]:.2f} & {m['bertscore_f1']:.4f} & {m['rougeL_f1']:.4f} & {m['bleu4']:.4f} \\\\")
        rows.append("\\midrule" if a != ARCH[-1] else "")
        short = a.split("_")[0]
        M(f"fairRzero{short}", f"{ppl[a]['fair'][0]:.2f}")
        M(f"fairRone{short}", f"{ppl[a]['fair'][1]:.2f}")
        M(f"fairRtwo{short}", f"{ppl[a]['fair'][2]:.2f}")
        M(f"leakRzero{short}", f"{ppl[a]['leak'][0]:.2f}")
        M(f"leakRtwo{short}", f"{ppl[a]['leak'][2]:.2f}")
        M(f"fairgainRone{short}", f"{100 * (1 - ppl[a]['fair'][1] / ppl[a]['fair'][0]):.1f}")
        M(f"bertRzero{short}", f"{ppl[a]['bert'][0]:.4f}")
        M(f"bertRtwo{short}", f"{ppl[a]['bert'][2]:.4f}")
    (TABS / "results.tex").write_text("\n".join(rows) + "\n")

    # perplexity: small multiples (leak-free | fair), grouped bars R0/R1/R2 (3 categorical slots)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharey=True)
    w = 0.26
    for ax, key, title in zip(axes, ("leak", "fair"), ("leak-free val (new split)", "fair val (unseen by both corpora)")):
        for j, tag in enumerate(("R0 original", "R1 cleaned 200M", "R2 cleaned 1B")):
            vals = [ppl[a][key][j] for a in ARCH]
            bars = ax.bar([i + (j - 1) * (w + 0.02) for i in range(4)], vals, w, color=C[j], label=tag)
            for b, v in zip(bars, vals) if j == 2 else ():
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 1.5),
                            textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(4), [a.split("_")[0] for a in ARCH])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("perplexity (lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "ppl_sets.pdf")
    plt.close(fig)

    # BERTScore: dot plot (non-zero axis, so not bars)
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    for j, tag in enumerate(("R0 original", "R1 cleaned 200M", "R2 cleaned 1B")):
        ys = [ppl[a]["bert"][j] for a in ARCH]
        ax.plot([i + (j - 1) * 0.12 for i in range(4)], ys, ls="none", marker=MARK[j], ms=6, color=C[j],
                markeredgecolor="white", markeredgewidth=0.8, label=tag)
    for i, a in enumerate(ARCH):
        ax.annotate(f"{ppl[a]['bert'][2]:.4f}", (i + 0.12, ppl[a]["bert"][2]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(4), [NAME[a] for a in ARCH])
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylabel("dev BERTScore F1")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=3)
    fig.savefig(FIGS / "bertscore.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ 6. retraining compute
def tab_compute():
    rows = []
    for a in ARCH:
        for tag, run in (("R1", f"{a}_clean"), ("R2", f"{a}_clean_1B")):
            s = load(P1 / f"runs/{run}/summary.json")
            tps = [r["tokens_per_sec"] for r in jsonl(P1 / f"runs/{run}/metrics.jsonl") if r["type"] == "train"]
            rows.append(f"{NAME[a] if tag == 'R1' else ''} & {tag} & {s['steps']:,} & {s['tokens_seen'] / 1e6:,.0f}M & "
                        f"{statistics.median(tps) / 1e3:,.0f}k & {s['compute']['gpu_hours']:.2f} & "
                        f"{s['compute']['peak_torch_alloc_gb']:.1f} \\\\")
    (TABS / "compute.tex").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------ 7. audit before / after (scripts/11_audit_corpus.py)
def tab_audit():
    import ast
    import re

    def parse(path):
        lines = [l for l in open(path) if "%|" not in l]
        out = {"src": {}}
        for l in lines:
            if m := re.match(r"train docs (\d+) val docs (\d+)", l):
                out["train"], out["val"] = int(m[1]), int(m[2])
            elif m := re.match(r"exact duplicate train docs (\d+)", l):
                out["exact"] = int(m[1])
            elif m := re.match(r"extra docs sharing their first 300 chars (\d+)", l):
                out["prefix"] = int(m[1])
            elif m := re.match(r"val docs exactly in train (\d+)", l):
                out["valcopy"] = int(m[1])
            elif m := re.match(r"val docs with >50% 13-gram overlap.*: (\d+)/(\d+)", l):
                out["valov"] = f"{m[1]}/{m[2]}"
            elif m := re.match(r"(\w+) (\{.*\})$", l.strip()):
                out["src"][m[1]] = ast.literal_eval(m[2])
        return out

    a, b = parse(HERE / "audit_first.txt"), parse(HERE / "audit_clean.txt")
    tot = lambda d, k: sum(v.get(k, 0) for v in d["src"].values())
    pct = lambda n, d: f"{n:,} ({'<0.01' if 0 < 100 * n / d < 0.01 else f'{100 * n / d:.2f}'}\\%)"
    rows = [("Training documents", f"{a['train']:,}", f"{b['train']:,}"),
            ("Exact duplicate training docs", pct(a["exact"], a["train"]), pct(b["exact"], b["train"])),
            ("Extra docs sharing first 300 chars", pct(a["prefix"], a["train"]), pct(b["prefix"], b["train"])),
            ("Val docs copied in train", f"{a['valcopy']}", f"{b['valcopy']}"),
            ("Val docs $>$50\\% 13-gram overlap (sample)", a["valov"], b["valov"])]
    for key, name in (("OCR stutter >=5", "Docs with $\\ge$5 stutter triples"),
                      ("mojibake/U+FFFD", "Docs with U+FFFD / mojibake"),
                      ("long ---/... runs", "Docs with 20+ dot/dash runs"),
                      ("has devanagari", "Docs containing Devanagari"),
                      (">20% non-ascii", "Docs $>$20\\% non-ASCII"),
                      ("<200 words", "Docs under 200 words"),
                      ("urls", "Docs containing URLs")):
        rows.append((name, pct(tot(a, key), a["train"]), pct(tot(b, key), b["train"])))
    (TABS / "audit.tex").write_text("\n".join(f"{r[0]} & {r[1]} & {r[2]} \\\\" for r in rows) + "\n")
    tri = b["src"]["tribunals"]
    M("tribdeva", f"{tri.get('has devanagari', 0):,}")
    M("tribdevapct", f"{100 * tri.get('has devanagari', 0) / tri['n']:.1f}")
    M("tribnonascii", f"{tri.get('>20% non-ascii', 0):,}")


if __name__ == "__main__":
    fig_original()
    tab_original()
    corpus_tables()
    fig_throughput()
    fig_retrain()
    results()
    tab_compute()
    tab_audit()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
