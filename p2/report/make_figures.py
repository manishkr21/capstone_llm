"""Figures, tables and number macros for the Phase-2 (L-NER) report, read from the run outputs.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p2/runs/* (the six experiments + A2 diagnostics), p2/sweeps/* (one-change sweep,
3-split confirmation, learning curve), sweeps/RESULTS.md (inference-only decoding sweep),
sweeps/confirm_scores.json, and the text outputs of scripts/check_alignment.py and
scripts/error_analysis.py saved next to this file.  "Change" columns are computed by diffing
each variant's config.yaml against its base run, so they cannot drift from what was run.
"""
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DATA = P2.parent / "capstone_llm" / "data" / "hf" / "Phase_2_LNER"
FIGS, TABS = HERE / "figs", HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated categorical order
SEQ = ["#e8f0fb", "#c4d8f3", "#8fb6e8", "#5a92da", "#2a78d6", "#1d5aa3", "#123d70"]  # blue, light -> dark
NEG, POS, MID = "#eb6834", "#2a78d6", "#c3c2b7"  # diverging poles + neutral
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


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def config_diff(run_dir, base_dir):
    a, b = flat(yaml.safe_load(open(run_dir / "config.yaml"))), flat(yaml.safe_load(open(base_dir / "config.yaml")))
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in ("run_name", "runs_dir"))
    return ", ".join(f"{k.split('.')[-1]}={a.get(k)}" for k in keys) or "(same)"


LABELS = [l.strip() for l in open(DATA / "Train" / "labels.txt") if l.strip()]
EXPS = ["E1_a3_full", "E2_a4_lorma", "E3_a1_lora", "E4_a2_adapter", "E5_a4_partial", "E6_kd_a3_small"]
ARCH_OF = {"prefixlm": "A3", "efficient": "A4", "gpt": "A1", "encdec": "A2"}


# ------------------------------------------------------------------ data
def data_tables():
    import collections
    counts, docs, words = collections.Counter(), 0, []
    for f in ("fold1", "fold2"):
        t, g = load(DATA / "Train" / f"{f}-text.json"), load(DATA / "Train" / f"{f}-gold.json")
        for k in t:
            docs += 1
            words.append(len(t[k].split()))
            counts.update(e[2] for e in g[k])
    test = load(DATA / "Test" / "fold3-text.json")
    rows = [f"{tex(l)} & {counts[l]:,} \\\\" for l in LABELS]
    rows.append(f"\\midrule Total & {sum(counts.values()):,} \\\\")
    (TABS / "labels.tex").write_text("\n".join(rows) + "\n")
    M("ntraindocs", docs)
    M("ntestdocs", len(test))
    M("medwords", f"{statistics.median(words):,.0f}")
    M("maxwords", f"{max(words):,}")
    M("totents", f"{sum(counts.values()):,}")
    al = (HERE / "alignment.txt").read_text()
    m = re.findall(r"presplit=(\w+): oracle macro-F1 ([\d.]+)", al)
    for flag, v in m:
        M("oracle" + ("split" if flag == "True" else "nosplit"), v)


# ------------------------------------------------------------------ six experiments
def main_results():
    rows, pts = [], []
    for r in EXPS + ["diag_E4_a2_adapter_encoder", "diag_a2_partial_encoder", "diag_a2_full"]:
        s = load(P2 / "runs" / r / "summary.json")
        cfg = yaml.safe_load(open(P2 / "runs" / r / "config.yaml"))
        d = s["dev"]
        arch = ARCH_OF[s["arch"]]
        method = s["method"] + (" (KD)" if s.get("distilled_from") else "")
        extra = "" if r.startswith("E") else " (enc.)"
        rows.append(f"{tex(r)} & {arch}{extra} & {tex(method)} & {s['trainable_params'] / 1e6:.2f}M & "
                    f"{s['trainable_pct']:.1f} & {d['macro_f1']:.4f} & {d['macro_p']:.3f} & {d['macro_r']:.3f} & "
                    f"{d['micro_f1']:.3f} & {d['span_only_f1']:.3f} & {s['train_minutes']:.1f} & "
                    f"{s['peak_gpu_gb']:.1f} & {s['checkpoint_mb']:.1f} \\\\")
        if r in EXPS:
            pts.append((r, arch, method, s["trainable_params"], d["macro_f1"], d["span_only_f1"]))
            M("ef" + r.split("_")[0], f"{d['macro_f1']:.4f}")
        if r == EXPS[-1]:
            rows.append("\\midrule")
    (TABS / "main.tex").write_text("\n".join(rows) + "\n")
    ks = load(P2 / "runs" / "E6_kd_a3_small" / "summary.json")
    M("kdparams", f"{ks['total_params'] / 1e6:.1f}M")
    M("teacherparams", f"{load(P2 / 'runs' / 'E1_a3_full' / 'summary.json')['total_params'] / 1e6:.1f}M")

    # macro-F1 vs trainable parameters (log x), direct labels
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for r, arch, method, tp, f1, span in pts:
        ax.plot(tp / 1e6, f1, "o", ms=7, color=C[0], markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"{r.split('_')[0]} {arch} {method}", (tp / 1e6, f1), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="dev strict macro-F1")
    ax.set_xlim(1, 400)
    fig.savefig(FIGS / "f1_vs_params.pdf")
    plt.close(fig)

    # per-label F1 heatmap (sequential blue), value in every cell
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    mat = [[load(P2 / "runs" / r / "summary.json")["dev"]["per_label"][l]["f1"] for l in LABELS] for r in EXPS]
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            col = SEQ[min(len(SEQ) - 1, int(v * len(SEQ)))]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=col))
            ax.text(j + 0.48, i + 0.46, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v >= 0.58 else INK)
    ax.set_xlim(0, len(LABELS))
    ax.set_ylim(len(EXPS), 0)
    ax.set_xticks([j + 0.48 for j in range(len(LABELS))], [l for l in LABELS], rotation=40, ha="right", fontsize=7)
    ax.set_yticks([i + 0.46 for i in range(len(EXPS))], [r.split("_")[0] + " " + r.split("_", 1)[1] for r in EXPS],
                  fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(FIGS / "per_label.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ one-change sweep
def sweep():
    text = (P2 / "sweeps" / "RESULTS.md").read_text()
    block = text.split("## One change at a time")[1].split("```")[1]
    rows, deltas = [], []
    for line in block.strip().splitlines()[1:]:
        p = line.split()
        name, base, bf1, f1, dlt, span = p[0], p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5])
        change = config_diff(P2 / "sweeps" / name, P2 / "runs" / base)
        rows.append(f"{tex(name)} & {tex(base.split('_')[0])} & {tex(change)} & {bf1:.4f} & {f1:.4f} & "
                    f"{dlt:+.4f} & {span:.3f} \\\\")
        deltas.append((name, base.split("_")[0], dlt))
    (TABS / "sweep.tex").write_text("\n".join(rows) + "\n")
    M("nsweep", len(rows))

    deltas.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(5.8, 6.4))
    ys = range(len(deltas))
    ax.barh(list(ys), [d for *_, d in deltas], height=0.65,
            color=[POS if d > 0.005 else NEG if d < -0.005 else MID for *_, d in deltas])
    ax.set_yticks(list(ys), [f"{n} (vs {b})" for n, b, _ in deltas], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    for y, (_, _, d) in zip(ys, deltas):
        ax.annotate(f"{d:+.3f}", (d, y), xytext=(3 if d >= 0 else -3, 0), textcoords="offset points",
                    ha="left" if d >= 0 else "right", va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("change in dev macro-F1 vs its base run (dev split 0)")
    ax.grid(axis="y", visible=False)
    lo, hi = min(d for *_, d in deltas), max(d for *_, d in deltas)
    ax.set_xlim(lo - 0.03, hi + 0.03)
    fig.savefig(FIGS / "sweep_deltas.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ inference-only decoding sweep
def inference():
    text = (P2 / "sweeps" / "RESULTS.md").read_text().split("## Inference-only")[1].split("## Confirmation")[0]
    res, run = {}, None
    for line in text.splitlines():
        if m := re.match(r"== runs/(\S+)", line.strip()):
            run = m[1]
        elif m := re.match(r"\s*stride\s+(\d+)\s+(\S+)\s+macro-F1 ([\d.]+)\s+P ([\d.]+)\s+R ([\d.]+)\s+span ([\d.]+)", line):
            res.setdefault(run, {})[(int(m[1]), m[2])] = tuple(float(x) for x in m.group(3, 4, 5, 6))
    opts = ["argmax", "viterbi", "viterbi+merge_space", "viterbi+propagate5", "viterbi+propagate10",
            "viterbi+merge+propagate10"]
    rows = []
    for r, d in res.items():
        for o in opts:
            vals = " & ".join(f"{d[(s, o)][0]:.4f}" for s in (64, 128, 256))
            rows.append(f"{tex(r.split('_')[0]) if o == opts[0] else ''} & {tex(o)} & {vals} & {d[(128, o)][1]:.3f} & "
                        f"{d[(128, o)][2]:.3f} \\\\")
        rows.append("\\midrule")
    (TABS / "inference.tex").write_text("\n".join(rows[:-1]) + "\n")
    e2 = res["E2_a4_lorma"]
    M("argmaxEtwo", f"{e2[(128, 'argmax')][0]:.4f}")
    M("viterbiEtwo", f"{e2[(128, 'viterbi')][0]:.4f}")
    M("propfiveEtwo", f"{e2[(128, 'viterbi+propagate5')][0]:.4f}")

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, r in enumerate(res):
        ys = [res[r][(128, o)][0] for o in opts]
        ax.plot(range(len(opts)), ys, marker="osD"[i], ms=6, color=C[i], label=r.split("_")[0] + " " + r.split("_", 1)[1])
    ax.set_xticks(range(len(opts)), ["argmax", "Viterbi", "+merge", "+prop5", "+prop10", "+merge\n+prop10"], fontsize=7.5)
    ax.set_ylabel("dev macro-F1 (stride 128)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "decoding.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ 3-split confirmation
NICE = {"f_E2": "E2 (A4 LoRMA, original backbone)", "f_E2invsqrt": "E2 + inv-sqrt class weights",
        "f_E2invsqrt_r32": "E2 + inv-sqrt + rank 32", "f_E2focal": "E2 + focal loss", "f_A4lora": "A4 LoRA",
        "f_A4lora_invsqrt": "A4 LoRA + inv-sqrt", "f_E2clean": "E2 on cleaned R1 backbone",
        "f_E2clean1B": "E2 on cleaned R2 (1B) backbone"}


def confirm():
    sc = load(P2 / "sweeps" / "confirm_scores.json")
    cands = sorted({k.rsplit("_s", 1)[0] for k in sc}, key=lambda c: list(NICE).index(c))
    posts = ["none", "merge", "prop5", "prop10", "merge+prop5", "merge+prop10"]
    rows, stats = [], {}
    for c in cands:
        runs = [sc[f"{c}_s{s}"] for s in (0, 1, 2) if f"{c}_s{s}" in sc]
        stats[c] = {p: (statistics.mean(r[p] for r in runs), statistics.pstdev(r[p] for r in runs)) for p in posts}
        rows.append(f"{tex(NICE[c])} & " + " & ".join(f"{stats[c][p][0]:.3f}$\\pm${stats[c][p][1]:.3f}"
                                                         for p in posts) + " \\\\")
    (TABS / "confirm.tex").write_text("\n".join(rows) + "\n")
    for c, n in (("f_E2", "Etwo"), ("f_E2clean", "Etwoclean"), ("f_E2clean1B", "EtwocleanB")):
        M(f"conf{n}none", f"{stats[c]['none'][0]:.3f}")
        M(f"conf{n}nonesd", f"{stats[c]['none'][1]:.3f}")
        M(f"conf{n}prop", f"{stats[c]['prop10'][0]:.3f}")
    best = max(((c, p) for c in cands for p in posts), key=lambda cp: stats[cp[0]][cp[1]][0])
    M("bestconf", f"{stats[best[0]][best[1]][0]:.3f}")
    M("bestconfsd", f"{stats[best[0]][best[1]][1]:.3f}")
    M("bestconfname", tex(NICE[best[0]]) + ", " + best[1])

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for k, (p, col, mk) in enumerate((("none", C[0], "o"), ("prop10", C[1], "s"))):
        ys = range(len(cands))
        ax.errorbar([stats[c][p][0] for c in cands], [y + (k - 0.5) * 0.25 for y in ys],
                    xerr=[stats[c][p][1] for c in cands], fmt=mk, ms=5, color=col, ecolor=col, elinewidth=1,
                    capsize=2, label="no post-processing" if p == "none" else "+ propagate10")
    ax.set_yticks(range(len(cands)), [NICE[c] for c in cands], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("dev macro-F1, mean $\\pm$ std over dev splits 0, 1, 2")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(FIGS / "confirm.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ learning curve
def learning_curve():
    pts = [(load(P2 / "sweeps" / f"lc_e2_n{n}" / "summary.json")["dev"]["macro_f1"], n) for n in (15, 30)]
    pts.append((load(P2 / "runs" / "E2_a4_lorma" / "summary.json")["dev"]["macro_f1"], 60))
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot([n for _, n in pts], [f for f, _ in pts], marker="o", ms=6, color=C[0])
    for f, n in pts:
        ax.annotate(f"{f:.3f}", (n, f), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5)
        M(f"lc{n}", f"{f:.3f}")
    ax.set(xlabel="training documents", ylabel="dev macro-F1", xticks=[15, 30, 60], ylim=(0.25, 0.55))
    fig.savefig(FIGS / "learning_curve.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ error analysis (scripts/error_analysis.py)
def errors():
    txt = (HERE / "error_E2.txt").read_text()
    shares = {m[0]: (int(m[1]), float(m[2])) for m in
              re.findall(r"^(exact|wrong_label|boundary|bound\+label|missed)\s+(\d+)\s+([\d.]+)%", txt, re.M)}
    for k, (n, s) in shares.items():
        M("err" + k.replace("_", "").replace("+", ""), f"{s:.1f}")
    M("errspurious", re.search(r"spurious pred\s+(\d+)", txt)[1])
    per = {}
    for m in re.finditer(r"^(\S+)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)$", txt, re.M):
        if m[1] in LABELS:
            per[m[1]] = (int(m[2]), [int(m[i]) for i in range(3, 8)], int(m[8]))
    rows = [f"{tex(l)} & {per[l][0]} & " + " & ".join(f"{v}\\%" for v in per[l][1]) + f" & {per[l][2]} \\\\"
            for l in LABELS]
    (TABS / "errors.tex").write_text("\n".join(rows) + "\n")
    conf = re.search(r"label confusions.*?: (.*)", txt)[1]
    M("topconfusions", tex(conf.replace("->", "$\\to$")).replace("\\$\\textbackslash{}to\\$", "$\\to$"))

    cats = ["exact", "wrong label", "boundary", "boundary + label", "missed"]
    order = sorted(LABELS, key=lambda l: per[l][1][0])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    left = [0] * len(order)
    for k, cat in enumerate(cats):
        vals = [per[l][1][k] for l in order]
        ax.barh(range(len(order)), vals, left=left, color=C[k], height=0.7, label=cat, edgecolor="white", linewidth=0.8)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(order)), [f"{l} ({per[l][0]})" for l in order], fontsize=7.5)
    ax.set_xlabel("share of gold entities (%)  -  E2, dev split 0")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.savefig(FIGS / "errors.pdf")
    plt.close(fig)


if __name__ == "__main__":
    data_tables()
    main_results()
    sweep()
    inference()
    confirm()
    learning_curve()
    errors()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
