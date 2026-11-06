"""Figures, tables and number macros for the Phase-3 (rhetorical roles) report, read from the run outputs.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p3/runs/<run>/cv_summary.json and fold*/summary.json (5-fold out-of-fold scores),
runs/ensembles/*.json (cross-run ensembles, calibration, post-processing study), configs/
(the tuning configs: their first comment line is the stated purpose, and the change is the
diff against the config they inherit from), submissions/ and the Phase-3 data files.
"""
import collections
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
P3 = HERE.parent
DATA = P3.parent / "capstone_llm" / "data" / "hf" / "Phase_3_RR"
FIGS, TABS = HERE / "figs", HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = ["#f4f7fc", "#dce8f7", "#b9d1f0", "#8fb6e8", "#5a92da", "#2a78d6", "#1d5aa3", "#123d70"]
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
LABELS = [k for k, _ in sorted(load(DATA / "Train" / "label_vocab.json").items(), key=lambda kv: kv[1])]
SHORT = {"ArgumentPetitioner": "ArgPet", "ArgumentRespondent": "ArgResp", "PrecedentReliedUpon": "PrecRelied",
         "PrecedentNotReliedUpon": "PrecNotRelied", "PrecedentOverruled": "PrecOverruled",
         "RulingByLowerCourt": "RulingLower", "RulingByPresentCourt": "RulingPresent", "RatioOfTheDecision": "Ratio"}
sh = lambda l: SHORT.get(l, l)


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def cv(run):
    return load(P3 / "runs" / run / "cv_summary.json")


def cfg_of(run):
    return yaml.safe_load(open(P3 / "runs" / run / "config.yaml"))


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


def arch_of(run):
    return cfg_of(run)["model"]["phase1_ckpt"].split("/")[1].split("_")[0]


def method_of(run):
    c = cfg_of(run)
    return c["peft"]["method"] + (" (KD)" if c.get("distill") else "")


# ------------------------------------------------------------------ data
def data():
    rows, dist = [], {}
    for dom in ("CL", "IT"):
        tr = load(DATA / "Train" / f"{dom}_train.json") + load(DATA / "Train" / f"{dom}_dev.json")
        te = load(DATA / "Test" / f"{dom}_test.json")
        n = [len(d["text"]) for d in tr]
        same = sum(a == b for d in tr for a, b in zip(d["labels"], d["labels"][1:]))
        pairs = sum(len(d["labels"]) - 1 for d in tr)
        dist[dom] = collections.Counter(l for d in tr for l in d["labels"])
        rows.append(f"{dom} & {len(tr)} & {sum(n):,} & {statistics.median(n):.0f} & {max(n):,} & {len(te)} & "
                    f"{sum(len(v) for v in te.values()):,} & {same / pairs:.3f} \\\\")
        M(f"persist{dom}", f"{same / pairs:.2f}")
        M(f"ndocs{dom}", len(tr))
    (TABS / "data.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for l in LABELS:
        rows.append(f"{tex(l)} & " + " & ".join(f"{dist[d][l]:,} & {100 * dist[d][l] / sum(dist[d].values()):.1f}\\%"
                                                for d in ("CL", "IT")) + " \\\\")
    (TABS / "labels.tex").write_text("\n".join(rows) + "\n")
    docs_with = {l: sum(l in set(d["labels"]) for dom in ("CL", "IT") for d in
                        load(DATA / "Train" / f"{dom}_train.json") + load(DATA / "Train" / f"{dom}_dev.json"))
                 for l in LABELS}
    M("docsDissent", docs_with["Dissent"])
    M("docsOverruled", docs_with["PrecedentOverruled"])
    M("sentsOverruled", dist["CL"]["PrecedentOverruled"] + dist["IT"]["PrecedentOverruled"])

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    order = sorted(LABELS, key=lambda l: dist["CL"][l] + dist["IT"][l])
    y = range(len(order))
    for k, dom in enumerate(("CL", "IT")):
        tot = sum(dist[dom].values())
        ax.barh([i + (k - 0.5) * 0.4 for i in y], [100 * dist[dom][l] / tot for l in order], 0.38, color=C[k],
                label=f"{dom} ({tot:,} sentences)")
    ax.set_yticks(list(y), [sh(l) for l in order], fontsize=7.5)
    ax.set_xlabel("share of training sentences (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "labels.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ run tables
FIRST = ["E1_a4_lorma", "E1_a4_lorma_origP1", "E0_a4_lorma_sentonly", "E2_a1_lora", "E3_a3_rslora", "E4_a2_ia3",
         "E5_a4_qlora", "E6_kd_a4_small", "E7_a3_adalora", "E8_a4_prefix", "E9_a1_prompt"]


def run_row(r):
    s = cv(r)
    o = s["oof"]
    return (f"{tex(r)} & {arch_of(r)} & {tex(method_of(r))} & {s['trainable_params'] / 1e6:.2f}M & "
            f"{o['all']['macro_f1']:.3f} & {o['CL']['macro_f1']:.3f} & {o['IT']['macro_f1']:.3f} & "
            f"{s['fold_macro_f1_mean']:.3f}$\\pm${s['fold_macro_f1_std']:.3f} & {o['all']['macro_p']:.3f} & "
            f"{o['all']['macro_r']:.3f} & {s['train_min_per_fold']:.1f} & {s['peak_gpu_gb']:.1f} & {s['ckpt_mb']:.1f} \\\\")


def first_round():
    (TABS / "first.tex").write_text("\n".join(run_row(r) for r in FIRST) + "\n")
    for r in FIRST:
        M("oof" + r.split("_")[0] + ("orig" if "orig" in r else ""), f"{cv(r)['oof']['all']['macro_f1']:.3f}")
    kd, te = cv("T13_kd_t7"), cv("T7_e1_causal_ep20_tau05")
    M("kdparams", f"{kd['total_params'] / 1e6:.1f}M")
    M("teacherparams", f"{te['total_params'] / 1e6:.1f}M")


def tuning():
    rows, deltas = [], []
    for d in ("tune", "tune2", "tune3", "tune4"):
        for f in sorted((P3 / "configs" / d).glob("*.yaml"), key=lambda p: int(p.stem.split("_")[0][1:])):
            raw = yaml.safe_load(open(f))
            name = raw["run_name"]
            if not (P3 / "runs" / name / "cv_summary.json").exists():
                continue
            purpose = next((l.lstrip("# ").strip() for l in open(f) if l.startswith("#")), "")
            purpose = purpose.split(":", 1)[1].strip() if ":" in purpose else purpose
            parent_cfg = (f.parent / raw["inherit"]).resolve()
            parent = yaml.safe_load(open(parent_cfg))["run_name"]
            a, b = flat(cfg_of(name)), flat(cfg_of(parent))
            change = ", ".join(f"{k.split('.')[-1]}={a.get(k)}" for k in sorted(set(a) | set(b))
                               if a.get(k) != b.get(k) and k not in ("run_name", "runs_dir") and
                               not k.startswith("distill.teacher_run")) or "--"
            if "distill.teacher_run" in a and a.get("distill.teacher_run") != b.get("distill.teacher_run"):
                change += ", teacher=" + a["distill.teacher_run"].split("/")[-1].split("_")[0]
            f1, pf1 = cv(name)["oof"]["all"]["macro_f1"], cv(parent)["oof"]["all"]["macro_f1"]
            s = cv(name)
            rows.append(f"{tex(name.split('_')[0])} & {tex(parent.split('_')[0])} & "
                        f"{tex(purpose)} & {tex(change)} & {f1:.3f} & {f1 - pf1:+.3f} & "
                        f"{s['fold_macro_f1_mean']:.3f}$\\pm${s['fold_macro_f1_std']:.3f} \\\\")
            deltas.append((name, parent, f1, pf1))
            M("oof" + name.split("_")[0], f"{f1:.3f}")
    (TABS / "tuning.tex").write_text("\n".join(rows) + "\n")

    # first round -> best tuned version of the same technique (dumbbell)
    pairs = [("LoRMA (A4)", "E1_a4_lorma", "T7_e1_causal_ep20_tau05"), ("LoRA (A1)", "E2_a1_lora", "T14_t8_win1024"),
             ("rsLoRA (A3)", "E3_a3_rslora", "T3_e3_ep20"), ("QLoRA (A4)", "E5_a4_qlora", "T10_e5_qlora_causal"),
             ("KD student (A4)", "E6_kd_a4_small", "T13_kd_t7"), ("prefix (A4)", "E8_a4_prefix", "T11_e8_prefix_causal"),
             ("prompt (A1)", "E9_a1_prompt", "T12_e9_prompt_causal"), ("IA3 (A2)", "E4_a2_ia3", None),
             ("AdaLoRA (A3)", "E7_a3_adalora", None)]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for i, (lab, e, t) in enumerate(pairs):
        fe = cv(e)["oof"]["all"]["macro_f1"]
        ax.plot([fe], [i], "o", ms=6, color=C[0], label="first round" if i == 0 else None)
        if t:
            ft = cv(t)["oof"]["all"]["macro_f1"]
            ax.plot([fe, ft], [i, i], color=GRID, lw=2, zorder=0)
            ax.plot([ft], [i], "s", ms=6, color=C[1], label="after tuning" if i == 0 else None)
            ax.annotate(f"{t.split('_')[0]} {ft:.3f}", (ft, i), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7)
        else:
            ax.annotate(f"{fe:.3f} (not tuned)", (fe, i), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK2)
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("out-of-fold macro-F1 (90 documents, 5-fold CV)")
    ax.set_xlim(0.25, 0.56)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(FIGS / "tuning.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ efficiency
def efficiency():
    runs = [r for r in FIRST if r not in ("E0_a4_lorma_sentonly", "E1_a4_lorma_origP1")] + \
           ["T7_e1_causal_ep20_tau05", "T14_t8_win1024", "T9_e3_ep20_tau05", "T10_e5_qlora_causal", "T13_kd_t7",
            "T11_e8_prefix_causal", "T12_e9_prompt_causal"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for r in runs:
        s = cv(r)
        tuned = r.startswith("T")
        ax.plot(s["trainable_params"] / 1e6, s["oof"]["all"]["macro_f1"], "s" if tuned else "o", ms=6,
                color=C[1] if tuned else C[0], markeredgecolor="white", markeredgewidth=0.8)
        off = {"T14": (-16, 2), "T7": (5, 3), "T10": (5, -7)}.get(r.split("_")[0], (4, 2))
        ax.annotate(r.split("_")[0], (s["trainable_params"] / 1e6, s["oof"]["all"]["macro_f1"]), xytext=off,
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.plot([], [], "o", color=C[0], label="first round")
    ax.plot([], [], "s", color=C[1], label="tuned")
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="out-of-fold macro-F1")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "efficiency.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ per-label and confusion
def per_label():
    runs = ["T14_t8_win1024", "T7_e1_causal_ep20_tau05", "T13_kd_t7", "T9_e3_ep20_tau05", "T10_e5_qlora_causal",
            "E4_a2_ia3", "E1_a4_lorma", "E0_a4_lorma_sentonly"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, r in enumerate(runs):
        pl = cv(r)["oof_per_label"]
        for j, l in enumerate(LABELS):
            v = pl.get(l, 0.0)
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=SEQ[min(len(SEQ) - 1, int(v * len(SEQ)))]))
            ax.text(j + 0.48, i + 0.46, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v >= 0.62 else INK)
    ax.set_xlim(0, len(LABELS))
    ax.set_ylim(len(runs), 0)
    ax.set_xticks([j + 0.48 for j in range(len(LABELS))], [sh(l) for l in LABELS], rotation=40, ha="right", fontsize=7)
    ax.set_yticks([i + 0.46 for i in range(len(runs))], [r.split("_")[0] + " " + arch_of(r) + " " + method_of(r)
                                                          for r in runs], fontsize=7)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIGS / "per_label.pdf")
    plt.close(fig)

    run = "T14_t8_win1024"
    conf = collections.defaultdict(collections.Counter)
    for f in sorted((P3 / "runs" / run).glob("fold*/summary.json")):
        for g, row in load(f)["confusion"].items():
            conf[g].update(row)
    present = [l for l in LABELS if sum(conf[l].values()) > 0]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for i, g in enumerate(present):
        tot = sum(conf[g].values())
        for j, p in enumerate(present):
            v = conf[g][p] / tot
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.94, color=SEQ[min(len(SEQ) - 1, int(v * len(SEQ)))]))
            if v >= 0.05:
                ax.text(j + 0.48, i + 0.47, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.2,
                        color="white" if v >= 0.62 else INK)
    ax.set_xlim(0, len(present))
    ax.set_ylim(len(present), 0)
    ax.set_xticks([j + 0.48 for j in range(len(present))], [sh(l) for l in present], rotation=45, ha="right",
                  fontsize=7)
    ax.set_yticks([i + 0.47 for i in range(len(present))], [f"{sh(l)} ({sum(conf[l].values())})" for l in present],
                  fontsize=7)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("gold label (sentences)")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIGS / "confusion.pdf")
    plt.close(fig)
    pairs = sorted(((conf[g][p], g, p) for g in present for p in present if g != p), reverse=True)[:6]
    M("topconf", "; ".join(f"{sh(g)}$\\to${sh(p)} {n}" for n, g, p in pairs))


# ------------------------------------------------------------------ ensembles, calibration, post-processing
def ensembles():
    rows = []
    for name in ("T5_ens_e1_e3", "C0_t4", "C4_t8_cal", "C1_ens_t7_t8", "C2_ens_t7_t8_t10", "C3_ens_t7_t8_t10_t9"):
        d = load(P3 / "runs" / "ensembles" / f"{name}.json")
        mem = ", ".join(m.split("/")[-1].split("_")[0] for m in d["members"])
        if "oof_uncalibrated" in d:
            u, c = d["oof_uncalibrated"], d["oof_calibrated_nested"]
            rows.append(f"{tex(name.split('_')[0])} & {mem} & {u['all']:.3f} & {u['CL']:.3f} & {u['IT']:.3f} & "
                        f"{c['all']:.3f} & {c['all'] - u['all']:+.3f} \\\\")
        else:
            o = d["oof"]
            rows.append(f"{tex(name.split('_')[0])} & {mem} & {o['all']['macro_f1']:.3f} & {o['CL']['macro_f1']:.3f} & "
                        f"{o['IT']['macro_f1']:.3f} & -- & -- \\\\")
        if name.startswith("C3"):
            M("ensCthree", f"{d['oof_uncalibrated']['all']:.3f}")
            M("ensCthreecal", f"{d['oof_calibrated_nested']['all']:.3f}")
    (TABS / "ensembles.tex").write_text("\n".join(rows) + "\n")

    pp = load(P3 / "runs" / "ensembles" / "C3_ens_t7_t8_t10_t9_postproc.json")
    rows = []
    res = sorted(pp["results"], key=lambda r: (r["block"], r["scale"], r["bias"]))
    for r in res:
        rows.append(f"{tex(r['block'])} & {r['scale']:g} & {'yes' if r['bias'] else 'no'} & {r['oof_all']:.4f} & "
                    f"{r['test_like_mean']:.4f} & {r['test_like_p10']:.3f}--{r['test_like_p90']:.3f} & "
                    f"{r['segments_per_doc']:.1f} \\\\")
    (TABS / "postproc.tex").write_text("\n".join(rows) + "\n")
    M("goldseg", f"{pp['gold_segments_per_doc']:.1f}")
    base = next(r for r in pp["results"] if r["block"] == "none" and r["scale"] == 1 and not r["bias"])
    M("simmean", f"{base['test_like_mean']:.3f}")
    M("simpten", f"{base['test_like_p10']:.3f}")
    M("simpninety", f"{base['test_like_p90']:.3f}")
    M("predseg", f"{base['segments_per_doc']:.1f}")

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    sel = [r for r in res if not r["bias"]] + [r for r in res if r["bias"]]
    for i, r in enumerate(sel):
        col = C[1] if r["bias"] else C[0]
        ax.plot([r["test_like_p10"], r["test_like_p90"]], [i, i], color=col, lw=1.2, alpha=0.6)
        ax.plot([r["test_like_mean"]], [i], "o", ms=4.5, color=col)
    ax.set_yticks(range(len(sel)), [f"{r['block']}, x{r['scale']:g}" + (", bias" if r["bias"] else "") for r in sel],
                  fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 on simulated test sets (5 CL + 5 IT docs): mean and 10th-90th percentile")
    fig.savefig(FIGS / "postproc.pdf")
    plt.close(fig)


def submissions():
    zips = sorted(p.name for p in (P3 / "submissions").glob("*.zip"))
    rows = []
    for z in zips:
        run = z.rsplit("_", 2)[0]
        rows.append(f"\\code{{{tex(z)}}} & {tex(run.split('_')[0])} \\\\")
    (TABS / "submissions.tex").write_text("\n".join(rows) + "\n")
    M("nzips", len(zips))


if __name__ == "__main__":
    data()
    first_round()
    tuning()
    efficiency()
    per_label()
    ensembles()
    submissions()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
