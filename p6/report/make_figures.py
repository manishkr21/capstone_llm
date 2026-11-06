"""Figures, tables and number macros for the Phase-6 (abstractive summarization) report.

    ../../cenv/bin/python make_figures.py      # -> figs/*.pdf, tables/*.tex, numbers.tex

Sources: p6/runs/<run>/{summary*.json, config.yaml, metrics.jsonl, val_pred*.json}, p6/runs/mbr/*.json, the
validation split (summ.data.load_splits) and p6/submissions/.  Per-document BERTScore and the extractive
baselines are computed here once (bert_score, roberta-large, as in summ/metrics.py) and cached in
report/cache.json.  No test reference exists or is read.
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
P6 = HERE.parent
sys.path.insert(0, str(P6))
from summ.data import load_splits  # noqa: E402
from summ.metrics import score  # noqa: E402
import mbr  # noqa: E402

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
load = lambda p: json.load(open(p))
macros = {}
DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
FIRST = ["F1_a4_full", "F2_a2_full_fid", "E1_a4_lorma", "E2_a1_lora", "E3_a3_adapter", "E4_a2_qlora_fid",
         "E5_a3_prefix", "E6_a1_adalora", "E7_a4_prompt"]
SECOND = ["L2_a1_full_ep6", "L3_a3_full_ep6", "K1_kd_a4_6l", "S2_a1_lora_spread"]
LONG = ["S1_a4_full_src2560", "S3_a4_full_src4096", "S3b_a4_full_src4096_fromS1", "S4_a4_full_src8192",
        "S5_a3_full_src4096", "S9_a3_full_src6144", "S6_a1_full_src2560", "S7_a1_full_src4096",
        "S7b_a1_full_src4096_seed1", "S7c_a1_full_src4096_head30", "S8_a1_full_src8192"]
NICE = {"full": "full fine-tuning", "lorma": "LoRMA", "lora": "LoRA", "adapter": "Houlsby adapters",
        "qlora": "QLoRA (NF4)", "prefix": "prefix tuning", "adalora": "AdaLoRA", "prompt": "prompt tuning"}


def M(name, value):
    macros["".join(DIG.get(c, c) for c in name)] = value


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def summ(r, tag=""):
    return load(P6 / "runs" / r / f"summary{'_' + tag if tag else ''}.json")


def cfg(r):
    return yaml.safe_load(open(P6 / "runs" / r / "config.yaml"))


def arch(r):
    return cfg(r)["model"]["phase1_ckpt"].split("/")[1].split("_")[0]


def method(r):
    c = cfg(r)
    m = NICE[c["peft"]["method"]]
    if c.get("kd"):
        return f"KD student, 6 blocks ({m})"
    return m + (" + FiD" if c["model"].get("n_chunks") else "")


def source(r):
    c = cfg(r)["model"]
    return f"{c['n_chunks']}$\\times${c['src_len']}" if c.get("n_chunks") else f"{c['src_len']:,}"


def short(r):
    return r.split("_")[0]


VAL = load_splits()["val"]
REFS = [" ".join(d["summary"].split()) for d in VAL]
IDS = [d["id"] for d in VAL]


def preds(r, tag=""):
    d = {x["id"]: x["summary"] for x in load(P6 / "runs" / r / f"val_pred{'_' + tag if tag else ''}.json")}
    return [d[i] for i in IDS]


def mbr_val(cands):
    V = [mbr.load(*(c.split(":") if ":" in c else (c, "")), "val") for c in cands]
    return [V[mbr.pick([v[i] for v in V], [1.0] * len(V))][i] for i in IDS]


def per_doc_bertscore(hyps):
    import transformers
    transformers.logging.set_verbosity_error()
    from bert_score import score as bs
    return bs(hyps, REFS, lang="en", batch_size=32, verbose=False)[2].tolist()


# ------------------------------------------------------------------ data + cached computations
def cached():
    if CACHE.exists():
        return load(CACHE)
    out = {}
    docs = [d["document"] for d in VAL]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(REFS))
    while (perm == np.arange(len(REFS))).any():
        perm = rng.permutation(len(REFS))
    sents = [[s for s in d.replace("\n", " ").split(". ") if s.strip()] for d in docs]

    def spread(ss, n_words):
        k = max(1, int(round(n_words / max(1, np.mean([len(s.split()) for s in ss])))))
        idx = np.unique(np.linspace(0, len(ss) - 1, min(k, len(ss))).round().astype(int))
        return ". ".join(ss[i] for i in idx)
    base = {"another document's reference summary": [REFS[j] for j in perm],
            "first 400 words of the document": [" ".join(d.split()[:400]) for d in docs],
            "evenly spread sentences, ~450 words": [spread(s, 450) for s in sents]}
    out["baselines"] = {k: score(h, REFS) for k, h in base.items()}
    out["per_doc"] = {"S7_a1_full_src4096": per_doc_bertscore(preds("S7_a1_full_src4096")),
                      "MBR7_all12": per_doc_bertscore(mbr_val(load(P6 / "runs" / "mbr" / "MBR7_all12.json")["cands"]))}
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def data():
    import hashlib
    raw = load(P6.parent / "capstone_llm" / "data" / "hf" / "Phase_6_Summ" / "Train" / "train.json")
    h = {hashlib.md5(" ".join(d["document"].split()).encode()).hexdigest() for d in raw}
    sp = load_splits()
    M("ntrainraw", f"{len(raw):,}")
    M("nuniq", f"{len(h):,}")
    M("ntrain", f"{len(sp['train']):,}")
    M("nval", f"{len(sp['val']):,}")
    M("ntest", f"{len(sp['test']):,}")
    rows = []
    for name in ("train", "val", "test"):
        dl = np.array([len(d["document"].split()) for d in sp[name]])
        sl = np.array([len(d["summary"].split()) for d in sp[name]]) if name != "test" else None
        rows.append(f"{name} & {len(dl):,} & {np.median(dl):,.0f} & {np.percentile(dl, 90):,.0f} & "
                    + (f"{np.median(sl):,.0f} & {np.percentile(sl, 90):,.0f} & {np.median(dl / sl):.1f}" if sl is not None else "-- & -- & --")
                    + " \\\\")
    (TABS / "data.tex").write_text("\n".join(rows) + "\n")
    sl = np.array([len(r.split()) for r in REFS])
    M("medref", f"{np.median(sl):.0f}")
    M("refover", f"{100 * (sl > 400).mean():.0f}")


# ------------------------------------------------------------------ experiments
def row(r, tag="", label=None):
    s = summ(r, tag)
    v = s["val_metrics"]
    base = summ(r)
    vl = f"{base['val_loss']:.3f}" if base.get("val_loss") else "--"
    ep = cfg(r)["train"]["epochs"]
    init = cfg(r)["model"].get("init_run")
    return (f"{tex(label or short(r))} & {arch(r)} & {tex(method(r))} & {source(r)} & {ep}{'*' if init else ''} & "
            f"{base['trainable_params'] / 1e6:.2f}M & {vl} & {v['bertscore_f1']:.4f} & {v['rouge1_f1']:.3f} & "
            f"{v['rougeL_f1']:.3f} & {v['words']:.0f} \\\\")


def results():
    rows = [row(r, "rp" if (P6 / "runs" / r / "summary_rp.json").exists() else "") for r in FIRST] + ["\\midrule"]
    rows += [row(r) for r in SECOND] + ["\\midrule"] + [row(r) for r in LONG]
    (TABS / "results.tex").write_text("\n".join(rows) + "\n")
    for r in FIRST + SECOND + LONG:
        tag = "rp" if (P6 / "runs" / r / "summary_rp.json").exists() else ""
        M("bs" + short(r), f"{summ(r, tag)['val_metrics']['bertscore_f1']:.4f}")

    # decoding: default vs repetition control, per first-round run
    rows = []
    for r in FIRST:
        a, b = summ(r)["val_metrics"], summ(r, "rp")["val_metrics"]
        rows.append(f"{tex(short(r))} & {arch(r)} & {tex(method(r))} & {a['bertscore_f1']:.4f} & {b['bertscore_f1']:.4f} & "
                    f"{a['rouge1_f1']:.3f} & {b['rouge1_f1']:.3f} & {a['distinct_trigram']:.3f} & {b['distinct_trigram']:.3f} & "
                    f"{a['words']:.0f} & {b['words']:.0f} \\\\")
    (TABS / "decoding.tex").write_text("\n".join(rows) + "\n")
    gains = [summ(r, "rp")["val_metrics"]["bertscore_f1"] - summ(r)["val_metrics"]["bertscore_f1"] for r in FIRST]
    M("rpgainmin", f"{min(gains):+.4f}")
    M("rpgainmax", f"{max(gains):+.4f}")

    # length / beam / sampling variants
    rows = []
    for r, tag in (("F1_a4_full", ""), ("F1_a4_full", "short"), ("F1_a4_full", "long"), ("F1_a4_full", "rp"),
                   ("S3_a4_full_src4096", ""), ("S3_a4_full_src4096", "long"), ("S3_a4_full_src4096", "beam4"),
                   ("S7_a1_full_src4096", ""), ("S7_a1_full_src4096", "beam4"), ("S7_a1_full_src4096", "samp1"),
                   ("S7_a1_full_src4096", "samp2")):
        s_ = summ(r, tag)
        v, g = s_["val_metrics"], s_["gen"]
        dec = f"beam {g['beams']}" if g.get("beams", 1) > 1 else (f"sample $T{{=}}{g['temperature']}$" if g["temperature"] else "greedy")
        rows.append(f"{short(r)} & {g['min_new_tokens']}--{g['max_new_tokens']} & {g['repetition_penalty']} & {g['no_repeat_ngram']} & "
                    f"{dec} & {g['val_docs']} & {v['words']:.0f} & {v['bertscore_f1']:.4f} & {v['bertscore_p']:.4f} & "
                    f"{v['bertscore_r']:.4f} & {v['rouge1_f1']:.3f} & {v['rougeL_f1']:.3f} \\\\")
    (TABS / "length.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for r in ["F1_a4_full", "K1_kd_a4_6l", "E1_a4_lorma", "E2_a1_lora", "L2_a1_full_ep6", "S7_a1_full_src4096", "S4_a4_full_src8192", "S9_a3_full_src6144"]:
        s = summ(r)
        rows.append(f"{tex(short(r))} & {arch(r)} & {tex(method(r))} & {source(r)} & {s['total_params'] / 1e6:.1f}M & "
                    f"{s['trainable_params'] / 1e6:.2f}M & {s['train_min']:.0f} & {s['decode_s_per_doc']:.2f} & {s['peak_gb']:.1f} \\\\")
    (TABS / "efficiency.tex").write_text("\n".join(rows) + "\n")
    t, st = summ("F1_a4_full"), summ("K1_kd_a4_6l")
    M("kdparams", f"{st['total_params'] / 1e6:.1f}M")
    M("teacherparams", f"{t['total_params'] / 1e6:.1f}M")
    M("kdteacherrp", f"{summ('F1_a4_full', 'rp')['val_metrics']['bertscore_f1']:.4f}")

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for k, (runs, lab) in enumerate(((FIRST, "first round (1,280-token source)"), (SECOND + LONG, "later rounds"))):
        for r in runs:
            tag = "rp" if (P6 / "runs" / r / "summary_rp.json").exists() else ""
            x, y = summ(r)["trainable_params"] / 1e6, summ(r, tag)["val_metrics"]["bertscore_f1"]
            ax.plot(x, y, "o", ms=6, color=C[k], markeredgecolor="white", label=lab if r == runs[0] else None)
            if r in FIRST:
                ax.annotate(f"{short(r)} {arch(r)}", (x, y), xytext=(4, -3), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="trainable parameters (millions, log scale)", ylabel="val BERTScore F1")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(FIGS / "bs_vs_params.pdf")
    plt.close(fig)


def source_length():
    series = {"A4": ["F1_a4_full", "S1_a4_full_src2560", "S3_a4_full_src4096", "S4_a4_full_src8192"],
              "A1": ["L2_a1_full_ep6", "S6_a1_full_src2560", "S7_a1_full_src4096", "S8_a1_full_src8192"],
              "A3": ["L3_a3_full_ep6", "S5_a3_full_src4096", "S9_a3_full_src6144"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for k, (a, runs) in enumerate(series.items()):
        xs = [cfg(r)["model"]["src_len"] for r in runs]
        ys = [summ(r, "rp" if r == "F1_a4_full" else "")["val_metrics"]["bertscore_f1"] for r in runs]
        ls = [summ(r)["val_loss"] for r in runs]
        axes[0].plot(xs, ys, "o-", color=C[k], label=a)
        axes[1].plot(xs, ls, "o-", color=C[k], label=a)
    for ax, yl in zip(axes, ("val BERTScore F1", "val loss (summary tokens)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1280, 2560, 4096, 8192], ["1,280", "2,560", "4,096", "8,192"])
        ax.set(xlabel="source tokens", ylabel=yl)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "source_length.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ MBR + baselines + per-document
def mbr_tables(cache):
    names = ["MBR1_top4", "MBR2_long", "MBR4_s7samp", "MBR5_s7samp_beam", "MBR6_long8", "MBR7_all12"]
    rows = []
    for n in names:
        m = load(P6 / "runs" / "mbr" / f"{n}.json")
        v = m["val_metrics"]
        rows.append(f"{tex(n)} & {len(m['cands'])} & {m['n_val']} & {v['bertscore_f1']:.4f} & {v['rouge1_f1']:.3f} & "
                    f"{v['rougeL_f1']:.3f} & {v['words']:.0f} \\\\")
        M("bs" + n.split("_")[0].replace("MBR", "mbr"), f"{v['bertscore_f1']:.4f}")
    e = load(P6 / "runs" / "ens" / "ENS_t3.json")
    rows.append(f"ENS\\_t3 (token-level ensemble) & {len(e['runs'])} & {e['n_val']} & {e['val_metrics']['bertscore_f1']:.4f} & "
                f"{e['val_metrics']['rouge1_f1']:.3f} & {e['val_metrics']['rougeL_f1']:.3f} & {e['val_metrics']['words']:.0f} \\\\")
    (TABS / "mbr.tex").write_text("\n".join(rows) + "\n")
    m7 = load(P6 / "runs" / "mbr" / "MBR7_all12.json")
    (TABS / "mbr_share.tex").write_text("\n".join(f"{tex(k)} & {100 * v:.1f} \\\\" for k, v in
                                                  sorted(m7["picked_share"].items(), key=lambda x: -x[1])) + "\n")
    M("rougeonembrseven", f"{m7['val_metrics']['rouge1_f1']:.3f}")
    M("rougelmbrseven", f"{m7['val_metrics']['rougeL_f1']:.3f}")

    rows = []
    for k, v in cache["baselines"].items():
        rows.append(f"{k} & {v['words']:.0f} & {v['bertscore_f1']:.4f} & {v['rouge1_f1']:.3f} & {v['rougeL_f1']:.3f} \\\\")
    (TABS / "baselines.tex").write_text("\n".join(rows) + "\n")

    # per-document BERTScore vs reference length
    rl = np.array([len(r.split()) for r in REFS])
    dl = np.array([len(d["document"].split()) for d in VAL])
    f = np.array(cache["per_doc"]["MBR7_all12"])
    s7 = np.array(cache["per_doc"]["S7_a1_full_src4096"])
    M("corrref", f"{np.corrcoef(rl, f)[0, 1]:.2f}")
    M("corrdoc", f"{np.corrcoef(np.log(dl), f)[0, 1]:.2f}")
    M("mbrbetter", f"{100 * (f > s7).mean():.0f}")
    q = np.quantile(rl, [0, .25, .5, .75, 1])
    rows = []
    for k in range(4):
        m = (rl >= q[k]) & ((rl <= q[k + 1]) if k == 3 else (rl < q[k + 1]))
        rows.append(f"Q{k + 1} & {int(q[k])}--{int(q[k + 1])} & {m.sum()} & {f[m].mean():.4f} & {s7[m].mean():.4f} \\\\")
    (TABS / "perdoc.tex").write_text("\n".join(rows) + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.scatter(rl, f, s=9, color=C[0], edgecolor="white", linewidth=0.3)
    ax.axvline(400, color=INK2, ls="--", lw=0.8)
    ax.annotate("~512 BERTScore tokens", (400, f.max()), xytext=(3, -6), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="reference summary length (words, log scale)", ylabel="BERTScore F1 (MBR7)")
    fig.savefig(FIGS / "perdoc.pdf")
    plt.close(fig)


def curves():
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    for k, r in enumerate(["F1_a4_full", "E1_a4_lorma", "E2_a1_lora", "S6_a1_full_src2560", "S7_a1_full_src4096"]):
        recs = [json.loads(l) for l in open(P6 / "runs" / r / "metrics.jsonl") if '"loss"' in l]
        st = np.array([x["step"] for x in recs])
        lo = np.array([x["loss"] for x in recs])
        w = 9
        sm = np.convolve(lo, np.ones(w) / w, mode="valid") if len(lo) > w else lo
        ax.plot(st[len(st) - len(sm):], sm, color=C[k], lw=1.2, label=f"{short(r)} {arch(r)} {method(r)}")
    ax.set(xlabel="optimizer step", ylabel="training loss (smoothed)")
    ax.legend(fontsize=6.5)
    fig.savefig(FIGS / "curves.pdf")
    plt.close(fig)


def submissions():
    zips = sorted(p.name for p in (P6 / "submissions").glob("*.zip"))
    (TABS / "submissions.tex").write_text("\n".join(f"\\code{{{tex(z)}}} \\\\" for z in zips) + "\n")
    M("nzips", len(zips))


if __name__ == "__main__":
    cache = cached()
    data()
    results()
    source_length()
    mbr_tables(cache)
    curves()
    submissions()
    with open(HERE / "numbers.tex", "w") as f:
        f.write("% generated by make_figures.py\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(list(FIGS.glob('*.pdf')))} figures, {len(list(TABS.glob('*.tex')))} tables, {len(macros)} macros")
