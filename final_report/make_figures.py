"""Figures for the CS-781 final report: one new data figure, plus figures copied from the phase reports.

    ../cenv/bin/python make_figures.py      # -> figs/*.pdf

The new figure reads the Phase-5 token cache (statute labels), the Phase-6 training split (document and summary
lengths) and the Phase-7 queries (gold cases, citation markers).  No test label is read.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# figures reused from the phase reports (generated there from the run outputs)
COPY = {"p1_val_retrain.pdf": ROOT / "p1/report/figs/val_retrain.pdf",
        "p6_source_length.pdf": ROOT / "p6/report/figs/source_length.pdf",
        "p7_curves.pdf": ROOT / "p7/report/figs/curves.pdf",
        "p5_perlabel.pdf": ROOT / "p5/report/figs/perlabel.pdf"}


def data_figure():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.4))
    y = np.load(ROOT / "p5/cache/train.npz")["labels"]
    ax[0].bar(range(y.shape[1]), np.sort(y.sum(0))[::-1], color=C[0], width=1.0)
    ax[0].set_yscale("log")
    ax[0].set(xlabel="statute (sorted)", ylabel="training documents", title="(a) P5 statute frequency")
    ax[0].grid(axis="x", visible=False)

    sys.path.insert(0, str(ROOT / "p6"))
    from summ.data import load_splits
    tr = load_splits()["train"]
    dl = np.array([len(d["document"].split()) for d in tr])
    sl = np.array([len(d["summary"].split()) for d in tr])
    ax[1].scatter(dl, sl, s=2, color=C[1], alpha=0.35, linewidths=0)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="document words", ylabel="summary words", title="(b) P6 document vs. summary")

    q = json.load(open(ROOT / "capstone_llm/data/hf/Phase_7_PCR/Train/train.json"))
    g = np.array([len(v["relevant_candidates"]) for v in q.values()])
    c = np.array([" ".join(v["text"]).count("<CITATION>") for v in q.values()])
    ax[2].scatter(c, g, s=4, color=C[2], alpha=0.5, linewidths=0)
    ax[2].set_xscale("symlog")
    ax[2].set_yscale("symlog")
    ax[2].set(xlabel="<CITATION> markers", ylabel="gold cases", title=f"(c) P7 queries (r = {np.corrcoef(c, g)[0, 1]:.2f})")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGS / "data.pdf")
    plt.close(fig)


if __name__ == "__main__":
    data_figure()
    for name, src in COPY.items():
        shutil.copy(src, FIGS / name)
    print(sorted(p.name for p in FIGS.glob("*.pdf")))
