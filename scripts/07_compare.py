"""Step 7: collect every run into one comparison table and plot validation curves.

    python scripts/07_compare.py --runs-dir runs

Writes runs/comparison.md, runs/comparison.csv and runs/val_loss.png.
"""
import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from legal_lm.utils import human, load_json

# fixed categorical order (colour follows the run, never its rank)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def collect(runs_dir: Path):
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        s = load_json(summary_path)
        run_dir = summary_path.parent
        best_metrics = {}
        for m in run_dir.glob("predictions/dev_*/metrics.json"):
            r = load_json(m)
            key = r.get("bertscore_f1", r.get("rougeL_f1", 0))
            if not best_metrics or key > best_metrics["_key"]:
                best_metrics = {**r, "_key": key, "decoding": m.parent.name[len("dev_"):]}
        rows.append({
            "run": s["run_name"], "arch": s["arch"],
            "params": human(s["params_total"]), "non_emb": human(s["params_non_embedding"]),
            "tokens_seen": human(s["tokens_seen"]),
            "val_loss": round(s["best_val_loss"], 4),
            "val_ppl": round(s["best_val_ppl"], 2) if s.get("best_val_ppl") else None,
            "bertscore_f1": round(best_metrics["bertscore_f1"], 4) if "bertscore_f1" in best_metrics else None,
            "rougeL_f1": round(best_metrics["rougeL_f1"], 4) if best_metrics else None,
            "bleu4": round(best_metrics["bleu4"], 4) if best_metrics else None,
            "best_decoding": best_metrics.get("decoding"),
            "train_hours": (s.get("compute") or {}).get("monitored_hours"),
            "peak_vram_gb": (s.get("compute") or {}).get("peak_vram_gb"),
        })
    return rows


def plot(runs_dir: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ends = []  # (y, x, name) of each curve's last point, for direct labels
    for i, metrics_path in enumerate(sorted(runs_dir.glob("*/metrics.jsonl"))):
        evals = [json.loads(line) for line in open(metrics_path)]
        evals = [e for e in evals if e["type"] == "eval"]
        if not evals:
            continue
        x = [e["tokens_seen"] / 1e6 for e in evals]
        y = [e["val_loss"] for e in evals]
        color = COLORS[i % len(COLORS)]
        name = metrics_path.parent.name
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ends.append([y[-1], x[-1], name])
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)  # push end labels apart so they never overlap
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for label_y, x_end, name in ends:
        ax.annotate(name, (x_end, label_y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#333333", annotation_clip=False)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation continuation loss")
    ax.set_title("Validation loss by architecture (same data, tokenizer and budget)", fontsize=10)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"plot -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    runs_dir = Path(ap.parse_args().runs_dir)
    rows = collect(runs_dir)
    if not rows:
        raise SystemExit(f"no runs with summary.json under {runs_dir}")
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    (runs_dir / "comparison.md").write_text(table + "\n")
    with open(runs_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    plot(runs_dir, runs_dir / "val_loss.png")
