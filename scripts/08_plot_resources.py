"""Step 8: turn the per-minute resource logs into compute graphs and a compute table.

    python scripts/08_plot_resources.py --runs-dir runs                 # every runs/*/resources.csv
    python scripts/08_plot_resources.py --csv data/logs/resources_03_tokenize.csv

For each log it writes <name>_usage.png next to the CSV: four panels (CPU %, RAM GB,
GPU util %, VRAM GB) over wall-clock time, with resumed sessions joined end to end
and dashed lines at each resume. With --runs-dir it also writes runs/compute_summary.csv
/.md (GPU-hours, peaks, mean utilisation per run) and runs/compute_comparison.png.
"""
import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from legal_lm.monitor import read_resources, summarize_resources

# same fixed categorical order as 07_compare.py
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRID, INK = "#e5e5e5", "#333333"

PANELS = [  # (title, y label, [(column, legend label)], fixed y-max, capacity column)
    ("CPU utilisation", "% of all cores", [("cpu_pct", "machine"), ("cpu_pct_max", "machine, minute max")], 100, None),
    ("RAM", "GB", [("ram_used_gb", "machine used"), ("proc_rss_gb", "this process")], None, "ram_total_gb"),
    ("GPU utilisation", "% busy", [("gpu_util_pct", "mean"), ("gpu_util_pct_max", "minute max")], 100, None),
    ("VRAM", "GB", [("vram_used_gb", "device used"), ("torch_reserved_gb", "torch reserved"),
                    ("torch_alloc_gb", "torch allocated")], None, "vram_total_gb"),
]


def timeline(rows):
    """Minutes since the first session started, with later sessions appended end to end."""
    xs, boundaries, offset, last_session, last_x = [], [], 0.0, None, 0.0
    for r in rows:
        if r["session"] != last_session and last_session is not None:
            offset = last_x
            boundaries.append(offset)
        last_session = r["session"]
        last_x = offset + (r["elapsed_min"] or 0)
        xs.append(last_x)
    return xs, boundaries


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)


def plot_usage(csv_path: Path, title: str, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_resources(csv_path)
    if not rows:
        return None
    xs, boundaries = timeline(rows)
    hours = xs[-1] > 180
    xs = [x / 60 for x in xs] if hours else xs
    boundaries = [b / 60 for b in boundaries] if hours else boundaries

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150, sharex=True)
    for ax, (name, ylabel, series, ymax, cap_col) in zip(axes.flat, PANELS):
        drawn = 0
        for k, (col, label) in enumerate(series):
            pts = [(x, r[col]) for x, r in zip(xs, rows) if r.get(col) is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), color=COLORS[k], linewidth=2 if k == 0 else 1.5,
                    marker="o" if len(pts) < 30 else None, markersize=3, label=label)
            drawn += 1
        for b in boundaries:
            ax.axvline(b, color="#999999", linestyle="--", linewidth=0.8)
        if drawn == 0:
            ax.text(0.5, 0.5, "not available on this machine", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888888")
        elif drawn > 1:
            ax.legend(frameon=False, fontsize=7, loc="best")
        total = next((r[cap_col] for r in rows if cap_col and r.get(cap_col)), None)
        ax.set_ylim(0, ymax or (total * 1.12 if total else None))
        if total:
            ax.axhline(total, color=INK, linewidth=0.8, linestyle=":")
            ax.annotate(f"capacity {total:.1f} GB", (1, total), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                        fontsize=7, color=INK)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=8, color=INK)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("wall-clock " + ("hours" if hours else "minutes"), fontsize=8, color=INK)
    gpu = next((r["gpu_name"] for r in rows if r.get("gpu_name")), "no GPU")
    fig.suptitle(f"{title}  ({gpu}; one point per logged interval"
                 f"{'; dashed = resumed session' if boundaries else ''})", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_comparison(summaries: dict, out_png: Path):
    """One bar chart per measure (never two scales on one axis); colour follows the run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = [("monitored_hours", "wall-clock hours"), ("peak_vram_gb", "peak VRAM (GB)"),
                ("mean_gpu_util_pct", "mean GPU util (%)"), ("peak_ram_gb", "peak RAM (GB)")]
    names = list(summaries)
    fig, axes = plt.subplots(1, len(measures), figsize=(3 * len(measures), 3.2), dpi=150)
    for ax, (key, label) in zip(axes, measures):
        vals = [summaries[n].get(key) or 0 for n in names]
        bars = ax.barh(names, vals, color=[COLORS[i % len(COLORS)] for i in range(len(names))], height=0.6)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.1f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        style(ax)
        ax.margins(x=0.2)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Compute used per run (all sessions)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--csv", nargs="*", default=None, help="plot these resource logs instead of a runs folder")
    args = ap.parse_args()

    if args.csv:
        for path in map(Path, args.csv):
            out = plot_usage(path, path.stem, path.with_name(path.stem + "_usage.png"))
            print(f"{path} -> {out}" if out else f"{path}: empty")
        raise SystemExit(0)

    runs_dir = Path(args.runs_dir)
    logs = sorted(runs_dir.glob("*/resources.csv"))
    if not logs:
        raise SystemExit(f"no resources.csv under {runs_dir}/*/ (train a model first)")
    summaries = {}
    for path in logs:
        out = plot_usage(path, f"{path.parent.name}: training", path.with_name("resources_usage.png"))
        if out:
            summaries[path.parent.name] = summarize_resources(path)
            print(f"plot -> {out}")

    cols = ["run"] + list(next(iter(summaries.values())))
    table = [{"run": n, **s} for n, s in summaries.items()]
    with open(runs_dir / "compute_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |" for r in table]
    (runs_dir / "compute_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    plot_comparison(summaries, runs_dir / "compute_comparison.png")
    print(f"plot -> {runs_dir / 'compute_comparison.png'}")
