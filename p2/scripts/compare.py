"""Table of all runs (dev scores, parameters, compute) -> runs/comparison.md and .csv

    python scripts/compare.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((ROOT / "runs").glob("*/summary.json")):
    s = json.load(open(f))
    dev = s.get("dev", {})
    rows.append({
        "run": s["run"], "arch": s["arch"], "method": s["method"] + (" (KD)" if s.get("distilled_from") else ""),
        "total_params": f"{s['total_params'] / 1e6:.1f}M", "trainable": f"{s['trainable_params'] / 1e6:.2f}M",
        "trainable_%": s["trainable_pct"], "macro_f1": dev.get("macro_f1"), "macro_p": dev.get("macro_p"),
        "macro_r": dev.get("macro_r"), "micro_f1": dev.get("micro_f1"), "span_only_f1": dev.get("span_only_f1"),
        "best_epoch": s["best_epoch"], "train_min": s["train_minutes"], "tok_per_s": s["train_tokens_per_s"],
        "peak_gpu_gb": s["peak_gpu_gb"], "ckpt_mb": s["checkpoint_mb"], "dev_infer_s": s.get("dev_infer_s"),
    })
if not rows:
    sys.exit("no runs/*/summary.json yet")
cols = list(rows[0])
with open(ROOT / "runs" / "comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
(ROOT / "runs" / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# per-label dev F1 for error analysis
labels = list(json.load(open(next((ROOT / "runs").glob("*/summary.json"))))["dev"]["per_label"])
print("\nper-label dev F1")
print("run".ljust(22) + "".join(l[:9].rjust(10) for l in labels))
for f in sorted((ROOT / "runs").glob("*/summary.json")):
    s = json.load(open(f))
    if "dev" in s:
        print(s["run"][:21].ljust(22) + "".join(f"{s['dev']['per_label'][l]['f1']:10.3f}" for l in labels))
