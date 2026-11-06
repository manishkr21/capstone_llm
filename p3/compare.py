"""Comparison of all Phase-3 runs (out-of-fold scores over the 90 train + dev documents).

    python compare.py        # -> runs/comparison.md, runs/comparison.csv
"""
import csv
import json

import yaml

from rr import P3_ROOT
from rr.data import load_split
import importlib.util

_spec = importlib.util.spec_from_file_location("p3_train", P3_ROOT / "train.py")  # not p2/train.py
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
aggregate = _train.aggregate

COLS = ["run", "arch", "method", "phase1_ckpt", "trainable", "trainable_%", "oof_macro_f1", "CL_f1", "IT_f1",
        "fold_mean", "fold_std", "macro_p", "macro_r", "min/fold", "peak_gb", "ckpt_mb", "infer_s"]


def main():
    data_dir = P3_ROOT / "../capstone_llm/data/hf/Phase_3_RR"
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    rows, labels = [], {}
    for run_dir in sorted((P3_ROOT / "runs").iterdir()):
        if not (run_dir / "config.yaml").exists():
            continue
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
        agg = aggregate(run_dir, docs, cfg["cv"]["n_folds"])
        if agg is None:
            continue
        arch = cfg["model"]["phase1_ckpt"].split("/")[1].split("_")[0]
        method = cfg["peft"]["method"] + (" (KD)" if cfg.get("distill") else "") + \
            ("" if cfg["data"]["context"] == "window" else ", sentence-only")
        rows.append([agg["run"], arch, method, cfg["model"]["phase1_ckpt"].split("/")[1],
                     f"{agg['trainable_params'] / 1e6:.2f}M", agg["trainable_pct"], agg["oof"]["all"]["macro_f1"],
                     agg["oof"]["CL"]["macro_f1"], agg["oof"]["IT"]["macro_f1"], agg["fold_macro_f1_mean"],
                     agg["fold_macro_f1_std"], agg["oof"]["all"]["macro_p"], agg["oof"]["all"]["macro_r"],
                     agg["train_min_per_fold"], agg["peak_gpu_gb"], agg["ckpt_mb"], agg["dev_infer_s"]])
        labels[agg["run"]] = agg["oof_per_label"]
    rows.sort(key=lambda r: -r[6])
    with open(P3_ROOT / "runs" / "comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows([COLS] + rows)
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    all_labels = sorted({l for v in labels.values() for l in v})
    lines += ["", "Out-of-fold F1 per label", "", "| run | " + " | ".join(all_labels) + " |",
              "|" + "---|" * (len(all_labels) + 1)]
    lines += ["| " + r[0] + " | " + " | ".join(f"{labels[r[0]].get(l, 0):.3f}" for l in all_labels) + " |" for r in rows]
    (P3_ROOT / "runs" / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
