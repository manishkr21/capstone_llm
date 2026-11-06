"""Run tuning variants (one change against a base experiment) in parallel and compare.

    python scripts/sweep.py                 # every variant below, 6 at a time
    python scripts/sweep.py --only c1 c2    # a subset
    python scripts/sweep.py --report        # table only

Each variant = (name, base config, overrides). Runs go to sweeps/<name>/, logs to logs/sweeps/.
A finished variant (summary.json present) is skipped, so the script can be re-run safely.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
E1, E2, E3, E4, E5, E6 = (f"configs/{c}.yaml" for c in
                          ("e1_a3_full", "e2_a4_lorma", "e3_a1_lora", "e4_a2_adapter", "e5_a4_partial", "e6_kd_a3_small"))
BASE_RUN = {E1: "E1_a3_full", E2: "E2_a4_lorma", E3: "E3_a1_lora", E4: "E4_a2_adapter", E5: "E5_a4_partial",
            E6: "E6_kd_a3_small"}

VARIANTS = [
    # --- (1) data, context, class imbalance
    ("a1_e2_win1024", E2, ["train.window=1024", "infer.window=1024", "train.stride=512", "train.batch_size=8"]),
    ("a2_e1_win1024", E1, ["train.window=1024", "infer.window=1024", "train.stride=512", "train.batch_size=8"]),
    ("a3_e2_oweight0.5", E2, ["train.o_weight=0.5"]),
    ("a4_e1_oweight0.5", E1, ["train.o_weight=0.5"]),
    ("a5_e2_focal", E2, ["train.loss=focal", "train.focal_gamma=2.0"]),
    ("a6_e2_invsqrt", E2, ["train.label_weight=inv_sqrt"]),
    ("a7_e2_casenorm", E2, ["data.case_norm=true"]),
    ("a8_e1_casenorm", E1, ["data.case_norm=true"]),
    ("a9_e2_nopresplit", E2, ["data.presplit=false"]),
    # --- (2) heads
    ("b1_e2_mlphead", E2, ["model.head=mlp"]),
    ("b2_e2_crf", E2, ["model.head=crf"]),
    ("b3_e1_crf", E1, ["model.head=crf"]),
    # --- (3) adaptation
    ("c1_a4_lora", E2, ["peft.method=lora"]),
    ("c2_a1_lorma", E3, ["peft.method=lorma"]),
    ("c3_a3_lorma", E1, ["peft.method=lorma", "peft.r=16", "peft.alpha=32", "train.lr=1e-3"]),
    ("c4_a3_lora", E1, ["peft.method=lora", "peft.r=16", "peft.alpha=32", "train.lr=1e-3"]),
    ("c5_e2_attnonly", E2, ["peft.targets=[q,k,v,o]"]),
    ("c6_e2_r8", E2, ["peft.r=8", "peft.alpha=16"]),
    ("c7_e2_r32", E2, ["peft.r=32", "peft.alpha=64"]),
    ("c8_a4_lorma_pi", E2, ["peft.method=lorma_pi"]),
    ("c9_e3_causal", E3, ["model.attention=causal"]),
    ("c10_e2_causal", E2, ["model.attention=causal"]),
    ("c11_e1_llrd0.8", E1, ["train.llrd=0.8"]),
    ("c12_e1_regular", E1, ["model.dropout=0.2", "train.weight_decay=0.1"]),
    ("c13_a4_ia3", E2, ["peft.method=ia3", "train.lr=3e-3"]),
    ("c14_a3_bitfit", E1, ["peft.method=bitfit", "train.lr=1e-3"]),
    ("c15_a2_full_dec", E4, ["peft.method=full", "train.lr=1e-4"]),
    ("c16_a2_lorma_dec", E4, ["peft.method=lorma", "peft.r=16", "peft.alpha=32"]),
    ("c17_e5_top6", E5, ["peft.n_train_layers=6"]),
    ("c18_e5_top2", E5, ["peft.n_train_layers=2"]),
    ("c19_a4_full", E5, ["peft.method=full", "train.lr=1e-4"]),
    ("c20_e2_ep25", E2, ["train.epochs=25", "train.patience=8"]),
    ("c21_e2_lr5e-4", E2, ["train.lr=5e-4"]),
    ("c22_e2_lr2e-3", E2, ["train.lr=2e-3"]),
    # --- (4) distillation
    ("d1_e6_noKD", E6, ["distill=null"]),
    ("d2_e6_a0.3_T4", E6, ["distill.alpha=0.3", "distill.temperature=4.0"]),
    ("d3_e6_teacherE2", E6, ["distill.teacher_run=runs/E2_a4_lorma"]),
]

# --- (5) confirmation: the strongest candidates on 3 different dev splits (10 docs each)
CANDIDATES = {
    "E2": (E2, []),
    "E2invsqrt": (E2, ["train.label_weight=inv_sqrt"]),
    "E2focal": (E2, ["train.loss=focal"]),
    "A4lora": (E2, ["peft.method=lora"]),
    "A4lora_invsqrt": (E2, ["peft.method=lora", "train.label_weight=inv_sqrt"]),
    "E2invsqrt_r32": (E2, ["train.label_weight=inv_sqrt", "peft.r=32", "peft.alpha=64"]),
}
for cand, (cfg, sets) in CANDIDATES.items():
    for seed in (0, 1, 2):
        VARIANTS.append((f"f_{cand}_s{seed}", cfg, sets + [f"data.split_seed={seed}"]))


def summary(path):
    f = Path(path) / "summary.json"
    return json.load(open(f)) if f.exists() else None


def report():
    rows = []
    for name, base, _ in VARIANTS:
        s, b = summary(ROOT / "sweeps" / name), summary(ROOT / "runs" / BASE_RUN[base])
        if not s or "dev" not in s:
            continue
        rows.append((name, BASE_RUN[base], b["dev"]["macro_f1"], s["dev"]["macro_f1"], s["dev"]["span_only_f1"],
                     s["trainable_params"], s["best_epoch"], s["train_minutes"]))
    print(f"{'variant':22s} {'base':15s} {'base F1':>8s} {'F1':>7s} {'delta':>7s} {'span':>6s} {'train':>9s} ep  min")
    for r in sorted(rows, key=lambda r: -r[3]):
        print(f"{r[0]:22s} {r[1]:15s} {r[2]:8.4f} {r[3]:7.4f} {r[3] - r[2]:+7.4f} {r[4]:6.3f} {r[5] / 1e6:8.2f}M {r[6]:2d} {r[7]:4.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    todo = [v for v in VARIANTS if (not args.only or v[0].split("_")[0] in args.only or v[0] in args.only)
            and not summary(ROOT / "sweeps" / v[0])]
    (ROOT / "logs" / "sweeps").mkdir(parents=True, exist_ok=True)
    running = []
    while todo or running:
        while todo and len(running) < args.parallel:
            name, base, sets = todo.pop(0)
            cmd = [PY, "train.py", "--config", base, "--set", "runs_dir=sweeps", f"run_name={name}", *sets]
            log = open(ROOT / "logs" / "sweeps" / f"{name}.log", "w")
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
            print(f"started {name}", flush=True)
        for name, p in list(running):
            if p.poll() is not None:
                running.remove((name, p))
                s = summary(ROOT / "sweeps" / name)
                res = f"dev macro-F1 {s['dev']['macro_f1']:.4f}" if s and "dev" in s else f"FAILED (exit {p.returncode})"
                print(f"done {name}: {res}", flush=True)
        time.sleep(5)
    report()


if __name__ == "__main__":
    main()
