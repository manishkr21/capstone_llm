"""Step 5: generate 128-word continuations and package submission_<date>_<time>.zip.

    python scripts/05_generate.py --run runs/A1_gpt --split dev
    python scripts/05_generate.py --run runs/A1_gpt --split test --set temperature=0.7 top_p=0.9

Outputs go to runs/<run>/predictions/<split>_<decoding-tag>/{prediction.json, submission_<date>_<time>.zip},
plus resources.csv (per-minute RAM/VRAM/CPU usage while generating).
"""
import argparse
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from legal_lm.generation import generate_predictions, write_submission
from legal_lm.metrics import check_format
from legal_lm.monitor import monitor_from_config
from legal_lm.trainer import load_checkpoint
from legal_lm.utils import load_json, pick_device, save_json


def decoding_tag(g):
    if g["temperature"] <= 0:
        return f"greedy_rp{g['repetition_penalty']}"
    return f"t{g['temperature']}_k{g['top_k']}_p{g['top_p']}_rp{g['repetition_penalty']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run folder, e.g. runs/A1_gpt")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (quick check)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="generation overrides")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg, tok = load_checkpoint(Path(args.run) / args.ckpt, device)
    g = dict(cfg["generation"])
    for item in args.set:
        k, _, v = item.partition("=")
        g[k.replace("generation.", "")] = yaml.safe_load(v)

    data_dir = Path(cfg["paths"]["data_dir"])
    prompt_file = data_dir / ("dev/dev_prompts.json" if args.split == "dev" else "test/test.json")
    prompts = load_json(prompt_file)
    if args.limit:
        prompts = dict(list(prompts.items())[: args.limit])
    print(f"{len(prompts)} prompts from {prompt_file}; decoding {g}")

    out_dir = Path(args.run) / "predictions" / f"{args.split}_{decoding_tag(g)}"
    with monitor_from_config(cfg, out_dir / "resources.csv", device, phase=f"generate_{args.split}"):
        preds = generate_predictions(model, tok, prompts, g, device)
    zip_path = write_submission(preds, out_dir)
    save_json({"generation": g, "checkpoint": str(Path(args.run) / args.ckpt), "split": args.split},
              out_dir / "generation_config.json")
    problems = check_format(preds, prompts, g["target_words"])
    print("format check:", "OK" if not problems else problems[:10])
    print(f"wrote {out_dir / 'prediction.json'} and {zip_path}")
