"""Step 6: score predictions on the local dev set (BERTScore F1, ROUGE-L F1, BLEU-4).

    python scripts/06_evaluate.py --pred runs/A1_gpt/predictions/dev_t0.8_k50_p0.95_rp1.1/prediction.json
    python scripts/06_evaluate.py --pred ... --no-bertscore            # fast, no model download
    python scripts/06_evaluate.py --pred ... --bertscore-model roberta-base
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from legal_lm.metrics import check_format, compute_metrics
from legal_lm.monitor import ResourceMonitor
from legal_lm.utils import load_json, pick_device, save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--prompts", default="data/dev/dev_prompts.json")
    ap.add_argument("--refs", default="data/dev/dev_references.json")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default=None, help="default: roberta-large (bert_score default)")
    args = ap.parse_args()

    pred, prompts, refs = load_json(args.pred), load_json(args.prompts), load_json(args.refs)
    problems = check_format(pred, prompts)
    print("format check:", "OK" if not problems else problems[:10])
    log = Path(args.pred).parent / "resources_eval.csv"
    with ResourceMonitor(log, pick_device(), phase="evaluate"):  # BERTScore runs on the GPU if present
        results = compute_metrics(pred, refs, args.bertscore_model, not args.no_bertscore)
    results["format_problems"] = len(problems)
    print(json.dumps(results, indent=2))
    save_json(results, Path(args.pred).parent / "metrics.json")
