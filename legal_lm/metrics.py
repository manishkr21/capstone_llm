"""Local evaluation that mirrors Codabench: format checks + BERTScore F1 (primary),
ROUGE-L F1 and BLEU-4 (secondary).  References come from our own held-out dev set."""
from __future__ import annotations

from typing import Dict, List, Optional


def check_format(pred: dict, prompts: dict, n_words: int = 128) -> List[str]:
    """Return a list of problems (empty list = the file is valid)."""
    problems = []
    missing = set(prompts) - set(pred)
    extra = set(pred) - set(prompts)
    if missing:
        problems.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    for key, value in pred.items():
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            problems.append(f"id {key}: value must be {{'text': str}}")
            continue
        words = value["text"].split()
        if not words:
            problems.append(f"id {key}: empty continuation")
        elif len(words) > n_words:
            problems.append(f"id {key}: {len(words)} words (> {n_words})")
        if key in prompts and len(words) > 20 and prompts[key]["text"][:200] in value["text"]:
            problems.append(f"id {key}: continuation seems to contain the prompt")
    return problems


def compute_metrics(pred: Dict[str, dict], refs: Dict[str, dict], bertscore_model: Optional[str] = None,
                    use_bertscore: bool = True, device: Optional[str] = None) -> dict:
    ids = [i for i in refs if i in pred]
    hyps = [pred[i]["text"] for i in ids]
    gold = [refs[i]["text"] for i in ids]
    results = {"n": len(ids)}

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results["rougeL_f1"] = sum(scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, gold)) / len(ids)

    import sacrebleu
    results["bleu4"] = sacrebleu.corpus_bleu(hyps, [gold]).score / 100.0

    if use_bertscore:
        from bert_score import score as bert_score
        kwargs = {"model_type": bertscore_model} if bertscore_model else {"lang": "en"}
        _, _, f1 = bert_score(hyps, gold, device=device, batch_size=32, verbose=False, **kwargs)
        results["bertscore_f1"] = float(f1.mean())
        results["bertscore_model"] = bertscore_model or "roberta-large (bert_score default for en)"
    return results
