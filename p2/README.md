# CS-781 Capstone, Phase 2: Legal Named Entity Recognition

Character-level legal NER (12 labels, strict span + label match, macro-F1) built on the
four Phase-1 language models in `../capstone_llm`. No other pretrained weights are used.

## Quick start

```bash
cd p2
bash run_all.sh                 # alignment check, train E1..E6, one zip per experiment
bash run_all.sh e3              # just one experiment (E6 needs E1: it is the teacher)
python scripts/compare.py       # dev table + per-label F1 -> runs/comparison.md
```

Each run writes `submissions/<run>_<date>_<time>.zip` (`prediction.json` at the zip root),
ready to upload to Codabench. Upload the six zips, one per experiment.

## The six experiments

| ID | Phase-1 model | Adaptation | Config |
|----|---------------|-----------|--------|
| E1 | A3 prefix-LM (bidirectional) | full fine-tuning | `configs/e1_a3_full.yaml` |
| E2 | A4 efficient decoder | **LoRMA+** (multiplicative low-rank) | `configs/e2_a4_lorma.yaml` |
| E3 | A1 GPT | LoRA (additive low-rank) | `configs/e3_a1_lora.yaml` |
| E4 | A2 encoder-decoder (encoder only) | Houlsby adapters | `configs/e4_a2_adapter.yaml` |
| E5 | A4 efficient decoder | partial / layer-wise FT (top 4 of 10 blocks) | `configs/e5_a4_partial.yaml` |
| E6 | A3, 4 of 8 blocks (student) | **knowledge distillation** from E1 | `configs/e6_kd_a3_small.yaml` |

All four Phase-1 architectures are used, five adaptation techniques plus KD.
`bitfit` and `head: mlp` / `infer.decode: argmax` / `train.o_weight` are also
implemented for extra experiments.

## Pipeline

1. **Tokenize with offsets** (`lner/data.py`). Phase-1 byte-level BPE, offsets with the
   leading space stripped. `presplit` splits punctuation runs (`).`, `],`) before BPE so
   more entity boundaries fall on token boundaries.
2. **BIO tags** (25 = O + B/I x 12) on tokens that overlap a gold span.
3. **Backbone** (`lner/backbones.py`). The Phase-1 model returns one vector per token:
   A2 uses only its encoder, A3 treats the whole window as the bidirectional prefix,
   A1/A4 run causal or with the causal mask removed (`model.attention`).
4. **Adaptation** (`lner/peft.py`): full, partial, BitFit, LoRA, LoRMA+, Houlsby adapters.
5. **Training** (`train.py`): 512-token windows (stride 256, random shift per epoch), CE
   loss (+ KL to the teacher for KD), dev evaluation every epoch, best checkpoint kept.
   Only trainable parameters are saved, so PEFT checkpoints are a few MB.
6. **Inference** (`lner/engine.py`): overlapping windows (stride 128). Each token keeps the
   logits of the window where it has the most context. Viterbi decoding allows only valid
   BIO sequences. Tokens are then mapped back to character spans with trailing whitespace trimmed.
7. **Scoring** (`lner/metrics.py`): strict macro P/R/F1 as on Codabench, plus micro-F1 and
   span-only F1 (boundary right, label ignored) for error analysis.

Dev split: 10 of the 70 training documents (`data.split_seed`), the same for every run.

## A2 note: encoder vs decoder states

Tagging the A2 **encoder** output gave only 0.11 dev macro-F1 (adapters), 0.13 (partial)
and 0.15 (full FT, `runs/diag_*`). The Phase-1 A2 encoder output is nearly collapsed
(mean cosine between token vectors 0.92, against 0.19 for A3); the decoder does most of the
language modelling. With `model.encdec_output: decoder` the decoder re-reads the window
while cross-attending to the encoder (BART-style tagging), and E4 rises to 0.39.

## Sanity check of offsets

`python scripts/check_alignment.py` encodes the gold spans as tags and decodes them back,
giving the best score any tagger can reach with this tokenization:

| presplit | oracle strict macro-F1 |
|---|---|
| false | 0.9866 |
| true  | 0.9945 |

## Useful commands

```bash
python train.py --config configs/e2_a4_lorma.yaml --set peft.r=8 run_name=E2_r8
python train.py --config configs/e1_a3_full.yaml --all-data      # train on all 70 docs
python predict.py --run runs/E1_a3_full --split dev              # per-label dev report
python predict.py --run runs/E1_a3_full --set infer.decode=argmax
```

## References

* LoRMA: Bihany, Patel, Modi (2025). *LoRMA: Low-Rank Multiplicative Adaptation for LLMs*, Findings of ACL. arXiv:2506.07621 (LoRMA+ variant: h = (I + (alpha/r)·BA)·W0·x, B = 0 at init).
* LoRA: Hu et al. (2022). Adapters: Houlsby et al. (2019). BitFit: Ben Zaken et al. (2022).
* Knowledge distillation: Hinton et al. (2015). Layer-subset student init: Sanh et al. (2019), DistilBERT.
