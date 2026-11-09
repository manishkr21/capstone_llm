# CS-781 Capstone: In-House Legal Intelligence System

A legal NLP stack built from scratch under a confidentiality constraint: no external APIs, no pretrained
weights, no external corpora. Four compact legal language models (82–89M parameters) are pre-trained on the
official Indian legal corpus (Phase 1) and then adapted to six downstream tasks (Phases 2–7) with full
fine-tuning, parameter-efficient methods and knowledge distillation. Data: `Exploration-Lab/CS-781-Capstone`
only.

Final report: [`final_report/CS781-CP-Manish-Kumar-251330604-report.pdf`](final_report/). Each phase also has its
own detailed report in `pN/report/report.pdf`, generated from the run outputs by `pN/report/make_figures.py`.

## Results

Best Codabench score per phase, with leaderboard rank as recorded on 6 November 2026.

| Phase | Task | Metric | Best submission | Codabench | Rank |
|---|---|---|---|---:|---:|
| 1 | Legal LM pre-training | BERTScore F1 | A1 GPT (R0) | 0.8286 | 95 |
| 2 | Legal NER | strict macro-F1 | E2: A4 + LoRMA | 0.4970 | 80 |
| 3 | Rhetorical roles | macro-F1 | T14: A1 + LoRA, 1,024-token windows | 0.5750 | 15 |
| 4 | Judgment prediction | macro-F1 | STACK_with_fusion | 0.8121 | 13 |
| 5 | Statute identification | macro-F1 | STACK11_full (hybrid) | 0.4624 | 6 |
| 6 | Summarization | BERTScore F1 | S3: A4 full, 4,096-token source, beam 4 | 0.8558 | 4 |
| 7 | Prior case retrieval | micro-F1@7 | STK4_rc_blend4 | 0.4950 | — |

## Repository layout

```
capstone_llm/                 Phase 1 (this folder): legal_lm package, configs, scripts, runs
  legal_lm/                   tokenizer, data pipeline, four model architectures, trainer, generation, metrics
  configs/ scripts/ tests/    Phase-1 configs, numbered pipeline scripts, model tests
  p1/                         Phase-1 starter notebook (GPT-2-tokenizer baseline) and Phase-1 report
  p2/ ... p7/                 one folder per downstream phase (package, train.py, configs/, report/)
  final_report/               final LaTeX report (template format), figures, bibliography
```

Every downstream phase follows the same pattern:

```
pN/
  <pkg>/            data loading, model, phase-specific helpers   (lner, rr, cj, lsi, summ, pcr)
  configs/          base.yaml + one YAML per experiment (inherit: base.yaml, overrides via --set)
  train.py          train one experiment -> runs/<name>/{config.yaml, metrics.jsonl, best.pt, summary.json}
  run_queue.sh      train several configs in parallel on one GPU
  submissions/      Codabench zips (prediction.json at the root)
  report/           make_figures.py -> figs/, tables/, numbers.tex -> report.pdf
```

The phase packages re-use each other: `p2/lner` provides the backbone wrapper and LoRA / LoRMA / IA³ / adapters;
`p3/rr` adds QLoRA, rsLoRA, AdaLoRA, prefix and prompt tuning; `p4/cj` the chunk encoder.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login          # the dataset is gated: accept its terms on the Hub first
```

**Folder layout for Phases 2–7.** The phase packages locate the Phase-1 code as a sibling folder named
`capstone_llm` (`P1_ROOT = <phase folder>/../capstone_llm`) and each other as siblings (`../p2`, `../p3`, ...).
After cloning, recreate that layout once:

```bash
mkdir workspace && cd workspace
git clone git@github.com:manishkr21/capstone_llm.git capstone_llm
for p in p2 p3 p4 p5 p6 p7; do ln -s capstone_llm/$p $p; done
cd p5 && python train.py --config configs/e1_a4_lorma.yaml      # example
```

Phase-1 checkpoints are read from `capstone_llm/runs/<Arch>_clean_1B/best.pt` (Phase 2 also uses the original
`runs/<Arch>/best.pt`). Model weights and datasets are not stored in git.

## Phase 1: legal language model pre-training

Continue a 512-word legal prompt with 128 words (BERTScore F1; ROUGE-L, BLEU-4). Four architectures are compared
under one controlled setup (same tokenizer, data, token budget, schedule and evaluation):

| ID | Architecture | Key design | Params |
|----|--------------|-----------|--------:|
| A1 | Decoder-only GPT | learned positions, LayerNorm, GELU MLP; 8 layers, d=768 | 82.1M |
| A2 | Encoder–decoder | 6+6 layers, cross-attention, sinusoidal positions; d=640 | 89.4M |
| A3 | Prefix-LM | one stack, bidirectional over the prompt, causal over the continuation, ALiBi | 81.3M |
| A4 | Efficient decoder | RoPE, RMSNorm, SwiGLU, grouped-query attention, 256-token sliding window + global layers | 87.5M |

* **Tokenizer:** byte-level BPE, 32k vocabulary trained on legal text (`scripts/02_train_tokenizer.py`).
* **Data, round R0:** a 200M-token sample of the four sources (Acts, High Courts, Supreme Court, Tribunals).
* **Corpus audit (`scripts/11_audit_corpus.py`):** found validation documents copied into training, duplicates,
  OCR word stutter and encoding residue.
* **Cleaned corpus (`legal_lm/data/clean.py`, `run_clean.sh`):** whole-file sampling, normalisation, stutter
  repair, quality filters, exact + MinHash near-duplicate removal, and a leak-free validation split. Models were
  retrained at 200M tokens (R1) and continued to 1B tokens (R2). Fair-validation perplexity of A4:
  13.32 (R0) → 11.25 (R1) → 8.26 (R2).
* **Training:** AdamW, peak LR 6e-4, cosine schedule, 65,536 tokens per step, bf16; the checkpoint with the best
  validation loss is kept (`best.pt`).
* **Decoding:** temperature 0.8, top-k 50, top-p 0.95, repetition penalty 1.1, end-of-text banned before 128 words.
* **Submissions:** the R0 models (A1 0.8286, A3 0.8285, A4 0.8280, A2 0.8184 on Codabench); the R1/R2 models were
  trained after the Phase-1 deadline and are the backbones of Phases 3–7.

```bash
bash run_smoke_test.sh                                # whole pipeline on toy data, no dataset needed
python scripts/01_prepare_data.py   --config configs/base.yaml
python scripts/02_train_tokenizer.py --config configs/base.yaml
python scripts/03_tokenize.py        --config configs/base.yaml
python scripts/04_train.py --config configs/a4_efficient.yaml      # a1_gpt / a2_encdec / a3_prefixlm
python scripts/05_generate.py --run runs/A4_efficient --split test # -> submission_<date>_<time>.zip
bash run_clean.sh                                     # cleaned corpus + R1/R2 retraining
```

<details><summary>Phase-1 implementation notes</summary>

* `model(idx, prefix_len)` returns logits for `idx[:, prefix_len:]`; decoder-only models train with
  `prefix_len = 1`, A2/A3 sample a random prefix. Validation scores the 257 tokens after a 768-token prefix,
  so the losses compare directly across architectures.
* Generation pre-fills the prompt into a KV cache and decodes token by token. Only the continuation is written.
* `legal_lm/monitor.py` logs CPU, RAM, GPU and VRAM usage every minute to `resources.csv` for every stage;
  `scripts/08_plot_resources.py` turns these logs into plots and compute tables.
* Every run folder holds `config.yaml`, `metrics.jsonl`, `resources.csv`, `summary.json`, `best.pt` and `last.pt`.
  Training resumes automatically from `last.pt`.
</details>

## Shared approach for Phases 2–7

* **Backbone wrapper (`p2/lner/backbones.py`):** exposes A1–A4 as encoders. A1 and A4 keep their causal mask
  (better than bidirectional from Phase 3 on), A3 reads a bidirectional prefix, and A2 contributes decoder
  states (its Phase-1 encoder output is nearly collapsed).
* **Adaptation techniques:** full and partial fine-tuning, gradual unfreezing, BitFit, LoRA, rsLoRA, **LoRMA**
  (W = (I + α/r·BA)W₀), QLoRA (NF4 base), AdaLoRA, IA³, Houlsby adapters, prefix tuning, prompt tuning.
* **Knowledge distillation:** the student keeps a subset of the teacher's blocks and is trained with
  α·task loss + (1−α)·T²·KL to a teacher run or an ensemble of runs.
* **Integrity checks before training:** de-duplication, removing development documents that also appear in the
  test set, never reading test labels, and a schema check on every zip before upload.

## Phase 2: legal NER (`p2/`)

* **Task:** 12 entity types as exact character spans; strict macro-F1.
* **Approach:** punctuation pre-split → BPE with an offset map → character spans to BIO tags → backbone over
  512-token windows (stride 128 at inference) → linear tag head → BIO-constrained Viterbi → character spans.
  The span round trip recovers 99.45% of gold spans.
* **Experiments:** E1 A3 full, **E2 A4 LoRMA (best, 0.4970)**, E3 A1 LoRA, E4 A2 adapters, E5 A4 partial,
  E6 KD into a 4-block A3 student; a sweep of 37 one-change variants and a decoding sweep.
* **Key files:** `lner/backbones.py`, `lner/peft.py`, `lner/decoding.py`, `train.py`, `predict.py`, `run_all.sh`.

## Phase 3: rhetorical role prediction (`p3/`)

* **Task:** one of 13 roles per sentence in Corporate Law and IT judgments; macro-F1.
* **Approach:** neighbouring sentences packed into backbone windows → mean-pool per sentence → BiLSTM with
  position and domain embeddings → sentence-level CRF.
  * Logit-adjusted loss for rare roles.
  * 5-fold document-level cross-validation over the 90 labelled documents; test emissions averaged over the
    fold models.
* **Experiments:** nine techniques across A1–A4, then tuning rounds: causal mask, 20 epochs, τ = 0.5, longer
  windows, lexical cue flags, ensembles and calibration (`ensemble_cv.py`, `calibrate.py`).
* **Best:** T14 (A1 LoRA, 1,024-token windows) 0.5750; ensembles of 4–8 taggers 0.570.
* **Key files:** `rr/model.py`, `rr/peft_extra.py`, `train.py`, `make_submission.py`.

## Phase 4: court judgment prediction (`p4/`)

* **Task:** allowed (1) or dismissed (0); macro-F1.
* **Data integrity:** `test.json` contains labels, and every test document is also in `dev.json`. Labels are
  never read; selection uses only the 1,988 development documents that are not in the test set.
* **Approach:** last 8 (or 16) chunks of 512 tokens → backbone → mean-pool → BiGRU + attention pooling →
  2 classes; class-weighted loss; threshold tuned on the clean validation set.
* **Experiments:**
  * Seven techniques plus KD (E1–E8).
  * Chunk-selection ablations: the last chunks beat the first (0.785 vs 0.754).
  * V1–V6 variants.
  * Document models on cached chunk vectors (`embed_chunks.py`, `doc_train.py`).
  * Mean ensembles, and a logistic-regression stacker over all runs (`stack_submission.py`).
* **Best:** STACK_with_fusion (26 runs, including 9 fused-encoder document models) 0.8121.

## Phase 5: legal statute identification (`p5/`)

* **Task:** a set of statute ids (out of 100) per document, with statute names masked in the text; macro-F1.
* **Approach:** 8 spread chunks → backbone → BiGRU → one attention query per statute → 100 sigmoids.
  * BCE with √-frequency positive weights.
  * Per-statute thresholds, reported cross-fitted (fitted on one half of validation, scored on the other).
* **Experiments:**
  * Ten first-round techniques on all four architectures (`configs/e*.yaml`).
  * Longer training, then KD students distilled from growing teacher ensembles (`dump_logits.py`).
* **Stacking:** MLP (`stack.py`) and per-statute LightGBM (`stack_gbm.py`) stackers trained out-of-fold on
  validation. Hybrid stacks add TF-IDF models (`tfidf_lr.py`), which use no pretrained weights.
* **Best:**
  * STACK11_full (hybrid) 0.4624.
  * Phase-1-only stack STACK6 0.4231.
  * Best single model: Y2, a 6-block KD student.

## Phase 6: abstractive summarization (`p6/`)

* **Task:** summary of a judgment; BERTScore F1 (ROUGE-1, ROUGE-L).
* **Approach:** salient condensation of the judgment (case opening + sentences closest to the TF-IDF centroid)
  → `[prompt][source] Summary:` → decoder-only Phase-1 LM fine-tuned on the summary tokens. A2 uses
  fusion-in-decoder.
  * Long sources go up to 8,192 tokens: A1's position table is interpolated, while RoPE and ALiBi extrapolate.
  * Each longer-source run continues from the shorter-source checkpoint.
* **Decoding:** greedy or beam search with a KV cache, repetition penalty 1.15, no repeated 3-grams.
  `mbr.py` picks the consensus candidate across runs; `ensemble.py` does token-level ensembling.
* **Experiments:** eight techniques plus KD (F1/F2, E1–E7, K1), longer training (L*), and source length
  1,280 → 8,192 (S1–S9). Re-decoding with other settings: `redecode.sh`.
* **Best:** S3 (A4 full, 4,096-token source, beam 4) 0.8558.

## Phase 7: prior case retrieval (`p7/`)

* **Task:** 7 ranked candidate ids per query from a split-specific pool; micro-F1@7.
* **Data structure used:**
  * Citations are masked as `<CITATION>`.
  * Every query also sits in its own pool, and is always excluded from its own ranking.
  * A pool is essentially the union of its queries' gold cases.
* **Lexical stage (`pcr/lexical.py`, `lex.py`, `lex_wincol.py`):** one BM25 query per citation window
  (±60 words), a date penalty, and per-case (column) and per-window normalisation across the split.
* **Neural stage (`pcr/model.py`, `train.py`):** a citation-window bi-encoder on a Phase-1 backbone.
  * Max over a case's chunks, soft-max over the query's windows.
  * Multi-positive InfoNCE with BM25 hard negatives.
  * Seven techniques + KD (E1–E7, K0/K1).
* **Re-ranking (`stack.py`, `pcr/ltr.py`):** LightGBM lambdarank over the top-100 lexical candidates, plus the
  mean score of several bi-encoders. A cross-encoder (`cross_train.py`) was tried and did not generalise.
* **Best:** STK4_rc_blend4 0.4950 on Codabench; STK6_wc_all13 scores 0.499 on dev.

## Reproducing a phase

```bash
cd p5
P=2 bash run_queue.sh e1_a4_lorma e2_a1_full ...           # train experiments (logs/ and runs/)
python make_submission.py --run runs/Y2_kd4_a4_alpha03     # zip with per-statute thresholds
python stack.py --all --name STACK6_all                    # stacker over all finished runs
python report/make_figures.py                              # figures, tables and numbers for the report
```

The other phases follow the same pattern; each `train.py`, stacker and submission script documents its own
options in the module docstring (`python <script> -h`).