| run | arch | params | non_emb | tokens_seen | val_loss | val_ppl | bertscore_f1 | rougeL_f1 | bleu4 | best_decoding | train_hours | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2_encdec | encdec | 89.42M | 68.94M | 196.61M | 3.0815 | 21.79 | 0.8154 | 0.1732 | 0.0318 | t0.8_k50_p0.95_rp1.1 | 0.867 | 30.683 |
| A3_prefixlm | prefixlm | 81.28M | 56.70M | 196.61M | 2.494 | 12.11 | 0.8349 | 0.2202 | 0.11 | t0.8_k50_p0.95_rp1.1 | 0.408 | 20.013 |
| A4_efficient | efficient | 87.51M | 62.93M | 196.61M | 2.4448 | 11.53 | 0.8346 | 0.2234 | 0.1104 | t0.8_k50_p0.95_rp1.1 | 0.75 | 20.482 |
