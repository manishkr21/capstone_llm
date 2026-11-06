| run | arch | params | non_emb | tokens_seen | val_loss | val_ppl | bertscore_f1 | rougeL_f1 | bleu4 | best_decoding | train_hours | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_gpt | gpt | 396.29K | 265.22K | 81.92K | 3.2919 | 26.89 |  | 0.2314 | 0.0221 | t0.8_k50_p0.95_rp1.1 | 0.001 | 11.13 |
| A2_encdec | encdec | 761.34K | 663.04K | 81.92K | 4.5873 | 98.23 |  | 0.2388 | 0.0044 | t0.8_k50_p0.95_rp1.1 | 0.009 | 11.183 |
| A3_prefixlm | prefixlm | 363.52K | 265.22K | 81.92K | 2.9965 | 20.02 |  | 0.2186 | 0.0296 | t0.8_k50_p0.95_rp1.1 | 0.001 | 11.134 |
| A4_efficient | efficient | 393.86K | 295.55K | 81.92K | 3.2848 | 26.7 |  | 0.2159 | 0.0172 | t0.8_k50_p0.95_rp1.1 | 0.001 | 11.138 |
