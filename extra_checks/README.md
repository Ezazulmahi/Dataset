# Extra checks

Scripts and results of the checks reported in the paper beyond the main evaluation. Every script saves each result as soon as it is finished and skips finished items when run again.

| Script | Paper | Results |
|---|---|---|
| `code/check_derivations.py` | Methods, Supplementary Note 3 | `results/derivations_report.txt`: every formula checked by algebra, worked examples and Monte Carlo through the embedder's own functions |
| `code/test1_baselines.py` | Table 3, Supplementary Table S12 | `results/test1_stats.csv` (eligible count, K, capacity and spectrum of fine-tuned, baseline, sharpened and noised baseline covers and 195 photographs), `results/test1_embed.csv` (matched payload embedding) |
| `code/test2_generation.py` | Supplementary Table S15 | `results/test2_generation.csv` and the generated covers in `test2_images/` (classifier-free guidance at 256 and 512 pixels) |
| `code/test3_random_bits.py` | Supplementary Note 2 | `results/test3_random_bits.csv` (uniformly random bits instead of text on 100 pairs) |
| `code/recompute_tableS2.py` | Supplementary Table S2 | `results/tableS2_recomputed.csv`, from the covers in `training_records/checkpoint_covers/` |
| `code/capacity_checks.py` | Results, Embedding capacity | capacity on the same 566 pairs and `results/used_slot_eligibility.csv` (share of the written slots that hold an eligible coefficient) |
| `code/analyze.py` | | `results/summary.txt` |

`code/common.py` holds the shared helpers and loads the embedder from `embedder/DCT_Adaptive_v5.py`. The folder paths near the top of each script point to our machine; change them to your copies. The 195 photographs are not redistributed; their Lorem Picsum identifiers are in `cover_statistics/reference_photo_picsum_ids.txt`.
