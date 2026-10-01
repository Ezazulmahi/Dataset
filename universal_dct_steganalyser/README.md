# Universal-DCT steganalyser

A convolutional steganalyser trained from scratch on a universal (generic) DCT embedding and
evaluated, without further training, on the 400 DEDS cover/stego pairs.

| Step | Script | Output |
|---|---|---|
| 1 | `01_generate_covers.py` | `covers/` - 5,000 covers (2,500 fine-tuned, 2,500 baseline generator; DDIM 50 steps, no guidance, 256x256; seeds 100000+, disjoint from the evaluation corpus) |
| 2 | `02_make_stego.py` | `stego/`, `pairs.csv` - universal DCT embedding (`embed_dct_random` of `gen_dataset.py`: standard JPEG luminance table, LSB replacement of a random 5-15% of the quantised coefficients larger than 1) |
| 3 | `03_train.py` | `model/universal_dct_srnet.pth`, `model/training_log.csv`, `model/test_scores_universal_dct.csv`, `split.csv` |
| 4 | `04_evaluate.py` | `results/universal_dct_steganalysis_results.xlsx`, `results/universal_dct_steganalysis_report.txt` |

Network: SRNet-style architecture of `Steganalyzer.py` (30 fixed high-pass filters, residual
blocks, squeeze-and-excitation), random initialisation, no weights from earlier detectors.
Split: 4,000 / 500 / 500 pairs (train / validation / test). The DEDS embedder is not used in training.

## Result

| Test set | Accuracy at trained threshold | Stego flagged | Covers flagged | ROC-AUC |
|---|---|---|---|---|
| Universal DCT, 500 held-out pairs | 99.4% | - | - | 1.000 |
| DEDS, 400 fine-tuned covers | 50.0% | 0 of 400 | 0 of 400 | 0.859 (95% CI 0.832-0.886) |
| DEDS, 400 baseline covers | 91.5% | 332 of 400 | 0 of 400 | 1.000 |

The training images (`covers/`, `stego/`, about 1.5 GB) are not included in the repository copy;
they are reproduced exactly by steps 1 and 2.

The scripts in this folder were written with the assistance of Claude (Anthropic), as declared
in the Methods of the article.
