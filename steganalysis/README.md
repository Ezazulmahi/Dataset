# Steganalysis

Code, settings, logs and per-image scores of the detectability experiments in the paper (Table 4, Fig. 4c,d, Supplementary Table S7). The large files (training images, trained detectors, features) are attached to the [v3.0 release](https://github.com/Ezazulmahi/Dataset/releases/tag/v3.0).

## Data

* **Training covers.** 4,000 covers generated with the paper's generation procedure, 2,000 by each model (baseline and fine-tuned Stable Diffusion v1.5), with prompts from the first 400 prompts of the bank and seeds from 100000 upwards. No prompt and seed pair of the evaluation corpus is reused, but prompts are shared with corpus pairs 0 to 399. All detectors work on the 8-bit luminance plane at 256 by 256.
* **Split.** `split.csv` assigns every training cover to `train` (3,195), `val` (399) or `test` (406, the held-out pairs). Only names `ft_00000` to `ft_01999` and `bs_00000` to `bs_01999` are used. A cover and its stego image are always in the same subset.
* **Training stego, 0.4 bits per pixel of random bits.**
  * S-UNIWARD, WOW, HILL and MiPOD: simulated at the entropy bound with the conseal library.
  * DEDS: written with the paper's embedder at a margin of 0.25 (26,208 bits per image, K forced from the payload), each image read back bit for bit. Settings in `deds_training_embedder_settings.json`, log in `deds_training_embed_log.csv`. Note that the corpus stego images used for testing were written at a margin of 0.15 and 0.30 to 0.60 bits per pixel of ASCII text, so the DEDS detectors are tested on different embedding settings from those they were trained on (see the paper).
* **Curriculum.** The content-adaptive detectors were first trained at 1.0 bits per pixel (`matched_1.0`) and then fine-tuned at 0.4.
* **JPEG reference.** Covers compressed at quality 75 and J-UNIWARD and J-MiPOD at 0.4 bits per non-zero AC coefficient (`matched_jpeg75`). `code/jmipod.py` is our implementation of J-MiPOD on conseal's MiPOD components.
* **Test sets.** The 1,000 fine-tuned and 1,000 baseline covers of the paper's corpus and 200 unseen-prompt pairs, each embedded by every method at 0.4 bits per pixel, and the paper's own DEDS stego images.

## Folder names used in the code and the release

| Name | Contents |
|---|---|
| `matched_0.4/covers_y/universal` | the 4,000 training covers (luminance) |
| `matched_0.4/covers_y/paper_ft`, `paper_bs`, `unseen_ft`, `unseen_bs` | corpus and unseen-prompt covers (luminance) |
| `matched_0.4/<method>/...` | stego images of each method at 0.4 bits per pixel, same sub-folders |
| `matched_0.4/deds_v4/...` | DEDS stego at margin 0.25 (training settings) |
| `matched_0.4/deds_v5_sub2/...` | the paper's DEDS stego images (margin 0.15), luminance |
| score set `paper_ft` | corpus fine-tuned covers against that method's own stego (for DEDS: margin 0.25 stego) |
| score set `v5_paper_ft` | corpus fine-tuned covers against the paper's DEDS stego images |
| score set `universal_test500` | the 406 held-out pairs |

## Detectors

* **SRNet-KV** (`code/02_train.py`, `code/common.py`): SRNet behind a fixed 5 by 5 KV high-pass filter. AdamW, learning rate 2e-4 with cosine decay, weight decay 2e-4, 16 pairs per batch, early stopping on validation error with patience 15 (at most 80 epochs). Run settings and learning curves in `runs/*/args.json` and `runs/*/log.csv`. Trained weights in the release (`steg_detectors.zip`, file `best.pt` per run). Test scores are averaged over the eight flips and rotations (`results/*_tta`).
* **SRM + FLD** (`code/10_srm_extract.py`, `code/11_srm_classify.py`): spatial rich model features (34,671) with the ensemble of Fisher linear discriminants (sealwatch). Features in the release (`steg_features_srm_part1` to `part3`).
* **DCTR + FLD** (`code/12_build_jpeg.py`, `code/13_jpeg_classify.py`): DCTR features (8,000) for the JPEG reference. Features in `steg_features_dctr.zip`.

## Scores

Every `results/<detector>/<set>_scores.csv` has one row per image with `label` (0 cover, 1 stego) and a score that is higher for stego (`margin` or `p_stego` for SRNet-KV, `score` for the ensembles). `results/summary_all.csv`, `results/srm_fld/summary.csv` and `results/dctr_fld/summary.csv` list AUC with bootstrap intervals and the error rates; `paper_outputs/` holds Table 4 and Fig. 4 as produced by `code/14_paper_outputs.py`.

## Running the code

Scripts are numbered in the order they were run. They still contain the folder paths of our machine (for example `F:\Steganalyzer_SOTA_2026`) near the top; change them to where you unpack the release. Main dependencies: PyTorch, NumPy, SciPy, OpenCV, scikit-learn, conseal, sealwatch, jpeglib.
