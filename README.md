# DEDS data and evaluation files

This repository holds the data behind the paper **"Diffusion-enhanced DCT steganography: reward-guided cover generation for increased embedding capacity with exact lossless-channel recovery"** (DEDS).

It contains the cover images, the LoRA adapter, the prompt bank, the embedder used in the paper (margin projected parity embedding, called v5 in the file names), every evaluation script, and the per image result files that the tables and figures were computed from. The stego images are too large for the repository itself and are attached to the [v2.0 release](https://github.com/Ezazulmahi/Dataset/releases/tag/v2.0) as zip files.

The fine tuning code is in a separate repository: https://github.com/Imraj-Rabbani/P3-Image-Steganography-Pipeline

## What is where

```
prompts/
    prompts.py               prompts 0 to 399 (40 texture subjects, 10 photographic modifiers each)
    prompts_400_999.txt      prompts 400 to 999 (60 further subjects, same modifiers), also as .json
    prompts_unseen.txt       200 prompts from 20 subjects outside the bank, for the unseen prompt check

lora_weights/
    deds_lora_adapter.pt     the trained LoRA factors only (rank 4, about 2.5 MB)

covers/
    baseline/                1,000 covers from unmodified Stable Diffusion v1.5
    finetuned/               1,000 covers from the fine tuned model, same prompt and seed as baseline
    unseen_baseline/         200 baseline covers for the unseen prompt check
    unseen_finetuned/        200 fine tuned covers for the unseen prompt check

generation/
    validate.py              loads Stable Diffusion v1.5 with or without the adapter and samples covers
    01_generate_covers.py    batched sampling helper used by generate_unseen.py
    generate_400_999.py      generates covers 400 to 999
    generate_unseen.py       generates the 200 unseen prompt pairs
    12_timing_single_process.py, timing_single_process.csv
                             generation time and peak memory per cover

cover_statistics/            cover statistics before embedding (Table 1, Fig. 2)
    exp_stats.py             eligible coefficient counts, adaptive K and capacity, saved as cover_stats.csv and elig_maps.npz
    make_figs.py             eligibility maps and radial power spectra, saved as psd.npz, and the Fig. 2 panels
    common.py, wow.py        shared helpers
    cover_stats_unseen.csv   the same statistics for the unseen prompt covers
    reference_photo_picsum_ids.txt
                             Lorem Picsum identifiers of the 195 reference photographs

training_records/
    ablation_results.csv     loss weight grid with average K per setting and links to its sample images (Supplementary Table S3)
    checkpoint_covers/       the four validation covers generated from every training checkpoint (Supplementary Table S2)

embedder/
    DCT_Adaptive_v5.py       the embedder and extractor (normal mode and robust mode)
    v5_derivation.md         derivation of the expected distortion and the guaranteed tolerance

evaluation/
    01_embed_normal_1000.py  embeds all 1,000 pairs at matched payload, checks exact recovery,
                             measures quality and runs the processing tests
    02_robust_mode_subset.py robust mode on 200 fine tuned covers at eight payloads
    03_fid_clip.py           FID and CLIP similarity between covers and stego images
    04_figures.py            Figs. 3 and 4 (a, b) and Supplementary Fig. S1
    05_report.py             prints every number reported in the paper from the result files
    08_validate_theory.py    numerical checks of the derivations (Supplementary Note 3, Fig. S2)
    09_unseen_v5.py          the unseen prompt check (Supplementary Table S7)
    inputs/comparison_results.xlsx
                             the matched payload and K assigned to each pair, and the results of the
                             bin centre write rule that the paper compares against
    inputs/fid.py            FID helper imported by 03_fid_clip.py
    results/                 per image results (normal_per_image.csv, robust_per_image.csv, ...),
                             the paper's tables as CSV (results/tables) and the numerical checks
                             (results/validation)
    figures/                 the figures as PDF and PNG

steganalysis/                detectors, settings, logs and per image scores (Table 4, Fig. 4c,d); see its README

extra_checks/                derivation checks, photographs and processed covers (Table 3, Supplementary Table S12),
                             generation settings (Supplementary Table S15), random bit messages (Supplementary Note 2)
                             and Supplementary Table S2; see its README
```

## Stego images

The [v2.0 release](https://github.com/Ezazulmahi/Dataset/releases/tag/v2.0) has five zip files:

| File | Images | Contents |
|---|---|---|
| `stego_finetuned.zip` | 1,000 | stego images of the fine tuned covers, normal mode |
| `stego_baseline.zip` | 1,000 | stego images of the baseline covers, same message and K as their fine tuned pair |
| `stego_robust.zip` | 200 | robust mode stego images of 200 fine tuned covers |
| `stego_unseen_finetuned.zip` | 200 | stego images of the unseen prompt fine tuned covers |
| `stego_unseen_baseline.zip` | 200 | stego images of the unseen prompt baseline covers |

Image `NNN.png` in a stego folder belongs to cover `NNN.png` of the matching cover folder. The message of pair *i* is drawn from a pseudorandom generator seeded with 424242 + *i*, so every message can be rebuilt with `01_embed_normal_1000.py` and checked against the extractor.

## Reproducing the results

1. Covers. Prompt *i* was sampled with seed 42 + *i* for both models, with 50 DDIM steps, no classifier free guidance, at 256 by 256 pixels. Load the adapter on top of `runwayml/stable-diffusion-v1-5` with the PEFT library (rank 4, alpha 8, query, key and value projections of every attention layer).
2. Embedding and evaluation. Run the scripts in `evaluation/` in numerical order. `05_report.py` prints the numbers in the paper.
3. Cover statistics. Run `cover_statistics/exp_stats.py`, then `make_figs.py`.

The scripts still hold the folder paths of our own machine (for example `F:\scirep results\...`) as constants near the top of each file. Change them to your local copies of the folders above before running.

## Notes

* All images are 256 by 256 PNG files. The covers are exactly the images the paper was evaluated on.
* Training images came from the public `gmongaras/Imagenet21K_Recaption` dataset on Hugging Face and are not redistributed here.
* The steganalysis code, settings, logs and per image scores are in `steganalysis/`; the training images, trained detectors and features are attached to the [v3.0 release](https://github.com/Ezazulmahi/Dataset/releases/tag/v3.0). See `steganalysis/README.md`.

## Licence

MIT, see `LICENSE`. If you use these files, please cite the DEDS paper.
