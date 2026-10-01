# DEDS Dataset and Evaluation Artifacts

Companion data repository for **"Diffusion-Enhanced DCT Steganography for High-Capacity Data Hiding" (DEDS)**.

The main pipeline code (LoRA fine-tuning, DCT embedding/extraction, Huffman compression, SRNet steganalysis) is released separately at:
https://github.com/Imraj-Rabbani/P3-Image-Steganography-Pipeline

This repository holds the artifacts needed to reproduce the paper's reported results: the LoRA adapter weights, the fixed prompt bank, the evaluation scripts used to produce each results table, the raw result spreadsheets, and the generated image sets themselves.

## Contents

```
lora_weights/
    deds_lora_adapter.pt   - LoRA adapter weights only (rank 4, ~2.5 MB), extracted
                              from the full fine-tuned UNet state dict. Base UNet,
                              VAE, and CLIP text encoder stay frozen at their stock
                              Stable Diffusion v1.5 values throughout training, so
                              only this small delta needs to be distributed. Load it
                              on top of the public runwayml/stable-diffusion-v1-5
                              checkpoint with the PEFT library to reconstruct the
                              exact fine-tuned model used in this work.

prompts.py                 - The fixed texture-prompt bank used to generate every
                              cover image in this work (baseline and fine-tuned).

scripts/
    DCT_Adaptive.py         - The adaptive DCT embedding/extraction pipeline
                               (capacity estimation, header, LSB matching,
                               verify-fix loop, colour repair).
    Compilot.py              - Baseline vs. fine-tuned comparison pipeline
                               (pilot, n=5): PSNR/SSIM, Wilcoxon, FID, CLIP.
    comparison.py            - Baseline vs. fine-tuned comparison pipeline,
                               full scale (n=400).
    Steganalysis.py          - SRNet/Aletheia/GBRAS-Net steganalysis evaluation.
    Robustness.py            - Channel distortion sweep (JPEG, resize, noise,
                               blur, crop, adversarial perturbation).
    newrobust.py             - Extended robustness conditions (custom-Q JPEG,
                               Hamming(7,4) ECC, adversarial + ECC combinations).
    fid.py / fidpilot.py     - FID computation, full-scale and pilot.
    clip.py                  - CLIP cosine similarity computation.
    validate.py               - End-to-end pipeline validation.

results/
    comparison_results.xlsx               - Full n=400 PSNR/SSIM/capacity results.
    comparison_results_pilot.xlsx         - n=5 pilot comparison results.
    fid_clip_pilot5.xlsx                  - FID/CLIP results, pilot.
    robustness_results.xlsx               - Channel distortion sweep results.
    robustness_customQ_adv_ecc_align.xlsx - Extended robustness conditions.
    steganalysis_results.xlsx             - SRNet/Aletheia/GBRAS-Net results.

images/
    full_400/
        baseline/         - 400 baseline (unmodified Stable Diffusion v1.5) covers.
        finetuned/         - 400 fine-tuned covers, same prompts and seeds as baseline.
        stego_finetuned/   - 400 stego images embedded into the fine-tuned covers.
    pilot_5/
        baseline/          - The 5 representative pairs' baseline covers
                              (Moss, Bark, Tree, Shallow water, Brick).
        finetuned/         - The same 5 covers, fine-tuned.
        stego/             - The 5 fine-tuned stego images.
```

## Notes

- All images are 256x256 PNG, generated from the fixed prompt bank in `prompts.py`.
- The `full_400` set is the evaluation corpus used throughout Section 4 of the paper
  (n = 400: 200 baseline/fine-tuned pairs, shared seed per pair).
- The `pilot_5` set is the five-cover qualitative anchor set used for the
  side-by-side figures in the paper; it is illustrative only and not
  statistically powered on its own.
- Result spreadsheets contain the raw per-image numbers the paper's tables
  were computed from.

## Citation

If you use this data, please cite the DEDS paper (see the main pipeline
repository for the full citation).

## Revision analysis

`revision_analysis/` holds the scripts and result files added for the revised
manuscript: cover statistics, the re-embedding run that verifies exact recovery
with known messages, the feature-based (SPAM) steganalysis with the S-UNIWARD
and LSB-matching references, the loss-weight grid record, and the figure files.
See `revision_analysis/README.md`.

## Universal-DCT steganalyser

`universal_dct_steganalyser/` holds the convolutional detector trained from scratch on a universal DCT
embedding (trained weights, data-generation, training and evaluation scripts, per-image results).
See `universal_dct_steganalyser/README.md`.
