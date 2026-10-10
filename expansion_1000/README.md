# Expansion of the evaluation corpus from 400 to 1,000 pairs

Pairs 400 to 999 were added during the Scientific Reports revision: 60 new texture subjects, each with the same
ten modifiers as the original prompt bank (`scripts/prompts_400_999.txt`), prompt i sampled with seed 42 + i.
Covers were generated with `load_pipeline()` and `generate_images()` of the authors' `validate.py` and the released
adapter weights; before use, the function reproduced a stored cover of the original corpus exactly.
Embedding uses the unchanged `DCT_Adaptive.py` and the matched-payload protocol of `comparison.py`
(fixed integer seeds instead of Python `hash()`), with every stego image read back from PNG and extracted.

- `images_400_999/`: baseline and fine-tuned covers and their stego images.
- `results_400_999/`: the authors' spreadsheets for the new pairs, produced by copies of their scripts with only the paths changed (`scripts/authors_scripts_paths_changed/`; `fid.py` also imports `clip_compat.py` for transformers 5).
- `merged_results/`: the authors' spreadsheets for all 1,000 pairs (column `corpus_part`), the universal-DCT detector results and `report_1000.txt`, which lists every number of the manuscript.
- `analysis_1000/`: cover statistics, SPAM detectors, S-UNIWARD and LSB matching on 1,000 pairs.

Results on the new 600 pairs alone match the original 400 (exact recovery 600 of 600 on both cover types,
SSIM 0.990 against 0.908, the same PSNR pattern by payload, the same processing-test outcomes).
