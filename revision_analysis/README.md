# Revision analysis (Scientific Reports submission)

Scripts and result files behind the statistics, tables and figures of
"Diffusion-enhanced DCT steganography: reward-guided cover generation for
increased embedding capacity with exact lossless-channel recovery".

All scripts expect the corpus folders of this repository (`images/full_400/...`)
and the original result spreadsheets (`results/`); edit the paths at the top of
`common.py` to point at your local copies.

| File | Purpose |
|---|---|
| `common.py` | Shared code: block DCT and adaptive-K statistics, SPAM-686 features, S-UNIWARD cost and embedding simulator, LSB-matching simulator |
| `verify.py` | Recomputes the distortion, steganalysis and processing tables from the original spreadsheets |
| `exp_stats.py` | Cover statistics of the 400 baseline / fine-tuned pairs (`cover_stats.csv`, `elig_maps.npz`) |
| `exp_embed.py` | Re-embeds all 800 covers with seed-reproducible messages, verifies byte-exact recovery after a PNG round trip, records runtime (`embed_rerun.csv`) |
| `exp_steg.py` | SPAM + shrinkage LDA steganalysis with subject-grouped five-fold cross-validation, including the S-UNIWARD and LSB-matching references (`steganalysis_spam.csv`, `steganalysis_spam_scores.npz`) |
| `gen_timing.py` | Cover-generation time and memory; checks that the stored covers are reproduced by the released weights |
| `make_figs.py`, `fig_schematic.py`, `fig_examples_selected.py`, `final_numbers.py` | Figures 1-4, Supplementary Fig. S1 and the numbers quoted in the text |
| `ablation_results.csv` | Loss-weight grid (average K per setting) with links to the generated covers of each setting |
| `reference_photo_picsum_ids.txt` | Lorem Picsum image identifiers of the 195 reference photographs used for the spectral comparison (https://picsum.photos/id/ID/256/256) |
| `figures/` | Figure files as submitted |

The analysis scripts in this folder were written with the assistance of Claude
(Anthropic), as declared in the Methods of the article.
