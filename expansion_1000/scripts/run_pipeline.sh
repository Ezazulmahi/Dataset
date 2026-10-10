#!/bin/bash
# Expansion pipeline (resumable). Covers 400-999 and the Robustness.py run are already complete.
# 02_embed.py resumes from embed_400_999.csv. Steps are sequential. Log: pipeline.log
# Start it ONCE, detached (PowerShell Start-Process), never through a harness background shell.
cd "/f/scirep results/09_expansion_1000" || exit 1
export PYTHONHASHSEED=42
A=authors_scripts_paths_changed
step() { echo "=== $(date +%H:%M:%S) $1" >> pipeline.log; }
run() { step "$1"; shift; "$@" >> pipeline.log 2>&1 || { echo "FAILED: $*" >> pipeline.log; exit 1; }; }

[ "$(ls new600/finetuned | wc -l)" -eq 600 ] && [ "$(ls new600/baseline | wc -l)" -eq 600 ] || { echo "FAILED: covers incomplete" >> pipeline.log; exit 1; }
[ -f results_new600/robustness_results_400_999.xlsx ] || run "Robustness.py (authors' script) on fine-tuned covers 400-999" python $A/Robustness.py
run "02_embed.py matched-payload protocol on pairs 400-999 (resumes)" python 02_embed.py
run "03 comparison spreadsheet for the new pairs (authors' summarize, FID, CLIP)" python 03_comparison_xlsx_new600.py
run "fid.py (authors' script) fine-tuned vs stego" python $A/fid.py
run "Steganalysis.py (authors' script, earlier detector and chi-square)" python $A/Steganalysis.py
run "newrobust.py (authors' script, additional processing tests)" python $A/newrobust.py
run "04 merge corpus and spreadsheets" python 04_merge_corpus.py
cd analysis1000
run "exp_stats (cover statistics, 1,000 pairs)" python exp_stats.py
run "exp_customq (own-table requantisation, covers 400-999)" python exp_customq.py
run "exp_steg (SPAM detectors, S-UNIWARD, LSB matching, 1,000 pairs)" python exp_steg.py
run "exp_steg_generic_dct" python exp_steg_generic_dct.py
cd ..
run "06 universal-DCT CNN on 1,000 pairs" python 06_cnn_eval_1000.py
step "PIPELINE DONE"
