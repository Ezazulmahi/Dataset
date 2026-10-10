r"""Make working copies of the authors' evaluation scripts for pairs 400 to 999. Only the path constants
in the CONFIG block are changed (and MAX_IMAGES 400 -> 600 in newrobust.py). The originals in
F:\Results are not touched. The copies are run with PYTHONHASHSEED=42 so that the hash()-based
message seeds of the original scripts are reproducible. fid.py additionally imports clip_compat (a
transformers 5 compatibility shim, inserted by hand after copying)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__)); DST = os.path.join(HERE, "authors_scripts_paths_changed"); os.makedirs(DST, exist_ok=True)
NEW = os.path.join(HERE, "new600"); OUT = os.path.join(HERE, "results_new600"); os.makedirs(OUT, exist_ok=True)
def r(p): return 'r"%s"' % p
EDITS = {
 "Robustness.py": {"FINETUNED_DIR": r(os.path.join(NEW, "finetuned")), "EXCEL_OUT_PATH": r(os.path.join(OUT, "robustness_results_400_999.xlsx")),
                   "CHECKPOINT_PATH": r(os.path.join(OUT, "checkpoints", "robustness_checkpoint.csv"))},
 "newrobust.py": {"STEGO_DIR": r(os.path.join(NEW, "stego_finetuned")), "EXCEL_PATH": r(os.path.join(OUT, "robustness_customQ_adv_ecc_align_400_999.xlsx")), "MAX_IMAGES": "600"},
 "Steganalysis.py": {"FINETUNED_COVER_DIR": r(os.path.join(NEW, "finetuned")), "STEGO_DIR": r(os.path.join(NEW, "stego_finetuned")),
                     "COMPARISON_XLSX": r(os.path.join(OUT, "comparison_results_400_999.xlsx")), "EXCEL_OUT_PATH": r(os.path.join(OUT, "steganalysis_results_400_999.xlsx")),
                     "REPORT_TXT_PATH": r(os.path.join(OUT, "steganalysis_report_400_999.txt"))},
 "fid.py": {"FINETUNED_DIR": r(os.path.join(NEW, "finetuned")), "STEGO_DIR": r(os.path.join(NEW, "stego_finetuned")), "EXCEL_PATH": r(os.path.join(OUT, "comparison_results_400_999.xlsx"))},
}
for fn, ed in EDITS.items():
    src = open(os.path.join(r"F:\Results", fn), encoding="utf-8").read(); n = 0
    for k, v in ed.items():
        src, c = re.subn(r"(?m)^(%s\s*=\s*)[^#\n]*?(\s*(#.*)?)$" % k, lambda m: m.group(1) + v + m.group(2), src, count=1); n += c
    assert n == len(ed), (fn, n)
    open(os.path.join(DST, fn), "w", encoding="utf-8", newline="").write(src); print(fn, "copied,", n, "constants changed")
