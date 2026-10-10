r"""Expansion, step 5: copy the revision analysis scripts into analysis1000\ and change only what the
larger corpus requires (corpus folder, 400 -> 1,000 images, 40 -> 100 prompt subjects, output paths).
Every replacement is asserted, so a script that no longer matches stops the build."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = r"F:\Publication\scirep_submission\analysis"; DST = os.path.join(HERE, "analysis1000")
os.makedirs(DST, exist_ok=True)
C1000 = os.path.join(HERE, "corpus1000"); SUB = "F:/Publication/scirep_submission"
P = {
 "common.py": [('RES = r"F:\\Results"', 'RES = r"%s"' % C1000, 1)],
 "exp_stats.py": [("range(400)", "range(1000)", 1)],
 "exp_steg.py": [("N = 400", "N = 1000", 1),
                 ("rng.permutation(40); fold_of = np.empty(40, int); fold_of[perm] = np.arange(40) % folds",
                  "rng.permutation(100); fold_of = np.empty(100, int); fold_of[perm] = np.arange(100) % folds", 1),
                 ('MY_FT = "stego_ft"', 'MY_FT = "stego_ft_not_used"', 1)],
 "exp_steg_generic_dct.py": [("N = 400;", "N = 1000;", 1), ("of 400", "of 1000", 1)],
 "exp_customq.py": [('cover = cv2.imread(os.path.join(RES, "finetuned", name))', 'cover = cv2.imread(os.path.join(r"%s", "finetuned", name))' % C1000, 1),
                    ("range(400)", "range(400, 1000)", 1), ('"customq_rerun.csv"', '"customq_400_999.csv"', 1)],
 "make_figs.py": [("range(400)", "range(1000)", 2), ("(of 400)", "(of 1,000)", 1), ("(n = 400)", "(n = 1,000)", 3), ("../figures/", SUB + "/figures/", 2),
                  ("; ax.set_ylim(0, 150)", "", 1)],
 "final_numbers.py": [("len(ft) == 400 and len(bs) == 400", "len(ft) == 1000 and len(bs) == 1000", 1), ("ft %d/400  base %d/400", "ft %d/1000  base %d/1000", 1),
                      ('of 400", "RecBase": f"{int(bs.exact.sum())} of 400"', 'of 1,000", "RecBase": f"{int(bs.exact.sum())} of 1,000"', 1),
                      ('"of 400")', '"of 1000")', 1), ('"../numbers.tex"', '"numbers.tex"', 1), ('"../main.tex"', '"%s/main.tex"' % SUB, 2), ("../figures/", SUB + "/figures/", 1),
                      ('"F:/Steganalyzer_UniversalDCT/results/universal_dct_steganalysis_results.xlsx"', '"%s"' % os.path.join(HERE, "results_1000", "universal_dct_steganalysis_results.xlsx").replace("\\", "/"), 1),
                      ('"DEDS on baseline covers (re-embedding run)"', '"DEDS on baseline covers"', 1)],
}
for fn, edits in P.items():
    s = open(os.path.join(SRC, fn), encoding="utf-8").read()
    for old, new, cnt in edits:
        assert s.count(old) == cnt, (fn, old, s.count(old)); s = s.replace(old, new)
    open(os.path.join(DST, fn), "w", encoding="utf-8", newline="").write(s); print(fn, "written with", len(edits), "changes")
