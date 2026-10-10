r"""Expansion, step 7: every number of the manuscript that depends on the corpus, recomputed on the
1,000 pairs (and by corpus part), written to report_1000.txt."""
import os, sys, warnings
import numpy as np, pandas as pd, cv2
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); C1000 = os.path.join(HERE, "corpus1000"); A = os.path.join(HERE, "analysis1000")
L = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
pd.set_option("display.width", 250); np.set_printoptions(precision=3, suppress=True, linewidth=160)
ms = lambda x: "%.3f +/- %.3f" % (np.mean(x), np.std(x, ddof=1))

# ------------------------------------------------------------------ A. cover statistics (Table 1)
cs = pd.read_csv(os.path.join(A, "cover_stats.csv")); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
part = np.where(np.array([int(n[:3]) for n in f.index]) < 400, "original 400", "expansion 600")
sys.path.insert(0, r"F:\Results")
import validate as V
def midratio(folder):
    p = os.path.join(A, "midratio_%s.npy" % folder)
    if os.path.exists(p): return np.load(p)
    r = np.array([V.mid_band_energy_ratio(V.compute_block_dct(V.uint8_bgr_to_float_rgb(cv2.imread(os.path.join(C1000, folder, n))))) for n in f.index]); np.save(p, r); return r
mb, mf = midratio("baseline"), midratio("finetuned")
log("== A. COVER STATISTICS, n = %d pairs" % len(f))
log("energy ratio 3<=u+v<=10: base", ms(mb), "| ft", ms(mf), "| first 400: base %.3f ft %.3f (manuscript 0.051, 0.352)" % (mb[:400].mean(), mf[:400].mean()))
log("eligible per block: base", ms(b.mean_eligible), "| ft", ms(f.mean_eligible))
log("  median (range): base %.2f (%.2f to %.2f) | ft %.2f (%.2f to %.2f)" % (b.mean_eligible.median(), b.mean_eligible.min(), b.mean_eligible.max(), f.mean_eligible.median(), f.mean_eligible.min(), f.mean_eligible.max()))
log("  ft > base in %d of %d pairs; Wilcoxon p = %.3g; CV base %.2f ft %.2f" % ((f.mean_eligible.values > b.mean_eligible.values).sum(), len(f), stats.wilcoxon(b.mean_eligible, f.mean_eligible).pvalue,
    b.mean_eligible.std() / b.mean_eligible.mean(), f.mean_eligible.std() / f.mean_eligible.mean()))
log("fallback covers: base %d, ft %d" % (b.fallback.sum(), f.fallback.sum()))
log("K all: base", ms(b.K), "| ft", ms(f.K)); nb = b[~b.fallback]
log("K cover-derived: base", ms(nb.K), "(n=%d)" % len(nb), "| ft", ms(f[~f.fallback].K))
log("capacity bpp all: base", ms(b.capacity_bpp), "| ft", ms(f.capacity_bpp), "| bits base %.0f ft %.0f" % (b.capacity_bits.mean(), f.capacity_bits.mean()))
log("capacity bpp cover-derived: base", ms(nb.capacity_bpp), "range %.3f to %.3f | ft range %.3f to %.3f" % (nb.capacity_bpp.min(), nb.capacity_bpp.max(), f.capacity_bpp.min(), f.capacity_bpp.max()))
log("capacity >= 0.30: base cover-derived %d of %d | ft %d of %d ; ratio of cover-derived means %.1f" % ((nb.capacity_bpp >= .3).sum(), len(nb), (f.capacity_bpp >= .3).sum(), len(f), f.capacity_bpp.mean() / nb.capacity_bpp.mean()))
for p in ("original 400", "expansion 600"):
    m = part == p
    log("  [%s] eligible base %.2f ft %.2f | fallback base %d | ft capacity %.3f, >=0.30: %d of %d | base cover-derived capacity %.3f (n=%d)" % (p, b.mean_eligible[m].mean(), f.mean_eligible[m].mean(),
        b.fallback[m].sum(), f.capacity_bpp[m].mean(), (f.capacity_bpp[m] >= .3).sum(), m.sum(), b[m][~b[m].fallback].capacity_bpp.mean(), (~b[m].fallback).sum()))
m_ = np.load(os.path.join(A, "elig_maps.npz")); S = np.zeros((8, 8), bool)
import importlib.util
spec = importlib.util.spec_from_file_location("commonA", os.path.join(A, "common.py")); CA = importlib.util.module_from_spec(spec); spec.loader.exec_module(CA)
log("eligibility map, baseline: max over M %.3f" % m_["base"][CA.STABLE].max()); log(np.where(CA.STABLE, m_["base"], np.nan))
log("eligibility map, fine-tuned:"); log(np.where(CA.STABLE, m_["ft"], np.nan))

# ------------------------------------------------------------------ B. CLIP / FID
x = pd.ExcelFile(os.path.join(C1000, "comparison_results.xlsx"))
log("\n== B. CLIP / FID"); log(x.parse("FID_CLIP").T.to_string()); log(x.parse("FID_CLIP_finetuned_vs_stego").T.to_string())
log("CLIP ft vs stego: mean %.4f min %.4f" % (x.parse("clip_per_image_finetuned_vs_stego").clip_cosine_similarity.mean(), x.parse("clip_per_image_finetuned_vs_stego").clip_cosine_similarity.min()))
log(x.parse("FID_by_corpus_part").round(3).to_string(index=False))

# ------------------------------------------------------------------ C. matched payload (Table 2)
d = x.parse("per_image_results")
log("\n== C. MATCHED PAYLOAD, n = %d" % len(d))
log("achieved bpp min %.3f mean %.3f max %.3f | n_bits %d to %d | K %d to %d" % (d.achieved_bpp.min(), d.achieved_bpp.mean(), d.achieved_bpp.max(), d.n_bits.min(), d.n_bits.max(), d.K.min(), d.K.max()))
fk = f.K.reindex(d.image).values; bk = b.K.reindex(d.image).values
log("forced K > adaptive K: fine-tuned %d of %d (first 400: %d)" % ((d.K.values > fk).sum(), len(d), (d.K.values[:400] > fk[:400]).sum()))
def rrb(diff):
    rk = stats.rankdata(np.abs(diff)); return (rk[diff > 0].sum() - rk[diff < 0].sum()) / rk.sum()
def fmtp(p):
    if p >= 1e-3: return "%.3f" % p
    e = int(np.floor(np.log10(p))); return "$%.1f\\times10^{%d}$" % (p / 10 ** e, e)
edges = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60001]; d["bin"] = pd.cut(d.target_bpp, edges, include_lowest=True); tex = []
def qrow(lab, s, last=False):
    dp = (s.finetuned_psnr - s.baseline_psnr).values; dsim = (s.finetuned_ssim - s.baseline_ssim).values
    pp = stats.wilcoxon(s.baseline_psnr, s.finetuned_psnr).pvalue; ps = stats.wilcoxon(s.baseline_ssim, s.finetuned_ssim).pvalue
    log("%-13s n=%4d PSNR base %.2f ft %.2f d %+.2f p %.3g r %+.2f | SSIM base %.3f ft %.3f d %+.3f p %.3g r %+.2f | ft SSIM higher in %d" % (lab, len(s), s.baseline_psnr.mean(), s.finetuned_psnr.mean(),
        dp.mean(), pp, rrb(dp), s.baseline_ssim.mean(), s.finetuned_ssim.mean(), dsim.mean(), ps, rrb(dsim), (dsim > 0).sum()))
    pcell = fmtp(pp) if last else "%s ($%+.2f$)" % (fmtp(pp), rrb(dp))
    tex.append("%s & %s & %.2f & %.2f & $%+.2f$ & %s & %.3f & %.3f & $%+.3f$ & %s \\\\" % (lab, f"{len(s):,}".replace(",", "{,}") if len(s) > 999 else len(s), s.baseline_psnr.mean(), s.finetuned_psnr.mean(), dp.mean(), pcell,
               s.baseline_ssim.mean(), s.finetuned_ssim.mean(), dsim.mean(), fmtp(ps)))
for (lo, hi), (bn, s) in zip(zip(edges[:-1], edges[1:]), d.groupby("bin", observed=True)):
    qrow("%.2f to %.2f" % (lo, min(hi, 0.60)), s)
tex.append("\\midrule"); qrow("All", d, last=True)
log("TABLE 2 ROWS:"); [log(t) for t in tex]
open(os.path.join(HERE, "table2_rows.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
log("SSIM ft", ms(d.finetuned_ssim), "range %.3f to %.3f | base" % (d.finetuned_ssim.min(), d.finetuned_ssim.max()), ms(d.baseline_ssim), "range %.3f to %.3f" % (d.baseline_ssim.min(), d.baseline_ssim.max()))
log("PSNR ft range %.1f to %.1f | mean delta %+.3f" % (d.finetuned_psnr.min(), d.finetuned_psnr.max(), d.delta_psnr.mean()))
for p, s in d.groupby("corpus_part"):
    log("  [%s] dPSNR %+.3f dSSIM %+.3f | SSIM ft %.3f base %.3f | PSNR ft %.2f base %.2f" % (p, s.delta_psnr.mean(), s.delta_ssim.mean(), s.finetuned_ssim.mean(), s.baseline_ssim.mean(), s.finetuned_psnr.mean(), s.baseline_psnr.mean()))
    s2 = s.copy(); s2["bin"] = pd.cut(s2.target_bpp, edges, include_lowest=True)
    log("     per-bin dPSNR:", " ".join("%+.2f" % v for v in s2.groupby("bin", observed=True).delta_psnr.mean()), "| dSSIM:", " ".join("%+.3f" % v for v in s2.groupby("bin", observed=True).delta_ssim.mean()))
low = d[d.target_bpp < 0.45]; high = d[d.target_bpp >= 0.50]
log("pooled <0.45: dPSNR %+.3f p %.3g | >=0.50: dPSNR %+.3f p %.3g" % (low.delta_psnr.mean(), stats.wilcoxon(low.delta_psnr).pvalue, high.delta_psnr.mean(), stats.wilcoxon(high.delta_psnr).pvalue))

# ------------------------------------------------------------------ D. recovery, verify-fix passes, runtime
e = pd.read_csv(os.path.join(A, "embed_rerun.csv")); ft = e[e.set == "ft"]; bs = e[e.set == "base"]
log("\n== D. RECOVERY AND RUNTIME (re-embedding run for 0-399, expansion run for 400-999)")
log("exact: ft %d of %d, base %d of %d | errors: %s" % (ft.exact.sum(), len(ft), bs.exact.sum(), len(bs), list(e.error.dropna().unique())))
log("fix passes: ft median %d max %d | base median %d max %d" % (ft.fix_passes.median(), ft.fix_passes.max(), bs.fix_passes.median(), bs.fix_passes.max()))
log("colour PSNR: ft", ms(ft.psnr_color), "| base", ms(bs.psnr_color))
log("embed time ft median: 0.30-0.40 %.0f s, >=0.50 %.0f s | extraction median %.2f s" % (ft[ft.bpp < .4].embed_s.median(), ft[ft.bpp >= .5].embed_s.median(), e.extract_s.median()))
for p, s in e.groupby("corpus_part"):
    log("  [%s] exact ft %d base %d | embed median ft %.0f s base %.0f s" % (p, s[s.set == "ft"].exact.sum(), s[s.set == "base"].exact.sum(), s[s.set == "ft"].embed_s.median(), s[s.set == "base"].embed_s.median()))

# ------------------------------------------------------------------ E. processing tests (Table 4, Supplementary Table S1)
r = pd.ExcelFile(os.path.join(C1000, "robustness_results.xlsx")); pi = r.parse("per_image_results")
log("\n== E. PROCESSING TESTS, half-capacity run, n = %d covers" % pi.image.nunique())
log("payload n_bits %d to %d | K %d to %d | rows with error: %d (%s)" % (pi.n_bits.min(), pi.n_bits.max(), pi.K.min(), pi.K.max(), pi.error.notna().sum(), list(pi[pi.error.notna()].distortion.unique())))
g = pi[pi.error.isna()].groupby("distortion", sort=False).agg(n=("image", "size"), bit_accuracy=("bit_accuracy", "mean"), header=("header_survives", "sum"), exact=("extract_success", "sum"))
log(g.round(4).to_string())
cq = pd.concat([pd.read_csv(r"F:\Publication\scirep_submission\analysis\customq_rerun.csv"), pd.read_csv(os.path.join(A, "customq_400_999.csv"))], ignore_index=True)
cq.to_csv(os.path.join(HERE, "results_1000", "customq_requantisation_1000.csv"), index=False)
log("own-table requantisation (repaired implementation), n = %d covers" % cq.image.nunique())
log(cq.groupby("condition", sort=False).agg(n=("exact", "size"), bit_accuracy=("bit_accuracy", "mean"), header_intact=("header_intact", "sum"), exact=("exact", "sum")).round(4).to_string())
log("additional tests (newrobust.py):"); log(pd.read_excel(os.path.join(C1000, "robustness_customQ_adv_ecc_align.xlsx"), sheet_name="summary_stats").round(4).to_string(index=False))

# ------------------------------------------------------------------ F. detection
log("\n== F. DETECTION")
s = pd.read_excel(os.path.join(C1000, "steganalysis_results.xlsx"), sheet_name="per_image_detection")
log("chi-square statistic decreased after embedding in %d of %d pairs" % ((s.chi_square_delta < 0).sum(), len(s)))
log(open(os.path.join(HERE, "results_1000", "universal_dct_steganalysis_report.txt"), encoding="utf-8").read())
for fn in ("steganalysis_spam.csv", "steganalysis_generic_dct.csv"):
    log(pd.read_csv(os.path.join(A, fn)).round(3).to_string(index=False))
open(os.path.join(HERE, "report_1000.txt"), "w", encoding="utf-8").write("\n".join(L))
