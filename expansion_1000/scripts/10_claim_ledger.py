r"""Claim ledger: recompute each key manuscript number from the source files and confirm that the
manuscript states it. Output: claim_ledger.csv and claim_ledger.txt (status Verified / MISMATCH)."""
import os, re, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "corpus1000"); A = os.path.join(HERE, "analysis1000")
SUB = "F:/Publication/scirep_submission/"
tex = open(SUB + "main.tex", encoding="utf-8").read() + open(SUB + "supplementary.tex", encoding="utf-8").read()
cs = pd.read_csv(os.path.join(A, "cover_stats.csv")); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
cmp_ = pd.read_excel(os.path.join(C, "comparison_results.xlsx"), sheet_name=None)
d = cmp_["per_image_results"]; e = pd.read_csv(os.path.join(A, "embed_rerun.csv"))
rob = pd.read_excel(os.path.join(C, "robustness_results.xlsx"), sheet_name="per_image_results")
cq = pd.read_csv(os.path.join(HERE, "results_1000", "customq_requantisation_1000.csv"))
nr = pd.read_excel(os.path.join(C, "robustness_customQ_adv_ecc_align.xlsx"), sheet_name="summary_stats")
ch = pd.read_excel(os.path.join(C, "steganalysis_results.xlsx"), sheet_name="per_image_detection")
cnn = pd.read_excel(os.path.join(HERE, "results_1000", "universal_dct_steganalysis_results.xlsx"), sheet_name="batch_summary")
cf = cnn[(cnn.set == "DEDS on fine-tuned covers") & (cnn.bpp_bin == "all")].iloc[0]; cb = cnn[(cnn.set == "DEDS on baseline covers") & (cnn.bpp_bin == "all")].iloc[0]
sp = pd.concat([pd.read_csv(os.path.join(A, "steganalysis_spam.csv")), pd.read_csv(os.path.join(A, "steganalysis_generic_dct.csv"))]).set_index("experiment")
g = lambda k: sp.loc[k]
nb = b[~b.fallback]; ft_e = e[e.set == "ft"]; bs_e = e[e.set == "base"]
r_ok = lambda cond, col: rob[(rob.distortion == cond) & rob.error.isna()][col]
rows = []
def claim(cid, where, text, value, source):
    ok = text in tex
    rows.append({"id": cid, "location": where, "manuscript text": text, "recomputed": value, "source": source, "status": "Verified" if ok else "MISMATCH (text not found)"})

claim("C01", "Abstract, Table 1", r"from 1.57 to 16.64", "%.2f -> %.2f" % (b.mean_eligible.mean(), f.mean_eligible.mean()), "analysis1000/cover_stats.csv")
claim("C02", "Abstract", r"from 0.056 (566 baseline covers) to 0.389", "%.3f (n=%d) -> %.3f" % (nb.capacity_bpp.mean(), len(nb), f.capacity_bpp.mean()), "cover_stats.csv")
claim("C03", "Results", r"an increase observed in all 1,000 pairs", "%d of 1000" % (f.mean_eligible.values > b.mean_eligible.values).sum(), "cover_stats.csv")
claim("C04", "Results", r"$p=3.3\times10^{-165}$", "%.2g" % stats.wilcoxon(b.mean_eligible, f.mean_eligible).pvalue, "cover_stats.csv")
claim("C05", "Table 1", r"434 of 1,000 & 0 of 1,000", "%d / %d" % (b.fallback.sum(), f.fallback.sum()), "cover_stats.csv")
claim("C06", "Table 1, Results", r"924 of 1,000", "%d" % (f.capacity_bpp >= 0.30).sum(), "cover_stats.csv")
claim("C07", "Results", r"25,471 bits (0.389\bpp; range 0.156 to 0.577)", "%.0f bits, %.3f to %.3f" % (f.capacity_bits.mean(), f.capacity_bpp.min(), f.capacity_bpp.max()), "cover_stats.csv")
claim("C08", "Results", r"13,163 bits (0.201\bpp)", "%.0f bits, %.3f" % (b.capacity_bits.mean(), b.capacity_bpp.mean()), "cover_stats.csv")
claim("C09", "Results", r"For 688 of the 1,000 fine-tuned covers", "%d" % (d.K.values > f.K.reindex(d.image).values).sum(), "comparison_results.xlsx + cover_stats.csv")
claim("C10", "Results", r"19,680 to 39,304 bits; $K$ from 19 to 38", "%d to %d; K %d to %d" % (d.n_bits.min(), d.n_bits.max(), d.K.min(), d.K.max()), "comparison_results.xlsx")
claim("C11", "Results (macro)", r"\newcommand{\RecFT}{1,000 of 1,000}", "ft %d/%d, base %d/%d" % (ft_e.exact.sum(), len(ft_e), bs_e.exact.sum(), len(bs_e)), "analysis1000/embed_rerun.csv")
claim("C12", "Results (macro)", r"\newcommand{\FixMaxFT}{17}", "ft median %d max %d; base median %d max %d" % (ft_e.fix_passes.median(), ft_e.fix_passes.max(), bs_e.fix_passes.median(), bs_e.fix_passes.max()), "embed_rerun.csv")
claim("C13", "Results (macro)", r"\newcommand{\PSNRcolFT}{$32.16\pm2.16$}", "%.2f +/- %.2f" % (ft_e.psnr_color.mean(), ft_e.psnr_color.std()), "embed_rerun.csv")
claim("C14", "Results", r"$0.909\pm0.055$ (0.650 to 0.993)", "%.3f +/- %.3f (%.3f to %.3f)" % (d.baseline_ssim.mean(), d.baseline_ssim.std(), d.baseline_ssim.min(), d.baseline_ssim.max()), "comparison_results.xlsx")
claim("C15", "Results", r"SSIM was $0.990\pm0.009$ (range 0.934 to 0.999)", "%.3f +/- %.3f (%.3f to %.3f)" % (d.finetuned_ssim.mean(), d.finetuned_ssim.std(), d.finetuned_ssim.min(), d.finetuned_ssim.max()), "comparison_results.xlsx")
claim("C16", "Results", r"higher in every one of the 1,000 pairs", "%d" % (d.delta_ssim > 0).sum(), "comparison_results.xlsx")
claim("C17", "Results", r"between 29.9 and 36.8\,dB", "%.1f to %.1f" % (d.finetuned_psnr.min(), d.finetuned_psnr.max()), "comparison_results.xlsx")
m = (d.target_bpp >= .45) & (d.target_bpp < .50)
claim("C18", "Results", r"$+0.07$\,dB at 0.45 to 0.50\bpp{} ($p=0.046$)", "%+.3f p=%.3f" % (d.delta_psnr[m].mean(), stats.wilcoxon(d.baseline_psnr[m], d.finetuned_psnr[m]).pvalue), "comparison_results.xlsx")
claim("C19", "Results", r"is 0.830 (95\% confidence interval 0.827 to 0.833)", "%.3f (%.3f to %.3f)" % tuple(cmp_["FID_CLIP"].iloc[0][["CLIP_cosine_mean", "CLIP_cosine_ci_low", "CLIP_cosine_ci_high"]]), "comparison_results.xlsx FID_CLIP")
claim("C20", "Results", r"between the two sets is 238", "%.1f" % cmp_["FID_CLIP"].iloc[0, 0], "comparison_results.xlsx FID_CLIP")
c2 = cmp_["clip_per_image_finetuned_vs_stego"].clip_cosine_similarity
claim("C21", "Results", r"(CLIP similarity 0.9993, minimum 0.9921; FID 3.6)", "%.4f min %.4f FID %.2f" % (c2.mean(), c2.min(), cmp_["FID_CLIP_finetuned_vs_stego"].iloc[0, 0]), "comparison_results.xlsx")
claim("C22", "Results", r"flagged 835 of 1,000 stego images and no cover", "%d stego, %d covers" % (cb.stego_flagged_at_val_threshold, cb.covers_flagged_at_val_threshold), "results_1000/universal_dct_steganalysis_results.xlsx")
claim("C23", "Results", r"flagged none of the 1,000 DEDS stego images of fine-tuned covers", "%d" % cf.stego_flagged_at_val_threshold, "universal_dct_steganalysis_results.xlsx")
claim("C24", "Results", r"it flagged 2 of 1,000 fine-tuned and 981 of 1,000", "%d / %d" % (cf["stego_flagged_at_0.5"], cb["stego_flagged_at_0.5"]), "universal_dct_steganalysis_results.xlsx")
claim("C25", "Results", r"(AUC) of 0.851 (95\% confidence interval 0.833 to 0.868)", "%.3f (%.3f to %.3f)" % (cf.roc_auc, cf.roc_auc_ci_low, cf.roc_auc_ci_high), "universal_dct_steganalysis_results.xlsx")
claim("C26", "Results", r"in 87\% of pairs", "%.3f" % cf.stego_score_above_cover_frac, "universal_dct_steganalysis_results.xlsx")
claim("C27", "Results", r"would give $P_\mathrm{E}=0.18$", "%.3f" % cf.P_E_min, "universal_dct_steganalysis_results.xlsx")
claim("C28", "Abstract, Discussion", r"84\% in baseline covers", "%.1f%%" % (100 * cb.stego_flagged_at_val_threshold / 1000), "universal_dct_steganalysis_results.xlsx")
claim("C29", "Table 3", r"SPAM (DEDS, same source) & 0.006 & 1.000", "PE %.3f AUC %.3f" % tuple(g("DEDS embedder, fine-tuned covers (matched detector)")[["PE", "AUC"]]), "analysis1000/steganalysis_spam.csv")
claim("C30", "Table 3", r"SPAM (S-UNIWARD, same source) & 0.128 & 0.938", "PE %.3f AUC %.3f" % tuple(g("DEDS embedder, fine-tuned covers (detector trained on S-UNIWARD)")[["PE", "AUC"]]), "steganalysis_spam.csv")
claim("C31", "Results", r"does not transfer ($P_\mathrm{E}=0.25$)", "%.3f" % g("DEDS embedder, fine-tuned covers (detector trained on generic DCT, 5-15% rate)").PE, "steganalysis_generic_dct.csv")
claim("C32", "Results", r"decreased after embedding in 967 of 1,000 pairs", "%d" % (ch.chi_square_delta < 0).sum(), "corpus1000/steganalysis_results.xlsx")
claim("C33", "Results, Table 4", r"JPEG, quality 90 / 70 / 50 & 72.7 / 47.4 / 50.8", " / ".join("%.1f" % (100 * r_ok(c, "bit_accuracy").mean()) for c in ("jpeg_standard_QF90", "jpeg_standard_QF70", "jpeg_standard_QF50")), "corpus1000/robustness_results.xlsx")
claim("C34", "Results, Table 4", r"The header survived in 4 of 1,000 images", "%d" % r_ok("gaussian_noise_sigma5", "header_survives").sum(), "robustness_results.xlsx")
q = cq.groupby("condition").agg(acc=("bit_accuracy", "mean"), h=("header_intact", "sum"), x=("exact", "sum"))
claim("C35", "Results, Table 4", r"99.0 / 96.2 / 99.9 & 519 / 104 / 988 & 0 / 0 / 20", "%.1f/%.1f/%.1f ; %d/%d/%d ; %d/%d/%d" % (100 * q.acc.customQ_QF90, 100 * q.acc.customQ_QF70, 100 * q.acc.customQ_QF50, q.h.customQ_QF90, q.h.customQ_QF70, q.h.customQ_QF50, q.x.customQ_QF90, q.x.customQ_QF70, q.x.customQ_QF50), "results_1000/customq_requantisation_1000.csv")
claim("C36", "Results, Discussion", r"only 20 of 1,000 messages were recovered", "%d" % q.x.customQ_QF50, "customq_requantisation_1000.csv")
claim("C37", "Results, Table 4 caption", r"5,104 to 18,904 bits", "%d to %d" % (rob.n_bits.min(), rob.n_bits.max()), "robustness_results.xlsx")
claim("C38", "Supp. Table S1", r"Control & unmodified stego image & 1,000 of 1,000", "%d of %d" % (round(nr[nr.test_type == "custom_q_control"].exact_match_rate.iloc[0] * 1000), nr[nr.test_type == "custom_q_control"].n_images.iloc[0]), "corpus1000/robustness_customQ_adv_ecc_align.xlsx")
claim("C39", "Results", r"rose from 0.051 to 0.354", "see report_1000.txt (validate.mid_band_energy_ratio)", "analysis1000/midratio_*.npy")
mb, mf = np.load(os.path.join(A, "midratio_baseline.npy")), np.load(os.path.join(A, "midratio_finetuned.npy")); rows[-1]["recomputed"] = "%.3f -> %.3f" % (mb.mean(), mf.mean())
L = pd.DataFrame(rows); L.to_csv(os.path.join(HERE, "claim_ledger.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
txt = L[["id", "location", "recomputed", "status"]].to_string(index=False); open(os.path.join(HERE, "claim_ledger.txt"), "w", encoding="utf-8").write(txt); print(txt)
print("\n%d of %d verified" % ((L.status == "Verified").sum(), len(L)))
