import pandas as pd, numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")
R = r"F:\Results"
d = pd.read_excel(R+r"\comparison_results.xlsx", sheet_name="per_image_results")
print("n", len(d), "achieved bpp min/mean/max", d.achieved_bpp.min(), d.achieved_bpp.mean(), d.achieved_bpp.max())
print("n_bits min/mean/max", d.n_bits.min(), d.n_bits.mean(), d.n_bits.max(), "K range", d.K.min(), d.K.max())
for c in ["baseline_psnr","finetuned_psnr","baseline_ssim","finetuned_ssim","delta_psnr","delta_ssim"]:
    print(c, "mean %.4f sd %.4f min %.4f max %.4f" % (d[c].mean(), d[c].std(), d[c].min(), d[c].max()))
edges = [0.30,0.35,0.40,0.45,0.50,0.55,0.60001]
d["bin"] = pd.cut(d.target_bpp, edges, include_lowest=True)
rows=[]
for b, s in d.groupby("bin", observed=True):
    r = {"bin": str(b), "n": len(s)}
    for m in ["psnr","ssim"]:
        x, y = s["baseline_"+m], s["finetuned_"+m]
        w = stats.wilcoxon(x, y)
        diff = (y-x).values
        # matched-pairs rank-biserial
        rk = stats.rankdata(np.abs(diff)); rp = rk[diff>0].sum(); rn = rk[diff<0].sum()
        rrb = (rp-rn)/(rp+rn)
        r.update({m+"_b": x.mean(), m+"_f": y.mean(), m+"_d": diff.mean(), m+"_p": w.pvalue, m+"_rrb": rrb})
    rows.append(r)
t = pd.DataFrame(rows); pd.set_option("display.width", 250); print(t.to_string())
t.to_csv("table_quality_bins.csv", index=False)
# overall paired tests
for m in ["psnr","ssim"]:
    w = stats.wilcoxon(d["baseline_"+m], d["finetuned_"+m]); print("overall", m, w.pvalue, (d["finetuned_"+m]-d["baseline_"+m]).mean())
x = pd.ExcelFile(R+r"\comparison_results.xlsx")
print(x.parse("FID_CLIP").T); print(x.parse("FID_CLIP_finetuned_vs_stego").T)
c = x.parse("clip_per_image_finetuned_vs_stego"); print("clip ft-vs-stego min", c.clip_cosine_similarity.min())
# steganalysis
s = pd.read_excel(R+r"\steganalysis_results.xlsx", sheet_name="per_image_detection")
from sklearn.metrics import roc_auc_score, roc_curve
y = np.r_[np.zeros(len(s)), np.ones(len(s))]; sc = np.r_[s.cover_pct_stego, s.stego_pct_stego]/100
fpr,tpr,th = roc_curve(y, sc); pe = ((fpr+1-tpr)/2).min()
print("existing detector pooled AUC %.4f  PE %.4f  acc@0.5 %.4f" % (roc_auc_score(y,sc), pe, ((sc>=0.5)==y).mean()))
print("paired: stego score > cover score in", (s.stego_pct_stego>s.cover_pct_stego).mean())
print(pd.read_excel(R+r"\steganalysis_results.xlsx", sheet_name="batch_summary")[["bpp_bin","pairs","accuracy_pct_at_0.5","accuracy_pct_at_best_thresh","best_thresh","roc_auc"]].to_string())
print("chi2 delta mean/median", s.chi_square_delta.mean(), s.chi_square_delta.median(), "stego>cover", (s.chi_square_delta>0).sum())
# robustness
r = pd.ExcelFile(R+r"\robustness_results.xlsx")
print(r.parse("summary_curve")[["distortion","n_images","ber_mean","bit_accuracy_mean","header_survival_rate","extraction_success_rate"]].to_string())
pi = r.parse("per_image_results"); print("robust payload: target_bpp", pi.target_bpp.min(), pi.target_bpp.max(), "n_bits", pi.n_bits.min(), pi.n_bits.max(), "K", pi.K.min(), pi.K.max()); print(pi.error.dropna().unique()[:5])
r2 = pd.ExcelFile(R+r"\robustness_customQ_adv_ecc_align.xlsx"); print(r2.parse("summary_stats").to_string())
print(pd.read_excel(R+r"\comparison_results_pilot.xlsx").to_string())
