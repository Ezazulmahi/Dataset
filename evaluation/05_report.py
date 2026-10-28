"""
05_report.py  -  every number for the revised manuscript from the v5 result files.

Writes results/report_v5.txt and CSV tables in results/tables/:
  table2_quality_v5.csv       Table 2 layout: per assigned-payload bin, PSNR_Y and
                              SSIM_Y of baseline and fine-tuned stego, difference,
                              paired Wilcoxon p, rank-biserial r; plus v5 - v3 change
  recovery_v5.csv             exact recovery, verify-fix passes, colour PSNR, times
  table4_processing_v5.csv    Table 4 / S1: per attack, % of payload bits recovered
                              (mean, s.d.), header intact, exact recovery
  robust_mode_v5.csv          robust mode: per payload level, copies, PSNR/SSIM and
                              % recovered / exact recovery per attack
Works on partial results (reports n).
"""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); TAB = os.path.join(RES, "tables"); os.makedirs(TAB, exist_ok=True)
CMP = r"F:\scirep results\09_expansion_1000\corpus1000\comparison_results.xlsx"
lines = []
def out(s=""):
    print(s); lines.append(str(s))

def rbc(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()

def fmt_p(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"

v3 = pd.read_excel(CMP, sheet_name="per_image_results").set_index("image")
d = pd.read_csv(os.path.join(RES, "normal_per_image.csv"))
q = d[d.condition == "none"].copy()
ft = q[q.covers == "finetuned"].set_index("image"); bs = q[q.covers == "baseline"].set_index("image")
common = ft.index.intersection(bs.index)
ft, bs = ft.loc[common], bs.loc[common]
tb = v3.loc[common, "target_bpp"]
edges = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60001]
lab = ["0.30 to 0.35", "0.35 to 0.40", "0.40 to 0.45", "0.45 to 0.50", "0.50 to 0.55", "0.55 to 0.60"]
binv = pd.cut(tb, edges, right=False, labels=lab)

out(f"== v5 NORMAL MODE, matched payload (Table 2 protocol), n = {len(common)} pairs")
rows = []
for name, idx in [(l, binv.index[binv == l]) for l in lab] + [("All", common)]:
    if len(idx) < 2: continue
    a, b_ = ft.loc[idx], bs.loc[idx]
    rows.append(dict(bin=name, n=len(idx),
        psnr_base=b_.psnr_y.mean(), psnr_ft=a.psnr_y.mean(), d_psnr=(a.psnr_y - b_.psnr_y).mean(),
        p_psnr=wilcoxon(a.psnr_y, b_.psnr_y).pvalue, r_psnr=rbc(a.psnr_y, b_.psnr_y),
        ssim_base=b_.ssim_y.mean(), ssim_ft=a.ssim_y.mean(), d_ssim=(a.ssim_y - b_.ssim_y).mean(),
        p_ssim=wilcoxon(a.ssim_y, b_.ssim_y).pvalue, ssim_ft_higher=int((a.ssim_y > b_.ssim_y).sum()),
        v3_psnr_base=v3.loc[idx, "baseline_psnr"].mean(), v3_psnr_ft=v3.loc[idx, "finetuned_psnr"].mean(),
        gain_vs_v3_base=(b_.psnr_y - v3.loc[idx, "baseline_psnr"]).mean(),
        gain_vs_v3_ft=(a.psnr_y - v3.loc[idx, "finetuned_psnr"]).mean(),
        v3_ssim_ft=v3.loc[idx, "finetuned_ssim"].mean(), v3_ssim_base=v3.loc[idx, "baseline_ssim"].mean()))
t2 = pd.DataFrame(rows); t2.to_csv(os.path.join(TAB, "table2_quality_v5.csv"), index=False)
for r in t2.itertuples():
    out(f"{r.bin:13s} n={r.n:4d} PSNR base {r.psnr_base:.2f} ft {r.psnr_ft:.2f} d {r.d_psnr:+.2f} p {fmt_p(r.p_psnr)} r {r.r_psnr:+.2f} | "
        f"SSIM base {r.ssim_base:.3f} ft {r.ssim_ft:.3f} d {r.d_ssim:+.3f} p {fmt_p(r.p_ssim)} | vs v3: base {r.gain_vs_v3_base:+.2f} dB, ft {r.gain_vs_v3_ft:+.2f} dB")
out(f"PSNR_Y fine-tuned range {ft.psnr_y.min():.1f} to {ft.psnr_y.max():.1f} dB; SSIM_Y ft {ft.ssim_y.mean():.3f} +/- {ft.ssim_y.std():.3f} "
    f"(range {ft.ssim_y.min():.3f} to {ft.ssim_y.max():.3f}); base {bs.ssim_y.mean():.3f} +/- {bs.ssim_y.std():.3f}")

out("\n== RECOVERY AND COST")
rec = []
for cov, g in q.groupby("covers"):
    rec.append(dict(covers=cov, n=len(g), exact=int(g.exact.sum()), errors=int(d[d.covers == cov].error.notna().sum()),
                    fix_median=g.fix_passes.median(), fix_max=g.fix_passes.max(),
                    psnr_c_mean=g.psnr_c.mean(), psnr_c_sd=g.psnr_c.std(), ssim_c_mean=g.ssim_c.mean(),
                    embed_median_s=g.embed_s.median(), embed_max_s=g.embed_s.max(), extract_median_s=g.extract_s.median()))
rec = pd.DataFrame(rec); rec.to_csv(os.path.join(TAB, "recovery_v5.csv"), index=False); out(rec.round(3).to_string(index=False))

out("\n== PROCESSING TESTS, normal mode (% of payload bits recovered at the known slots)")
p = d[d.condition.notna()]
t4 = p.groupby(["covers", "condition"], sort=False).agg(n=("image", "size"), recovered_pct=("recovered_pct", "mean"),
        recovered_sd=("recovered_pct", "std"), header_intact=("header_intact", "sum"), exact=("exact", "sum")).reset_index()
t4.to_csv(os.path.join(TAB, "table4_processing_v5.csv"), index=False)
out(t4[t4.covers == "finetuned"].drop(columns="covers").round(1).to_string(index=False))

rp = os.path.join(RES, "robust_per_image.csv")
if os.path.exists(rp):
    r = pd.read_csv(rp)
    out(f"\n== ROBUST MODE, random subset ({r.image.nunique()} fine-tuned covers so far)")
    meta = r[r.condition == "none"].groupby("level_bpp").agg(n=("image", "size"), copies=("copies", "mean"),
            psnr_y=("psnr_y", "mean"), ssim_y=("ssim_y", "mean"), psnr_c=("psnr_c", "mean"))
    acc = r.pivot_table(index="level_bpp", columns="condition", values="recovered_pct", aggfunc="mean")
    ex = r.pivot_table(index="level_bpp", columns="condition", values="exact", aggfunc="sum")
    order = [c for c in ["none", "own_table_1.0", "own_table_0.6", "own_table_0.2", "jpeg_95", "jpeg_90", "jpeg_70", "jpeg_50",
                         "noise_2", "noise_5", "noise_10", "noise_15", "resize_0.90", "resize_0.75", "resize_0.50",
                         "blur_3x3", "crop_5pct", "custom_q_table", "shift_1px", "shift_2px", "rotate_1deg", "crop_pad_2px"] if c in acc.columns]
    out(meta.round(2).to_string())
    out("\n% of message recovered (mean):"); out(acc[order].round(1).to_string())
    out("\nexact recoveries (count):"); out(ex[order].astype(int).to_string())
    t = meta.join(acc[order].add_prefix("pct_")).join(ex[order].add_prefix("exact_"))
    t.to_csv(os.path.join(TAB, "robust_mode_v5.csv"))

open(os.path.join(RES, "report_v5.txt"), "w", encoding="utf-8").write("\n".join(lines))
