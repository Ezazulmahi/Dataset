"""(a) capacity on the same pairs; (b) share of the payload slots actually used that hold an eligible coefficient."""
import os, sys
import numpy as np, pandas as pd, cv2
from scipy.stats import wilcoxon
sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

D = C.load_v5()
cs = pd.read_csv(r"F:\scirep results\09_expansion_1000\analysis1000\cover_stats.csv")
b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
keep = b.index[~b.fallback]
print(f"(a) pairs whose baseline cover has a cover-derived K: {len(keep)}")
print(f"    baseline {b.loc[keep].capacity_bpp.mean():.3f} bpp, fine-tuned on the same pairs {f.loc[keep].capacity_bpp.mean():.3f} bpp, "
      f"ratio {f.loc[keep].capacity_bpp.mean()/b.loc[keep].capacity_bpp.mean():.2f}, fine-tuned higher in {(f.loc[keep].capacity_bpp > b.loc[keep].capacity_bpp).sum()} pairs, "
      f"p = {wilcoxon(f.loc[keep].capacity_bpp, b.loc[keep].capacity_bpp).pvalue:.2g}")
print(f"    e_bar on the 434 fallback pairs: baseline {b[b.fallback].mean_eligible.mean():.2f}, fine-tuned {f.loc[b.index[b.fallback]].mean_eligible.mean():.2f}")

mt = C.matched_tasks()
rows = []
for i in range(1000):
    K, n = mt[i]
    slots = D._build_slot_order(K, 32, 32)[:n]
    I, J, U, V, Q = D._slot_arrays(slots)
    for cs_name in ("finetuned", "baseline"):
        blocks, _, _ = D._to_blocks(C.luma(cv2.imread(os.path.join(C.CORP, cs_name, f"{i:03d}.png"))))
        q = np.round(blocks[I, J, U, V] / Q)
        rows.append(dict(i=i, covers=cs_name, elig=float(np.mean(np.abs(q) >= 2)), small=float(np.mean(np.abs(q) <= 1))))
r = pd.DataFrame(rows)
g = r.groupby("covers")[["elig", "small"]].mean()
print("(b) share of the payload slots actually used at the matched payload that are eligible (|q| >= 2) / hold |q| <= 1:")
print(g.round(3).to_string())
p = r.pivot(index="i", columns="covers", values="elig")
print(f"    fine-tuned higher in {(p.finetuned > p.baseline).sum()} of 1000 pairs, p = {wilcoxon(p.finetuned, p.baseline).pvalue:.2g}")
r.to_csv(r"F:\DEDS_extra_checks\results\used_slot_eligibility.csv", index=False)
