"""Cover statistics for the 400 baseline / fine-tuned pairs (no embedding)."""
import numpy as np, pandas as pd, common as C
rows = []; maps = {"base": [], "ft": []}; emaps = {"base": [], "ft": []}
for i in range(400):
    n = f"{i:03d}.png"
    for tag, d in (("base", C.BASE_DIR), ("ft", C.FT_DIR)):
        st = C.cover_stats(C.read_y(d + "\\" + n))
        maps[tag].append(st["elig_map"]); emaps[tag].append(st["energy_map"])
        rows.append({"image": n, "set": tag, "mean_eligible": st["mean_eligible"], "K": st["K"],
                     "fallback": st["fallback"], "capacity_bits": st["capacity_bits"],
                     "capacity_bpp": st["capacity_bits"] / 65536})
df = pd.DataFrame(rows); df.to_csv("cover_stats.csv", index=False)
np.savez("elig_maps.npz", base=np.mean(maps["base"], 0), ft=np.mean(maps["ft"], 0),
         ebase=np.mean(emaps["base"], 0), eft=np.mean(emaps["ft"], 0))
g = df.groupby("set")
print(g[["mean_eligible", "K", "capacity_bits", "capacity_bpp"]].agg(["mean", "std", "min", "median", "max"]).T.to_string())
print("fallback counts", g.fallback.sum().to_dict())
nf = df[~df.fallback].groupby("set")[["K", "capacity_bpp", "mean_eligible"]].agg(["mean", "std"]); print("excluding fallback\n", nf.T.to_string())
b = df[df.set == "base"].set_index("image"); f = df[df.set == "ft"].set_index("image")
from scipy import stats
print("paired wilcoxon mean_eligible", stats.wilcoxon(b.mean_eligible, f.mean_eligible).pvalue, "ft>base in", (f.mean_eligible > b.mean_eligible).mean())
print("capacity_bpp ft>=0.3:", (f.capacity_bpp >= 0.3).mean(), " base>=0.3:", (b.capacity_bpp >= 0.3).mean(), "(incl. fallback)")
print("CV mean_eligible base %.3f ft %.3f" % (b.mean_eligible.std() / b.mean_eligible.mean(), f.mean_eligible.std() / f.mean_eligible.mean()))
m = np.load("elig_maps.npz"); np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("eligibility rate map base\n", m["base"], "\nft\n", m["ft"])
