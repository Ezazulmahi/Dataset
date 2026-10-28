"""Recomputes Supplementary Table S2 from the stored checkpoint covers with the definitions of Table 1:
energy ratio = validate.mid_band_energy_ratio (as for Table 1), eligible count and K = the embedder's rule,
capacity = 1022 (K + 1). Writes results/tableS2_recomputed.csv and compares K with the old record."""
import os, sys, glob, re
import numpy as np, pandas as pd, cv2
import common as C

sys.path.insert(0, r"F:\Results"); sys.argv = ["validate.py"]
import importlib.util
spec = importlib.util.spec_from_file_location("validate", r"F:\Results\validate.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

SWEEP = r"F:\DEDS_project\validation_output\sweep"
D = C.load_v5()
old = pd.read_csv(os.path.join(SWEEP, "sweep_metrics.csv"))
rows = []
for _, o in old.iterrows():
    folder = os.path.join(SWEEP, o.label)
    covers = sorted(glob.glob(os.path.join(folder, "cover_*.png")))
    e, K, cap, ratio = [], [], [], []
    for p in covers:
        img = cv2.imread(p)
        s = C.cover_stats(D, img)
        k = s["K"] if s["K"] is not None else D.FALLBACK_K
        e.append(s["e_bar"]); K.append(k); cap.append(1022 * (k + 1))
        ratio.append(V.mid_band_energy_ratio(V.compute_block_dct(V.uint8_bgr_to_float_rgb(img))))
    rows.append(dict(step=int(o.step), label=o.label, n_covers=len(covers), e_bar=np.mean(e), K_mean=np.mean(K),
                     K_old=o.K_mean, energy_ratio=np.mean(ratio), ratio_old=o.ratio, capacity_bits=np.mean(cap),
                     capacity_bpp=np.mean(cap) / 65536, capacity_old=o.capacity_bits,
                     fallback=int(sum(s is None for s in [C.cover_stats(D, cv2.imread(p))["K"] for p in covers]))))
df = pd.DataFrame(rows).sort_values("step")
df.to_csv(os.path.join(C.HERE, "results", "tableS2_recomputed.csv"), index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
