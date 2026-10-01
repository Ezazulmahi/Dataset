r"""Step 2 - embed every cover with the universal (generic) DCT embedding.

The embedding is embed_dct_random from F:\Thesis\gen_dataset.py, unchanged: the luminance
channel is transformed with an 8x8 block DCT, quantised with the standard JPEG luminance table,
and the least significant bits of the first rate * N quantised coefficients larger than 1 are
replaced by random bits. The DEDS embedder (DCT_Adaptive.py) is NOT used anywhere in training.

The rate is drawn uniformly from [0.05, 0.15] per image, as in gen_dataset.py.
Output: stego/<same file name>, pairs.csv
"""
import os, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, cv2

ROOT = Path(__file__).resolve().parent
COV = ROOT / "covers"; STG = ROOT / "stego"; STG.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("gen_dataset", r"F:\Thesis\gen_dataset.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

rows = []
files = sorted(p.name for p in COV.glob("*.png"))
for idx, name in enumerate(files):
    cover = cv2.imread(str(COV / name))
    np.random.seed(500000 + idx)
    rate = float(np.random.uniform(0.05, 0.15))
    stego = G.embed_dct_random(cover, rate)
    changed = bool((stego != cover).any())
    cv2.imwrite(str(STG / name), stego)
    mse = ((stego.astype(float) - cover.astype(float)) ** 2).mean()
    rows.append({"image": name, "rate": rate, "changed": changed, "psnr": 10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf})
    if idx % 500 == 0:
        print(idx, name, f"rate {rate:.3f} psnr {rows[-1]['psnr']:.1f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "pairs.csv", index=False)
print("pairs:", len(df), "| unchanged stego:", int((~df.changed).sum()), "| PSNR mean %.2f dB" % df.psnr[np.isfinite(df.psnr)].mean())
