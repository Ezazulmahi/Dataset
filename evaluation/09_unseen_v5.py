"""
09_unseen_v5.py  -  Supplementary Table S7 rerun with the v5 embedder (normal mode):
the 200 cover pairs from prompts outside the bank (08_unseen_prompts), with the SAME
per-pair target payload, K and message (seed 900000 + index) as the original run.
Output: results/unseen_v5.csv, stego_unseen_finetuned/, stego_unseen_baseline/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, time, random, string, contextlib, importlib.util
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
UN = r"F:\scirep results\08_unseen_prompts"
ALPHABET = string.ascii_letters + string.digits + " .,!?"
D = None


def _init():
    global D
    spec = importlib.util.spec_from_file_location("D", os.path.join(HERE, "embedder", "DCT_Adaptive_v5.py"))
    D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


def work(task):
    name, K, n_bits = task
    idx = int(name[:3])
    msg = "".join(random.Random(900000 + idx).choice(ALPHABET) for _ in range(n_bits // 8))
    rows = []
    for tag, folder in (("ft", "finetuned"), ("base", "baseline")):
        cover = cv2.imread(os.path.join(UN, folder, name))
        o = D._compute_K; D._compute_K = lambda b, K=K: K
        t = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                st = D.embed(cover, msg); err = None
        except Exception as e:
            st, err = None, repr(e)
        finally:
            D._compute_K = o
        row = dict(image=name, set=tag, K=K, n_bits=n_bits, bpp=n_bits / 65536, embed_s=time.time() - t, error=err)
        if st is not None:
            p = os.path.join(HERE, f"stego_unseen_{folder}", name); cv2.imwrite(p, st); st = cv2.imread(p)
            with contextlib.redirect_stdout(io.StringIO()):
                row["exact"] = D.extract(st) == msg
            row["fix_passes"] = len([l for l in f.getvalue().splitlines() if "Fix pass" in l]) + 1
            q = D.quality_metrics(cover, st, n_bits)
            row.update(psnr_luma=q["psnr_luma"], ssim_luma=q["ssim_luma"], psnr_color=q["psnr_color"])
        rows.append(row)
    return rows


if __name__ == "__main__":
    e = pd.read_csv(os.path.join(UN, "embed_unseen.csv"))
    e = e[e.set == "ft"]
    for d in ("stego_unseen_finetuned", "stego_unseen_baseline"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    tasks = [(r.image, int(r.K), int(r.n_bits)) for r in e.itertuples()]
    with Pool(3, initializer=_init) as p:
        rows = [r for rs in p.imap_unordered(work, tasks) for r in rs]
    d = pd.DataFrame(rows); d.to_csv(os.path.join(HERE, "results", "unseen_v5.csv"), index=False)
    from scipy.stats import wilcoxon
    ft = d[d.set == "ft"].set_index("image"); bs = d[d.set == "base"].set_index("image")
    print("n", len(ft), "exact ft", ft.exact.sum(), "base", bs.exact.sum(), "errors", d.error.notna().sum())
    print(f"PSNR ft {ft.psnr_luma.mean():.2f} +/- {ft.psnr_luma.std():.2f}  base {bs.psnr_luma.mean():.2f} +/- {bs.psnr_luma.std():.2f}"
          f"  p {wilcoxon(ft.psnr_luma, bs.psnr_luma.loc[ft.index]).pvalue:.2e}")
    print(f"SSIM ft {ft.ssim_luma.mean():.3f} base {bs.ssim_luma.mean():.3f}; fix passes ft {ft.fix_passes.median():.0f} ({ft.fix_passes.max():.0f}) "
          f"base {bs.fix_passes.median():.0f} ({bs.fix_passes.max():.0f})")
