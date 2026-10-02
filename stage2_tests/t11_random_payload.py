r"""T11 - does the structure of the test messages (ASCII text, every eighth bit zero) affect detection?
50 fine-tuned covers (every eighth image) are re-embedded with uniformly random bits at the same
K and the same payload length as in the original run. The embedder is the unchanged DCT_Adaptive
(called through the bit-level wrapper of F:\DCT_Adaptive_Robust with one copy, r = 1)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, contextlib, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "t11_stego_random"); os.makedirs(OUT, exist_ok=True)
IDX = list(range(0, 400, 8))

def work(task):
    i, K, n_bits = task
    sys.path.insert(0, r"F:\DCT_Adaptive_Robust")
    import dct_adaptive_robust as RB
    name = f"{i:03d}.png"; cover = cv2.imread(os.path.join(r"F:\Results\finetuned", name))
    B = (n_bits - 32) // 8
    msg = np.random.default_rng(31000 + i).integers(0, 256, B, dtype=np.uint8).tobytes()
    RB.D._compute_K = lambda blocks, K=K: K
    with contextlib.redirect_stdout(io.StringIO()):
        stego = RB.robust_embed(cover, msg, 1)
    p = os.path.join(OUT, name); cv2.imwrite(p, stego)
    with contextlib.redirect_stdout(io.StringIO()):
        out = RB.robust_extract(cv2.imread(p), 1)
    return {"image": name, "K": K, "n_bits": n_bits, "exact": bool(out == msg)}

if __name__ == "__main__":
    d = pd.read_excel(r"F:\Results\comparison_results.xlsx", sheet_name="per_image_results").set_index("image")
    tasks = [(i, int(d.loc[f"{i:03d}.png", "K"]), int(d.loc[f"{i:03d}.png", "n_bits"])) for i in IDX]
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(work, tasks))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(HERE, "t11_embed.csv"), index=False)
    print("embedded", len(r), "| exact recovery", int(r.exact.sum()), flush=True)
