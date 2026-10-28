r"""Like-for-like timing on one CPU process (AMD Ryzen 5 7500F, no other experiment running):
DEDS embedding (unchanged DCT_Adaptive.embed, K forced as in the matched-payload protocol, PNG write
not included), DEDS extraction, S-UNIWARD and WOW cost computation, and S-UNIWARD cost computation plus
the entropy-bound change simulation. 12 fine-tuned covers: 6 with assigned payload 0.30 to 0.40 bpp and
6 with 0.50 to 0.60 bpp (first six of each band in index order). One warm-up call per function."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
import io, sys, time, random, string, contextlib, importlib.util
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); C1000 = os.path.join(HERE, "corpus1000")
sys.path.insert(0, os.path.join(HERE, "analysis1000"))
import common as C, wow as W
spec = importlib.util.spec_from_file_location("DCT_Adaptive", r"F:\Results\DCT_Adaptive.py"); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
cmp_ = pd.read_excel(os.path.join(C1000, "comparison_results.xlsx"), sheet_name="per_image_results")
lo = cmp_[(cmp_.achieved_bpp < 0.40)].head(6); hi = cmp_[(cmp_.achieved_bpp >= 0.50)].head(6)
AL = string.ascii_letters + string.digits + " .,!?"; rows = []
def t(f):
    t0 = time.perf_counter(); r = f(); return time.perf_counter() - t0, r
y0 = C.read_y(os.path.join(C1000, "finetuned", "000.png")); C.suniward_costs(y0); W.wow_costs(y0)   # warm-up
for band, sub in (("0.30 to 0.40", lo), ("0.50 to 0.60", hi)):
    for _, r in sub.iterrows():
        cover = cv2.imread(os.path.join(C1000, "finetuned", r.image)); y = C.read_y(os.path.join(C1000, "finetuned", r.image))
        msg = "".join(random.Random(5).choice(AL) for _ in range(int(r.n_bits) // 8)); K = int(r.K)
        orig = D._compute_K; D._compute_K = lambda b, K=K: K
        with contextlib.redirect_stdout(io.StringIO()):
            te, stego = t(lambda: D.embed(cover, msg))
        D._compute_K = orig
        ok = cv2.imencode(".png", stego)[1]; st2 = cv2.imdecode(ok, cv2.IMREAD_COLOR)
        with contextlib.redirect_stdout(io.StringIO()):
            tx, out = t(lambda: D.extract(st2))
        tsu, _ = t(lambda: C.suniward_costs(y)); twow, _ = t(lambda: W.wow_costs(y)); tsim, _ = t(lambda: C.suniward(y, int(r.n_bits), 1))
        rows.append({"image": r.image, "band": band, "bpp": r.achieved_bpp, "K": K, "deds_embed_s": te, "deds_extract_s": tx, "exact": out == msg,
                     "suniward_cost_s": tsu, "wow_cost_s": twow, "suniward_cost_plus_simulation_s": tsim})
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(HERE, "timing_single_process.csv"), index=False)
g = d.groupby("band").agg(n=("image", "size"), embed_median=("deds_embed_s", "median"), embed_min=("deds_embed_s", "min"), embed_max=("deds_embed_s", "max"))
print(g.round(1).to_string())
print("extraction median %.3f s | S-UNIWARD cost median %.3f s | WOW cost median %.3f s | S-UNIWARD cost+simulation median %.3f s"
      % (d.deds_extract_s.median(), d.suniward_cost_s.median(), d.wow_cost_s.median(), d.suniward_cost_plus_simulation_s.median()))
