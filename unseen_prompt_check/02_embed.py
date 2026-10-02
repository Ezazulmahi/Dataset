r"""Unseen-prompt check, step 2: cover statistics, then the evaluation protocol of comparison.py on the
200 new pairs (target payload uniform 0.30 to 0.60 bpp, K forced from the target, the same random
ASCII message on both covers of a pair, PNG round trip, exact comparison). The embedder is the
unchanged F:\Results\DCT_Adaptive.py. Seeds are fixed integers (no Python hash())."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, time, random, string, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); N = 200
ALPHABET = string.ascii_letters + string.digits + " .,!?"
_D = None
def _init():
    global _D
    spec = importlib.util.spec_from_file_location("DCT_Adaptive", r"F:\Results\DCT_Adaptive.py")
    _D = importlib.util.module_from_spec(spec); spec.loader.exec_module(_D)

def work(task):
    idx, target_bpp = task
    name = f"{idx:03d}.png"
    target_bits = int(round(target_bpp * 65536)); K = max(1, min(_D.MAX_K, int(np.ceil(target_bits / 1022)) - 1))
    msg = "".join(random.Random(900000 + idx).choice(ALPHABET) for _ in range(target_bits // 8)); n_bits = 8 * len(msg)
    rows = []
    for tag, folder in (("ft", "finetuned"), ("base", "baseline")):
        cover = cv2.imread(os.path.join(HERE, folder, name))
        orig = _D._compute_K; _D._compute_K = lambda blocks, K=K: K
        buf = io.StringIO(); t0 = time.perf_counter(); err = ""; stego = None
        try:
            with contextlib.redirect_stdout(buf):
                stego = _D.embed(cover, msg)
        except Exception as e:
            err = repr(e)
        _D._compute_K = orig
        row = {"image": name, "set": tag, "target_bpp": target_bpp, "K": K, "n_bits": n_bits, "bpp": n_bits / 65536,
               "embed_s": time.perf_counter() - t0, "fix_passes": buf.getvalue().count("Fix pass"), "error": err}
        if stego is not None:
            p = os.path.join(HERE, "stego_" + folder, name); cv2.imwrite(p, stego)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = _D.extract(cv2.imread(p))
            except Exception as e:
                out = None; row["error"] = repr(e)
            row["exact"] = bool(out == msg)
            qm = _D.quality_metrics(cover, stego, n_bits)
            row.update({k: qm[k] for k in ("psnr_luma", "ssim_luma", "psnr_color", "ssim_color")})
        rows.append(row)
    return rows

if __name__ == "__main__":
    sys.path.insert(0, r"F:\Publication\scirep_submission\analysis"); import common as C
    st = []
    for i in range(N):
        for tag, folder in (("ft", "finetuned"), ("base", "baseline")):
            s = C.cover_stats(C.read_y(os.path.join(HERE, folder, f"{i:03d}.png")))
            st.append({"image": f"{i:03d}.png", "set": tag, "mean_eligible": s["mean_eligible"], "K": s["K"], "fallback": s["fallback"],
                       "capacity_bpp": s["capacity_bits"] / 65536})
    pd.DataFrame(st).to_csv(os.path.join(HERE, "cover_stats_unseen.csv"), index=False); print("cover stats written", flush=True)
    for f in ("stego_finetuned", "stego_baseline"):
        os.makedirs(os.path.join(HERE, f), exist_ok=True)
    bpp = np.random.default_rng(700000).uniform(0.30, 0.60, N)
    tasks = sorted([(i, float(bpp[i])) for i in range(N)], key=lambda t: -t[1])
    rows = []; t0 = time.time(); OUT = os.path.join(HERE, "embed_unseen.csv")
    with cf.ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        for k, r in enumerate(ex.map(work, tasks)):
            rows.extend(r)
            if (k + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False); print(k + 1, "of", N, "elapsed %.0fs" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("DONE", len(rows), flush=True)
