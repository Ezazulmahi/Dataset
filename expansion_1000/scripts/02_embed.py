r"""Expansion to 1,000 pairs, step 2: the matched-payload protocol of F:\Results\comparison.py on pairs
400 to 999. The embedder is the unchanged F:\Results\DCT_Adaptive.py and K is sized with
comparison.k_for_target_bpp(). Differences from comparison.py, none of which changes the method:
the per-image payload and message seeds are fixed integers (comparison.py uses Python hash(), which
changes between runs), the baseline stego image is saved as well, and every stego image is read
back from its PNG file and extracted, so exact recovery is recorded for both cover sets."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, time, random, string, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); NEW = os.path.join(HERE, "new600")
IDX = list(range(400, 1000)); ALPHABET = string.ascii_letters + string.digits + " .,!?"
_D = _CMP = None
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def _init():
    global _D, _CMP
    _D = _load("DCT_Adaptive", r"F:\Results\DCT_Adaptive.py")
    with contextlib.redirect_stdout(io.StringIO()):
        _CMP = _load("comparison_authors", r"F:\Results\comparison.py")

def work(idx):
    name = f"{idx:03d}.png"
    target_bpp = random.Random(1_000_000 + idx).uniform(_CMP.BPP_MIN, _CMP.BPP_MAX)
    cover = {t: cv2.imread(os.path.join(NEW, t, name)) for t in ("baseline", "finetuned")}
    K, target_bits = _CMP.k_for_target_bpp(_D, cover["baseline"], target_bpp)
    rng = random.Random(2_000_000 + idx); msg = "".join(rng.choice(ALPHABET) for _ in range(target_bits // 8)); n_bits = 8 * len(msg)
    row = {"image": name, "target_bpp": target_bpp, "achieved_bpp": n_bits / 65536, "K": K, "n_bits": n_bits}
    for tag, folder in (("baseline", "baseline"), ("finetuned", "finetuned")):
        buf = io.StringIO(); t0 = time.perf_counter()
        orig = _D._compute_K; _D._compute_K = lambda blocks, K=K: K
        try:
            with contextlib.redirect_stdout(buf):
                stego = _D.embed(cover[folder], msg)
        except Exception as e:
            row[tag + "_error"] = repr(e); _D._compute_K = orig; continue
        _D._compute_K = orig
        row[tag + "_embed_s"] = time.perf_counter() - t0; row[tag + "_fix_passes"] = buf.getvalue().count("Fix pass")
        p = os.path.join(NEW, "stego_" + folder, name); cv2.imwrite(p, stego)
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = _D.extract(cv2.imread(p))
        except Exception as e:
            out = None; row[tag + "_error"] = repr(e)
        row[tag + "_extract_s"] = time.perf_counter() - t0; row[tag + "_exact"] = bool(out == msg)
        qm = _D.quality_metrics(cover[folder], stego, n_bits)
        row[tag + "_psnr"] = qm["psnr"]; row[tag + "_ssim"] = qm["ssim"]
        row[tag + "_psnr_color"] = qm["psnr_color"]; row[tag + "_ssim_color"] = qm["ssim_color"]
        row[tag + "_changed_px_frac"] = float((cover[folder] != stego).any(axis=2).mean())
    if "baseline_psnr" in row and "finetuned_psnr" in row:
        row["delta_psnr"] = row["finetuned_psnr"] - row["baseline_psnr"]; row["delta_ssim"] = row["finetuned_ssim"] - row["baseline_ssim"]
    return row

if __name__ == "__main__":
    for f in ("stego_finetuned", "stego_baseline"):
        os.makedirs(os.path.join(NEW, f), exist_ok=True)
    OUT = os.path.join(HERE, "embed_400_999.csv")
    done = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    have = {r["image"] for r in done}; todo = [i for i in IDX if f"{i:03d}.png" not in have]
    rows = done; t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10, initializer=_init) as ex:
        for k, r in enumerate(ex.map(work, todo)):
            rows.append(r)
            if (k + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False); print(k + 1, "of", len(todo), "elapsed %.0fs" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("DONE", len(rows), flush=True)
