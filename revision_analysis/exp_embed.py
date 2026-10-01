"""Re-embed every baseline / fine-tuned cover with a known, seed-reproducible message at the
payload recorded in comparison_results.xlsx; verify byte-exact recovery after a PNG
round trip; record runtime, verify-fix passes and distortion."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, time, random, string, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
RES = r"F:\Results"; OUT = "embed_rerun.csv"
ALPHABET = string.ascii_letters + string.digits + " .,!?"
_D = None
def _init():
    global _D
    spec = importlib.util.spec_from_file_location("DCT_Adaptive", RES + r"\DCT_Adaptive.py")
    _D = importlib.util.module_from_spec(spec); spec.loader.exec_module(_D)

def work(task):
    idx, K, n_bits = task
    name = f"{idx:03d}.png"
    msg = "".join(random.Random(424242 + idx).choice(ALPHABET) for _ in range(n_bits // 8))
    rows = []
    for tag, folder, outdir in (("ft", "finetuned", "stego_ft"), ("base", "baseline", "stego_base")):
        cover = cv2.imread(os.path.join(RES, folder, name))
        orig = _D._compute_K; _D._compute_K = lambda blocks, K=K: K
        buf = io.StringIO(); t0 = time.perf_counter(); err = ""
        try:
            with contextlib.redirect_stdout(buf):
                stego = _D.embed(cover, msg)
        except Exception as e:
            stego = None; err = repr(e)
        t_embed = time.perf_counter() - t0
        _D._compute_K = orig
        row = {"image": name, "set": tag, "K": K, "n_bits": n_bits, "bpp": n_bits / 65536, "embed_s": t_embed,
               "fix_passes": buf.getvalue().count("Fix pass"), "error": err}
        if stego is not None:
            p = os.path.join(outdir, name); cv2.imwrite(p, stego)
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = _D.extract(cv2.imread(p))
            except Exception as e:
                out = None; row["error"] = repr(e)
            row["extract_s"] = time.perf_counter() - t0
            row["exact"] = bool(out == msg)
            if out is not None and len(out) == len(msg):
                a = np.unpackbits(np.frombuffer(msg.encode(), np.uint8)); b = np.unpackbits(np.frombuffer(out.encode("utf-8", "replace")[:len(msg)], np.uint8))
                row["ber"] = float((a != b).mean()) if len(a) == len(b) else np.nan
            qm = _D.quality_metrics(cover, stego, n_bits)
            row.update({k: qm[k] for k in ("psnr_luma", "ssim_luma", "psnr_color", "ssim_color")})
            row["changed_px_frac"] = float((cover != stego).any(axis=2).mean())
        rows.append(row)
    return rows

if __name__ == "__main__":
    os.makedirs("stego_ft", exist_ok=True); os.makedirs("stego_base", exist_ok=True)
    d = pd.read_excel(RES + r"\comparison_results.xlsx", sheet_name="per_image_results").set_index("image")
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set(done.image) if len(done) else set()
    tasks = [(i, int(d.loc[f"{i:03d}.png", "K"]), int(d.loc[f"{i:03d}.png", "n_bits"])) for i in range(400) if f"{i:03d}.png" not in have]
    tasks.sort(key=lambda t: -t[1])                       # slowest (largest K) first
    rows = done.to_dict("records"); t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10, initializer=_init) as ex:
        for k, r in enumerate(ex.map(work, tasks)):
            rows.extend(r)
            if (k + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False); print(k + 1, "of", len(tasks), "elapsed %.0fs" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("DONE", len(rows), flush=True)
