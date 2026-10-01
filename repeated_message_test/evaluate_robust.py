r"""End-to-end robustness test of the robust mode on the 400 fine-tuned covers.

For each repetition factor r, every cover is embedded once with a random message that fills
capacity/r (adaptive K, as in the original robustness test), the stego image is processed, and
robust_extract() must return the exact message. Distortions are those of F:\Results\Robustness.py
plus requantisation with the embedder's own table (with the reassembly bug fixed)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
R_LIST = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3, 9, 31, 101]
N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 400
_RB = _RO = None

def _init():
    global _RB, _RO
    sys.path.insert(0, HERE)
    import dct_adaptive_robust as RB
    spec = importlib.util.spec_from_file_location("Robustness", r"F:\Results\Robustness.py"); RO = importlib.util.module_from_spec(spec); spec.loader.exec_module(RO)
    _RB, _RO = RB, RO

def own_table(img, scale_quality):
    D = _RB.D; scale = _RO._ijg_scale(scale_quality)
    Qs = np.clip(np.floor((D._Q_TABLE * scale + 50.0) / 100.0), 1, 255)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = _RO._blockify(Y)
    q = np.round(_RO._dct2_batch(blocks - 128.0) / Qs) * Qs
    rec = np.clip(np.round(_RO._idct2_batch(q) + 128.0), 0, 255)
    Yo = Y.copy(); Yo[:h8, :w8] = _RO._deblockify(rec, h8, w8).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Yo, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

CONDS = [("none", lambda s: s.copy()),
         ("own table x1.0", lambda s: own_table(s, 50)), ("own table x0.6", lambda s: own_table(s, 70)), ("own table x0.2", lambda s: own_table(s, 90)),
         ("JPEG 95", lambda s: _RO.apply_distortion(_RB.D, s, "jpeg_standard", 95)),
         ("JPEG 90", lambda s: _RO.apply_distortion(_RB.D, s, "jpeg_standard", 90)),
         ("JPEG 70", lambda s: _RO.apply_distortion(_RB.D, s, "jpeg_standard", 70)),
         ("JPEG 50", lambda s: _RO.apply_distortion(_RB.D, s, "jpeg_standard", 50)),
         ("noise sigma 5", lambda s: _RO.apply_distortion(_RB.D, s, "gaussian_noise", 5.0)),
         ("noise sigma 10", lambda s: _RO.apply_distortion(_RB.D, s, "gaussian_noise", 10.0)),
         ("blur 3x3", lambda s: _RO.apply_distortion(_RB.D, s, "gaussian_blur_mild", None)),
         ("resize 0.90", lambda s: _RO.apply_distortion(_RB.D, s, "resize", 0.90)),
         ("crop 5%", lambda s: _RO.apply_distortion(_RB.D, s, "crop_mild", 0.05))]

def work(task):
    i, r = task
    name = f"{i:03d}.png"; cover = cv2.imread(os.path.join(r"F:\Results\finetuned", name))
    B = _RB.max_message_bytes(cover, r)
    if B < 1:
        return [{"image": name, "r": r, "condition": "ALL", "error": "capacity too small"}]
    msg = np.random.default_rng(900000 + 1000 * r + i).integers(0, 256, B, dtype=np.uint8).tobytes()
    with contextlib.redirect_stdout(io.StringIO()):
        stego = _RB.robust_embed(cover, msg, r)
    y0 = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))[0].astype(float); y1 = cv2.split(cv2.cvtColor(stego, cv2.COLOR_BGR2YCrCb))[0].astype(float)
    psnr = 10 * np.log10(255 ** 2 / max(((y0 - y1) ** 2).mean(), 1e-12))
    rows = []
    for label, fn in CONDS:
        np.random.seed(1234 + i)
        d = fn(stego)
        with contextlib.redirect_stdout(io.StringIO()):
            out = _RB.robust_extract(d, r)
        rows.append({"image": name, "r": r, "condition": label, "message_bytes": B, "net_bpp": 8 * B / 65536, "coded_bpp": r * (32 + 8 * B) / 65536,
                     "psnr_y": psnr, "exact": bool(out == msg), "wrong_message": bool(out is not None and out != msg)})
    return rows

if __name__ == "__main__":
    tasks = [(i, r) for r in R_LIST for i in range(0, 400, max(1, 400 // N_IMG))]   # evenly spaced over the 40 prompt subjects
    rows = []
    with cf.ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        for k, res in enumerate(ex.map(work, tasks)):
            rows.extend(res)
            if (k + 1) % 100 == 0: print(k + 1, "of", len(tasks), flush=True)
    d = pd.DataFrame(rows); out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
    tag = "_".join(map(str, R_LIST))
    summ = d.groupby(["r", "condition"], sort=False).agg(images=("exact", "size"), exact=("exact", "sum"), wrong=("wrong_message", "sum"),
                                                        net_bpp=("net_bpp", "mean"), message_bytes=("message_bytes", "mean"), psnr_y=("psnr_y", "mean")).reset_index()
    with pd.ExcelWriter(os.path.join(out, f"robust_mode_results_r{tag}.xlsx"), engine="openpyxl") as w:
        d.to_excel(w, sheet_name="per_image", index=False); summ.to_excel(w, sheet_name="summary", index=False)
    pd.set_option("display.width", 200)
    piv = summ.pivot(index="condition", columns="r", values="exact").reindex([c for c, _ in CONDS])
    print(piv.to_string()); print(summ.groupby("r")[["net_bpp", "message_bytes", "psnr_y", "wrong"]].agg({"net_bpp": "mean", "message_bytes": "mean", "psnr_y": "mean", "wrong": "sum"}).round(4).to_string())
