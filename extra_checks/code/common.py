"""Shared helpers for the extra checks. Reads the v5 embedder from the submission folder (read only)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, contextlib, importlib.util
import numpy as np, cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

V5 = r"F:\scirep submission 2.0\3_embedder\DCT_Adaptive_v5.py"
CORP = r"F:\scirep results\09_expansion_1000\corpus1000"
PHOTOS = r"F:\DEDS_project\finetune_extra\covers"
CMP = os.path.join(CORP, "comparison_results.xlsx")
NORMAL_CSV = r"F:\scirep submission 2.0\4_evaluation\results\normal_per_image.csv"


def load_v5():
    spec = importlib.util.spec_from_file_location("DCT_Adaptive_v5", V5)
    D = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(D)
    return D


def luma(bgr):
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0]


def cover_stats(D, bgr):
    """Mean eligible count per non-header block, cover-derived K (None on fallback), raw slot capacity."""
    blocks, _, _ = D._to_blocks(luma(bgr))
    nr, nc = blocks.shape[:2]
    pos = D._PAYLOAD_CANDIDATES
    U = np.array([p[0] for p in pos]); V = np.array([p[1] for p in pos])
    q = np.round(blocks[:, :, U, V] / D._Q_TABLE[U, V])
    elig = (np.abs(q) >= D.ELIGIBILITY_THRESHOLD).sum(axis=2).astype(float)
    mask = np.ones((nr, nc), bool)
    for (i, j) in D.HEADER_BLOCKS:
        mask[i, j] = False
    e_bar = float(elig[mask].mean())
    f = int(e_bar)
    K = None if f <= 0 else max(1, min(D.MAX_K, f + f // 2))
    n_payload = int(mask.sum())
    cap = None if K is None else n_payload * (K + 1)
    return dict(e_bar=e_bar, K=K, capacity_bits=cap, capacity_bpp=None if cap is None else cap / (bgr.shape[0] * bgr.shape[1]))


def spectrum_stats(bgr):
    """Share of luminance power above 0.25 cycles/pixel and log-log slope 0.02 to 0.39 cycles/pixel (Hann window)."""
    y = luma(bgr).astype(np.float64)
    y = y - y.mean()
    h, w = y.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(y * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h)); fx = np.fft.fftshift(np.fft.fftfreq(w))
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    hf = float(P[R > 0.25].sum() / P[R > 0].sum())
    r_int = np.round(R * h).astype(int)
    radial = np.bincount(r_int.ravel(), P.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    freqs = np.arange(len(radial)) / h
    sel = (freqs >= 0.02) & (freqs <= 0.39) & (radial > 0)
    slope = float(np.polyfit(np.log10(freqs[sel]), np.log10(radial[sel]), 1)[0])
    return dict(hf_share=hf, slope=slope)


def embed_bits(D, cover_bgr, bits, K):
    """Normal mode of v5 with an arbitrary bit string, same steps as D._embed_normal but without UTF-8."""
    bits = [int(b) for b in bits]
    n = len(bits)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = D._to_blocks(Y)
    nr, nc = blocks.shape[:2]
    n_payload = sum(1 for i in range(nr) for j in range(nc) if (i, j) not in D._HEADER_BLOCK_SET)
    if n > n_payload * (K + 1):
        raise ValueError("message too large")
    header_bits = D._pack_header(n, K)
    D._write_header(blocks, n, K)
    slot_order = D._build_slot_order(K, nr, nc)
    for k in range(n):
        i, j, u, v = slot_order[k]
        blocks[i, j, u, v] = D._parity_write(blocks[i, j, u, v], u, v, bits[k])
    stego_y = D._from_blocks(blocks, Y, h8, w8)
    state = {}
    for it in range(D.MAX_FIX_ITERATIONS):
        stego_y, rem = D._verify_and_fix(slot_order, stego_y, h8, w8, header_bits, bits, K, state)
        if rem == 0:
            break
    else:
        raise RuntimeError("verify-fix did not converge")
    with contextlib.redirect_stdout(io.StringIO()):
        out = D._exact_y_repair(cover_bgr, stego_y, Cr, Cb, header_bits, bits, slot_order)
    return out, it + 1


def read_bits(D, stego_bgr):
    blocks, _, _ = D._to_blocks(luma(stego_bgr))
    n, K, version, ok = D._read_header(blocks)
    if not ok:
        return None
    nr, nc = blocks.shape[:2]
    slots = D._build_slot_order(K, nr, nc)
    return D._parities(blocks, D._slot_arrays(slots[:n]))


def quality(cover_bgr, stego_bgr):
    a, b = luma(cover_bgr), luma(stego_bgr)
    return dict(psnr_y=float(peak_signal_noise_ratio(a, b, data_range=255)),
                ssim_y=float(structural_similarity(a, b, data_range=255)))


HERE = os.path.dirname(os.path.abspath(__file__))
STOP = os.path.join(HERE, "STOP")


def matched_tasks():
    """(index, K, n_bits) of the matched-payload protocol, one per pair."""
    import pandas as pd
    c = pd.read_excel(CMP, sheet_name="per_image_results").drop_duplicates("image")
    return {int(r.image[:3]): (int(r.K), int(r.n_bits)) for r in c.itertuples()}


def run_resumable(tasks, work, out_csv, init=None, workers=int(os.environ.get("DEDS_WORKERS", "2")), key="key"):
    """Runs work(task) -> list of row dicts in a pool. Each result is appended to out_csv at once,
    tasks whose key is already in out_csv are skipped, and the run stops cleanly when the STOP file appears."""
    import sys, pandas as pd
    from multiprocessing import Pool
    done = set(pd.read_csv(out_csv)[key].astype(str)) if os.path.exists(out_csv) else set()
    todo = [t for t in tasks if str(t[0]) not in done]
    print(f"{os.path.basename(out_csv)}: {len(todo)} to do, {len(tasks) - len(todo)} already saved", flush=True)
    if not todo:
        return True
    if os.path.exists(STOP):
        print("STOP file present, not starting", flush=True); return False
    p = Pool(workers, initializer=init)
    try:
        for k, rows in enumerate(p.imap_unordered(work, todo), 1):
            df = pd.DataFrame(rows)
            if os.path.exists(out_csv):      # keep the column order of the file; refuse rows that would break it
                cols = list(pd.read_csv(out_csv, nrows=0).columns)
                extra = [c for c in df.columns if c not in cols]
                if extra:
                    raise RuntimeError(f"row has columns {extra} that {out_csv} does not have")
                df = df.reindex(columns=cols)
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            print(f"  {k}/{len(todo)} saved ({rows[0][key]})", flush=True)
            if os.path.exists(STOP):
                print("STOP file found: everything finished so far is saved, exiting", flush=True)
                p.terminate(); return False
    finally:
        p.terminate(); p.join()
    return True


def png_roundtrip(bgr, path):
    cv2.imwrite(path, bgr)
    return cv2.imread(path)
