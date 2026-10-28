"""
01_embed_normal_1000.py  -  v5 normal mode on all 1,000 cover pairs.

Protocol (as Table 2 of the manuscript): for pair i the K and the payload length
n_bits of the matched-payload run (comparison_results.xlsx, column K / n_bits,
target payload drawn uniformly from 0.30-0.60 bpp) are forced on BOTH covers, and
the same seed-reproducible ASCII message (seed 424242 + i, alphabet of the
original runs) is embedded.  The stego image is written as PNG, read back, and:

  quality   : luminance PSNR / SSIM, colour PSNR / SSIM (cover vs stego)
  recovery  : extract() byte-for-byte; verify-fix passes; embed / extract time
  processing (fine-tuned AND baseline stego): the Table 4 / Table S1 attacks.
            For each: % of payload bits recovered at the known slots
            (recovered_pct), header intact, exact recovery.

Resumable: images already in the output CSV are skipped.
Usage: python 01_embed_normal_1000.py [workers] [first] [last]
Output: results/normal_per_image.csv  (one row per image x cover set x condition)
        stego_finetuned/, stego_baseline/  (PNG)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, io, time, random, string, contextlib, importlib.util
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = r"F:\scirep results\09_expansion_1000\corpus1000"
OUT = os.path.join(HERE, "results", "normal_per_image.csv")
ALPHABET = string.ascii_letters + string.digits + " .,!?"
D = None

CUSTOM_Q = np.array([[5, 3, 9, 6, 18, 11, 24, 15], [4, 8, 7, 14, 10, 22, 13, 26], [11, 6, 16, 9, 21, 12, 27, 14],
                     [7, 15, 8, 19, 11, 25, 13, 30], [17, 10, 20, 12, 28, 14, 33, 16], [9, 23, 12, 26, 14, 31, 15, 38],
                     [22, 13, 29, 15, 34, 16, 40, 18], [12, 27, 14, 32, 16, 37, 18, 44]], dtype=np.float64)  # Table S1


def _init():
    global D
    spec = importlib.util.spec_from_file_location("DCT_Adaptive_v5", os.path.join(HERE, "embedder", "DCT_Adaptive_v5.py"))
    D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


# ── attacks ──────────────────────────────────────────────────────────────────
def requant_y(bgr, Qs):
    Y, Cr, Cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))
    b, h8, w8 = D._to_blocks(Y)
    return cv2.cvtColor(cv2.merge([D._from_blocks(np.round(b / Qs) * Qs, Y, h8, w8), Cr, Cb]), cv2.COLOR_YCrCb2BGR)


def own_table(bgr, quality):
    scale = 200.0 - 2.0 * quality if quality >= 50 else 5000.0 / quality
    return requant_y(bgr, np.clip(np.floor((D._Q_TABLE.astype(np.float64) * scale + 50.0) / 100.0), 1, 255))


def jpeg(bgr, q):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])[1], cv2.IMREAD_COLOR)


def resize(bgr, f):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(8, round(w * f)), max(8, round(h * f))), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def noise(bgr, s, seed):
    return np.clip(bgr.astype(np.float64) + np.random.default_rng(seed).normal(0, s, bgr.shape), 0, 255).astype(np.uint8)


def crop(bgr, f):
    h, w = bgr.shape[:2]; ch, cw = round(h * f), round(w * f)
    return cv2.resize(bgr[ch:h - ch, cw:w - cw], (w, h), interpolation=cv2.INTER_CUBIC)


def shift(bgr, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def rotate(bgr, a):
    h, w = bgr.shape[:2]
    return cv2.warpAffine(bgr, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)


def crop_pad(bgr, px):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr[px:h - px, px:w - px], (w, h), interpolation=cv2.INTER_LINEAR)


def attacks(seed):
    return [("none", lambda s: s),
            ("own_table_1.0", lambda s: own_table(s, 50)), ("own_table_0.6", lambda s: own_table(s, 70)),
            ("own_table_0.2", lambda s: own_table(s, 90)),
            ("jpeg_95", lambda s: jpeg(s, 95)), ("jpeg_90", lambda s: jpeg(s, 90)), ("jpeg_70", lambda s: jpeg(s, 70)),
            ("jpeg_50", lambda s: jpeg(s, 50)),
            ("resize_0.90", lambda s: resize(s, 0.90)), ("resize_0.75", lambda s: resize(s, 0.75)),
            ("resize_0.50", lambda s: resize(s, 0.50)),
            ("noise_2", lambda s: noise(s, 2, seed)), ("noise_5", lambda s: noise(s, 5, seed)),
            ("noise_10", lambda s: noise(s, 10, seed)), ("noise_15", lambda s: noise(s, 15, seed)),
            ("blur_3x3", lambda s: cv2.GaussianBlur(s, (3, 3), 0.5)), ("crop_5pct", lambda s: crop(s, 0.05)),
            ("custom_q_table", lambda s: requant_y(s, CUSTOM_Q)),
            ("shift_1px", lambda s: shift(s, 1)), ("shift_2px", lambda s: shift(s, 2)),
            ("rotate_1deg", lambda s: rotate(s, 1)), ("crop_pad_2px", lambda s: crop_pad(s, 2))]


def work(task):
    i, K, n_bits = task
    name = f"{i:03d}.png"
    msg = "".join(random.Random(424242 + i).choice(ALPHABET) for _ in range(n_bits // 8))
    bits = np.array([(b >> (7 - k)) & 1 for b in msg.encode() for k in range(8)])
    rows = []
    for cover_set, out_dir in (("finetuned", "stego_finetuned"), ("baseline", "stego_baseline")):
        cover = cv2.imread(os.path.join(CORP, cover_set, name))
        orig = D._compute_K; D._compute_K = lambda blocks, K=K: K
        t = time.time(); f = io.StringIO(); err = None
        try:
            with contextlib.redirect_stdout(f):
                stego = D.embed(cover, msg, mode="normal")
        except Exception as e:
            stego, err = None, repr(e)
        finally:
            D._compute_K = orig
        base = dict(image=name, covers=cover_set, K=K, n_bits=n_bits, bpp=n_bits / 65536, embed_s=time.time() - t, error=err)
        if stego is None:
            rows.append(base); continue
        base["fix_passes"] = len([l for l in f.getvalue().splitlines() if "Fix pass" in l]) + 1
        path = os.path.join(HERE, out_dir, name)
        cv2.imwrite(path, stego); stego = cv2.imread(path)
        q = D.quality_metrics(cover, stego, n_bits)
        base.update(psnr_y=q["psnr_luma"], ssim_y=q["ssim_luma"], psnr_c=q["psnr_color"], ssim_c=q["ssim_color"])
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ok = D.extract(stego) == msg
            except Exception:
                ok = False
        base["extract_s"] = time.time() - t
        arrs = D._slot_arrays(D._build_slot_order(K, 32, 32)[:len(bits)])
        for cond, fn in attacks(i * 2 + (cover_set == "finetuned")):
            a = fn(stego)
            Y = cv2.split(cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb))[0]
            blk = D._to_blocks(Y)[0]
            acc = float(np.mean(D._parities(blk, arrs) == bits))
            hn, hK, _, hok = D._read_header(blk)
            hdr = bool(hok and hn == len(bits) and hK == K)
            exact = ok if cond == "none" else (hdr and acc == 1.0)
            rows.append(dict(base, condition=cond, recovered_pct=100 * acc, header_intact=hdr, exact=exact))
    return rows


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for d in ("stego_finetuned", "stego_baseline"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    c = pd.read_excel(os.path.join(CORP, "comparison_results.xlsx"), sheet_name="per_image_results")
    done = set(pd.read_csv(OUT).image) if os.path.exists(OUT) else set()
    tasks = [(int(r.image[:3]), int(r.K), int(r.n_bits)) for r in c.itertuples()
             if first <= int(r.image[:3]) <= last and r.image not in done]
    print(f"{len(tasks)} pairs to do ({len(done)} already done)", flush=True)
    with Pool(workers, initializer=_init) as p:
        for k, rows in enumerate(p.imap_unordered(work, tasks), 1):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{k}/{len(tasks)} {rows[0]['image']}", flush=True)
