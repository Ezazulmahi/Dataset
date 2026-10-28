r"""Step 1 - embed the 5,000 training covers with the DEDS embedder (unchanged F:\Results\DCT_Adaptive.py).

Protocol of 08_unseen_prompts\02_embed.py / comparison.py: target payload uniform 0.30-0.60 bpp,
K forced from the target, random ASCII message, PNG round trip, extraction checked.
Seeds (payload 810000, message 910000+i) differ from every seed used for the paper corpus.

Resumable: an image whose stego PNG already exists and is listed in embed_log.csv is skipped,
so the script can be stopped (Ctrl+C or closing the window) and started again at any time.
Covers are processed ft_00000, bs_00000, ft_00001, ... so any finished prefix is balanced.

    python 01_embed_training_stego.py [--workers 10] [--limit N]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, csv, time, random, string, argparse, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ALPHABET = string.ascii_letters + string.digits + " .,!?"
LOG = C.ROOT / "embed_log.csv"
FIELDS = ["image", "order", "target_bpp", "K", "n_bits", "bpp", "embed_s", "exact", "psnr_luma", "ssim_luma", "error"]
_D = None

def _init():
    global _D
    spec = importlib.util.spec_from_file_location("DCT_Adaptive", r"F:\Results\DCT_Adaptive.py")
    _D = importlib.util.module_from_spec(spec); spec.loader.exec_module(_D)

def work(task):
    order, name, target_bpp = task
    cover = cv2.imread(str(C.TRAIN_COVERS / name))
    h, w = cover.shape[:2]
    target_bits = int(round(target_bpp * h * w)); K = max(1, min(_D.MAX_K, int(np.ceil(target_bits / 1022)) - 1))
    msg = "".join(random.Random(910000 + order).choice(ALPHABET) for _ in range(target_bits // 8)); n_bits = 8 * len(msg)
    row = {"image": name, "order": order, "target_bpp": target_bpp, "K": K, "n_bits": n_bits, "bpp": n_bits / (h * w), "error": ""}
    orig = _D._compute_K; _D._compute_K = lambda blocks, K=K: K
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stego = _D.embed(cover, msg)
    except Exception as e:
        stego = None; row["error"] = repr(e)
    _D._compute_K = orig
    row["embed_s"] = round(time.perf_counter() - t0, 2)
    if stego is not None:
        p = C.TRAIN_STEGO / name; tmp = C.TRAIN_STEGO / ("tmp_" + name)
        cv2.imwrite(str(tmp), stego); os.replace(tmp, p)          # never leave a half-written PNG
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                row["exact"] = bool(_D.extract(cv2.imread(str(p))) == msg)
        except Exception as e:
            row["exact"] = False; row["error"] = repr(e)
        qm = _D.quality_metrics(cover, stego, n_bits)
        row["psnr_luma"] = round(float(qm["psnr_luma"]), 3); row["ssim_luma"] = round(float(qm["ssim_luma"]), 5)
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    C.TRAIN_STEGO.mkdir(exist_ok=True)
    for t in C.TRAIN_STEGO.glob("tmp_*.png"):
        t.unlink()
    ft = sorted(p.name for p in C.TRAIN_COVERS.glob("ft_*.png")); bs = sorted(p.name for p in C.TRAIN_COVERS.glob("bs_*.png"))
    names = [n for pair in zip(ft, bs) for n in pair]
    bpp = np.random.default_rng(810000).uniform(0.30, 0.60, len(names))
    done = set()
    if LOG.exists():
        with open(LOG, newline="") as f:
            done = {r["image"] for r in csv.DictReader(f) if not r["error"] and (C.TRAIN_STEGO / r["image"]).exists()}
    tasks = [(i, n, float(bpp[i])) for i, n in enumerate(names) if n not in done]
    if a.limit is not None:
        tasks = tasks[: max(0, a.limit - len(done))]
    print(f"{len(done)} already embedded, {len(tasks)} to do, {a.workers} workers", flush=True)
    if not tasks:
        return
    new = not LOG.exists()
    t0 = time.time()
    with open(LOG, "a", newline="") as f, cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        futs = [ex.submit(work, t) for t in tasks]
        for k, fu in enumerate(cf.as_completed(futs)):
            row = fu.result()
            wr.writerow(row); f.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                el = time.time() - t0; eta = el / (k + 1) * (len(tasks) - k - 1)
                print(f"{len(done) + k + 1}/{len(names)}  last {row['image']} psnr {row.get('psnr_luma')} exact {row.get('exact')}  "
                      f"elapsed {el/60:.0f} min, eta {eta/3600:.1f} h", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
