"""
02_robust_mode_subset.py  -  v5 robust mode on a random subset of fine-tuned covers.

200 fine-tuned covers are drawn at random (numpy seed 2026) from the 1,000 and each
is assigned, at random, one of eight net payloads (25 covers per level):
    0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6 bits per pixel of message.
Robust mode spreads the framed message over every slot (copies = slots / frame
bits, about 0.98 at 0.6 bpp and about 120 at 0.005 bpp) and decodes by soft voting.

Per stego image and attack (same attacks as 01_embed_normal_1000.py):
  recovered_pct : % of message bits correct after soft voting (frame length known)
  exact         : the extractor returned the message byte for byte (header, or
                  length search when the header is lost; CRC-16 checked)
Output: results/robust_per_image.csv, stego_robust/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, io, time, random, string, contextlib, importlib.util
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("E", os.path.join(HERE, "01_embed_normal_1000.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
OUT = os.path.join(HERE, "results", "robust_per_image.csv")
LEVELS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6]
N_SLOTS = 1022 * 39


def _init():
    E._init()


def work(task):
    i, bpp = task
    D = E.D
    name = f"{i:03d}.png"
    cover = cv2.imread(os.path.join(E.CORP, "finetuned", name))
    nb = min(max(1, int(round(bpp * 65536 / 8))), (N_SLOTS - 32) // 8)
    msg = "".join(random.Random(777000 + i).choice(E.ALPHABET) for _ in range(nb))
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        stego = D.embed(cover, msg, mode="robust", verbose=False)
    emb = time.time() - t
    path = os.path.join(HERE, "stego_robust", name)
    cv2.imwrite(path, stego); stego = cv2.imread(path)
    q = D.quality_metrics(cover, stego, 8 * nb)
    base = dict(image=name, level_bpp=bpp, net_bpp=8 * nb / 65536, msg_bytes=nb, copies=N_SLOTS / (8 * nb + 32),
                psnr_y=q["psnr_luma"], ssim_y=q["ssim_luma"], psnr_c=q["psnr_color"], embed_s=emb)
    rows = []
    for cond, fn in E.attacks(i * 2 + 1):
        a = fn(stego)
        try:
            ex = D.robust_extract(a) == msg
        except Exception:
            ex = False
        rows.append(dict(base, condition=cond, recovered_pct=100 * D.robust_bit_recovery(a, msg), exact=ex))
    return rows


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rng = np.random.default_rng(2026)
    imgs = rng.choice(1000, size=200, replace=False)
    lv = np.repeat(LEVELS, 25); rng.shuffle(lv)
    os.makedirs(os.path.join(HERE, "stego_robust"), exist_ok=True)
    pd.DataFrame(dict(image=[f"{i:03d}.png" for i in imgs], level_bpp=lv)).to_csv(
        os.path.join(HERE, "results", "robust_subset_assignment.csv"), index=False)
    done = set(pd.read_csv(OUT).image) if os.path.exists(OUT) else set()
    tasks = [(int(i), float(b)) for i, b in zip(imgs, lv) if f"{int(i):03d}.png" not in done]
    print(f"{len(tasks)} covers to do", flush=True)
    with Pool(workers, initializer=_init) as p:
        for k, rows in enumerate(p.imap_unordered(work, tasks), 1):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{k}/{len(tasks)} {rows[0]['image']} {rows[0]['level_bpp']}", flush=True)
