"""Test 3: v5 normal mode with uniformly random bits instead of ASCII text.
100 pairs (every tenth), both covers, same K and payload length as the matched-payload run.
Compares exact recovery, PSNR and SSIM with the ASCII results of the paper for the same images.
Resumable: results/test3_random_bits.csv gets one row per cover as soon as it is done."""
import os, tempfile
import numpy as np, cv2
import common as C

OUT = os.path.join(C.HERE, "results", "test3_random_bits.csv")
D = None


def init():
    global D
    D = C.load_v5()


def work(task):
    i, K, n = task
    name = f"{i:03d}.png"
    bits = np.random.default_rng(900000 + i).integers(0, 2, n)
    rows = []
    for cs in ("finetuned", "baseline"):
        cover = cv2.imread(os.path.join(C.CORP, cs, name))
        stego, passes = C.embed_bits(D, cover, bits, K)
        tmp = os.path.join(tempfile.gettempdir(), f"deds_t3_{os.getpid()}.png")
        stego = C.png_roundtrip(stego, tmp)
        got = C.read_bits(D, stego)
        exact = got is not None and len(got) == n and bool(np.all(got == bits))
        rows.append(dict(key=i, image=name, covers=cs, K=K, n_bits=n, bpp=n / 65536, fix_passes=passes,
                         exact=exact, **C.quality(cover, stego)))
    return rows


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mt = C.matched_tasks()
    tasks = [(i, *mt[i]) for i in range(0, 1000, 10)]
    C.run_resumable(tasks, work, OUT, init=init)
