r"""Step 0 - rebuild the universal-DCT training set without the JPEG-compression confound.

The original set (F:\Steganalyzer_UniversalDCT) pairs an UNCOMPRESSED cover with a stego image made by
embed_dct_random (F:\DEDS_project\gen_dataset.py), which requantises every 8x8 luminance block with the
standard JPEG luminance table (quality 50) and only then flips coefficient LSBs. The cover/stego
difference is therefore mostly JPEG compression (PSNR 34.8 dB) and only slightly the message (51.8 dB),
so a detector trained on it learns to spot compression.

Here every cover is passed through exactly the same requantisation with no message, and the stego is
the original one (same seed and rate as F:\Steganalyzer_UniversalDCT\02_make_stego.py, regenerated and
checked byte for byte against the existing stego file). The only difference left is the hidden message.

Output: universal_fixed\covers, universal_fixed\stego (same file names), universal_fixed\pairs.csv
Original folder is only read, never modified. Resumable: existing output pairs are skipped.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, csv, importlib.util, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)

SRC = Path(r"F:\Steganalyzer_UniversalDCT"); OUT = Path(__file__).resolve().parent / "universal_fixed"
_G = None

def _init():
    global _G
    spec = importlib.util.spec_from_file_location("gen_dataset", r"F:\DEDS_project\gen_dataset.py")
    _G = importlib.util.module_from_spec(spec); spec.loader.exec_module(_G)

def requantise(cover_bgr):
    """embed_dct_random with no message: identical colour conversion, block DCT, quantise, dequantise, IDCT."""
    img = cover_bgr.astype(np.float32); h, w = img.shape[:2]
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb); Y, Cb, Cr = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    Q = _G.JPEG_STD_LUM_QUANT_TABLE; Yr = np.zeros((h, w), np.float32)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            zz = _G._zigzag(np.around(cv2.dct(Y[y:y + 8, x:x + 8]) / Q))      # same rounding path as the embedder
            Yr[y:y + 8, x:x + 8] = cv2.idct(_G._inv_zigzag(zz) * Q)
    out = cv2.cvtColor(np.stack([Yr, Cb, Cr], axis=2), cv2.COLOR_YCrCb2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)

def work(task):
    idx, name = task
    cover = cv2.imread(str(SRC / "covers" / name))
    np.random.seed(500000 + idx); rate = float(np.random.uniform(0.05, 0.15))   # as 02_make_stego.py
    stego = _G.embed_dct_random(cover, rate)
    same_as_original = bool(np.array_equal(stego, cv2.imread(str(SRC / "stego" / name))))
    cq = requantise(cover)
    if np.array_equal(stego, cover):        # nothing embedded: the embedder returned the raw cover
        return {"image": name, "rate": rate, "ok": False, "reason": "no embedding"}
    diff = stego.astype(int) - cq.astype(int)
    for sub, im in (("covers", cq), ("stego", stego)):
        tmp = OUT / sub / ("tmp_" + name); cv2.imwrite(str(tmp), im); os.replace(tmp, OUT / sub / name)
    mse_msg = (diff ** 2).mean(); mse_q = ((cq.astype(float) - cover) ** 2).mean()
    return {"image": name, "rate": rate, "ok": True, "reason": "", "stego_matches_original": same_as_original,
            "pixels_changed_frac": float((diff != 0).any(2).mean()),
            "psnr_message_only": 10 * np.log10(255 ** 2 / mse_msg) if mse_msg else float("inf"),
            "psnr_requantisation": 10 * np.log10(255 ** 2 / mse_q) if mse_q else float("inf")}

def main():
    for sub in ("covers", "stego"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (SRC / "covers").glob("*.png"))       # same order as 02_make_stego.py
    tasks = [(i, n) for i, n in enumerate(names) if not ((OUT / "covers" / n).exists() and (OUT / "stego" / n).exists())]
    print(f"{len(names) - len(tasks)} done, {len(tasks)} to do", flush=True)
    rows = []
    with cf.ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        for k, r in enumerate(ex.map(work, tasks, chunksize=20)):
            rows.append(r)
            if (k + 1) % 500 == 0:
                print(k + 1, "of", len(tasks), flush=True)
    if rows:
        old = list(csv.DictReader(open(OUT / "pairs.csv", newline=""))) if (OUT / "pairs.csv").exists() else []
        keys = list(rows[0].keys()) if rows[0]["ok"] else ["image", "rate", "ok", "reason"]
        allr = {r["image"]: r for r in old}; allr.update({r["image"]: r for r in rows})
        keys = sorted({k for r in allr.values() for k in r}, key=lambda k: (k != "image", k))
        with open(OUT / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(sorted(allr.values(), key=lambda r: r["image"]))
    ok = [r for r in rows if r["ok"]]
    print(f"built {len(ok)} pairs, skipped {len(rows) - len(ok)}", flush=True)
    if ok:
        print("stego identical to the original stego file: %d of %d" % (sum(r["stego_matches_original"] for r in ok), len(ok)))
        print("PSNR of the requantisation the covers now share: %.1f dB" % np.mean([r["psnr_requantisation"] for r in ok]))
        print("PSNR cover vs stego now (message only):          %.1f dB" % np.mean([r["psnr_message_only"] for r in ok]))
        print("pixels changed by the message: %.1f%%" % (100 * np.mean([r["pixels_changed_frac"] for r in ok])))

if __name__ == "__main__":
    main()
