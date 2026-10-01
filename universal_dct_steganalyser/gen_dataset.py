"""
gen_dataset.py  —  Download free cover images and generate stego pairs locally.

Uses picsum.photos (free, no API key) to download natural color photos,
resizes to 256x256, then generates stego images using the same embed_dct_random
function from the training pipeline (JPEG Q50 table, Y-channel, random rate).

Usage:
    python gen_dataset.py --n 500 --out_dir F:\\Thesis\\LocalDataset
"""

import os, math, random, argparse, urllib.request, time
import numpy as np
import cv2


# ── Same embed function as training pipeline ─────────────────────────────────
JPEG_STD_LUM_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float32)

def _zigzag(block):
    h, w = block.shape
    result = np.empty(h * w, dtype=block.dtype)
    idx = 0
    for s in range(h + w - 1):
        if s % 2 == 0:
            x = 0 if s < w else s - w + 1; y = s if s < w else w - 1
            while x < h and y >= 0:
                result[idx] = block[x, y]; idx += 1; x += 1; y -= 1
        else:
            x = s if s < h else h - 1; y = 0 if s < h else s - h + 1
            while x >= 0 and y < w:
                result[idx] = block[x, y]; idx += 1; x -= 1; y += 1
    return result

def _inv_zigzag(vector, h=8, w=8):
    block = np.empty((h, w), dtype=vector.dtype)
    idx = 0
    for s in range(h + w - 1):
        if s % 2 == 0:
            x = 0 if s < w else s - w + 1; y = s if s < w else w - 1
            while x < h and y >= 0:
                block[x, y] = vector[idx]; idx += 1; x += 1; y -= 1
        else:
            x = s if s < h else h - 1; y = 0 if s < h else s - h + 1
            while x >= 0 and y < w:
                block[x, y] = vector[idx]; idx += 1; x -= 1; y += 1
    return block

def embed_dct_random(cover_bgr: np.ndarray, rate: float) -> np.ndarray:
    img = cover_bgr.astype(np.float32)
    h, w = img.shape[:2]
    new_h = math.ceil(h / 8) * 8; new_w = math.ceil(w / 8) * 8
    if (new_h, new_w) != (h, w):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]

    Y_blocks = [Y[y:y+8, x:x+8] for y in range(0,h,8) for x in range(0,w,8)]
    zigzag_blocks = [_zigzag(np.around(cv2.dct(b) / JPEG_STD_LUM_QUANT_TABLE))
                     for b in Y_blocks]

    embeddable = [(bi, ki) for bi, zz in enumerate(zigzag_blocks)
                  for ki in range(1, 64) if int(round(zz[ki])) > 1]

    n_embed = int(len(embeddable) * rate)
    if n_embed == 0:
        return cover_bgr
    bits = np.random.randint(0, 2, n_embed)
    for idx, (bi, ki) in enumerate(embeddable[:n_embed]):
        c = int(round(zigzag_blocks[bi][ki]))
        zigzag_blocks[bi][ki] = float((c & ~1) | bits[idx])

    Y_idct = [cv2.idct(_inv_zigzag(zz) * JPEG_STD_LUM_QUANT_TABLE)
              for zz in zigzag_blocks]
    Y_rec = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for y in range(0,h,8):
        for x in range(0,w,8):
            Y_rec[y:y+8, x:x+8] = Y_idct[idx]; idx += 1

    stego = cv2.cvtColor(np.stack([Y_rec, Cb, Cr], axis=2), cv2.COLOR_YCrCb2BGR)
    return np.clip(stego, 0, 255).astype(np.uint8)


# ── Download from picsum.photos (free, no API key, real photos) ──────────────
def download_image(img_id: int, size: int = 256) -> np.ndarray | None:
    url = f"https://picsum.photos/id/{img_id}/{size}/{size}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=200,
                        help="Number of cover/stego pairs to generate")
    parser.add_argument("--out_dir", type=str, default=r"F:\Thesis\LocalDataset",
                        help="Output root directory")
    parser.add_argument("--size",    type=int, default=256,
                        help="Image size (square)")
    parser.add_argument("--rate",    type=float, default=0.10,
                        help="Embedding rate (0.05–0.15)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    cover_dir = os.path.join(args.out_dir, "covers")
    stego_dir = os.path.join(args.out_dir, "stego")
    os.makedirs(cover_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    # picsum has ~1000 valid IDs; shuffle to get varied images
    ids = list(range(1, 1084))
    random.shuffle(ids)

    done = 0
    skipped = 0
    print(f"Downloading {args.n} images at {args.size}×{args.size}, "
          f"embed rate={args.rate:.2f} ...")

    for img_id in ids:
        if done >= args.n:
            break

        fname = f"img_{img_id:04d}.png"
        cpath = os.path.join(cover_dir, fname)
        spath = os.path.join(stego_dir, fname)

        if os.path.exists(cpath) and os.path.exists(spath):
            done += 1
            continue

        img = download_image(img_id, args.size)
        if img is None:
            skipped += 1
            continue

        stego = embed_dct_random(img, args.rate)

        cv2.imwrite(cpath, img)
        cv2.imwrite(spath, stego)

        done += 1
        if done % 20 == 0:
            print(f"  {done}/{args.n} done  (skipped {skipped})")
        time.sleep(0.05)   # be polite to the server

    print(f"\nDone. {done} pairs in {args.out_dir}")
    print(f"  covers -> {cover_dir}")
    print(f"  stego  -> {stego_dir}")
    print(f"\nTo train, set in stega.py:")
    print(f"  COVER_DIR  = r'{cover_dir}'")
    print(f"  DCT_DIR    = r'{stego_dir}'")
    print(f"  USE_PREMADE = True")


if __name__ == "__main__":
    main()
