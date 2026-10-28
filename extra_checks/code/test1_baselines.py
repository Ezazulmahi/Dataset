"""Test 1: can real photos, or simply sharpened or noised baseline covers, match the fine-tuned covers?

Phase "stats": eligible count, cover-derived K and capacity, high-frequency power share and spectral slope for
  finetuned (1,000), baseline (1,000), photos (195 reference photographs),
  baseline + unsharp mask of amount 0.5, 1, 2, 4 (Gaussian sigma 1), baseline + Gaussian noise sigma 2, 4, 8.
  Also PSNR of each processed cover against the original baseline cover (how much the processing changes it).
Phase "embed": matched-payload embedding (same K, same message as the paper) on 200 pairs (every fifth) for
  finetuned, baseline, and the sharpened and noised variants whose mean eligible count is closest to finetuned;
  the 195 photos get the K and payload of pairs 0 to 194 and are compared with the finetuned covers of those pairs.
Resumable: results/test1_stats.csv and results/test1_embed.csv get a row per image as soon as it is done."""
import os, sys, random, string, tempfile
import numpy as np, pandas as pd, cv2
import common as C

STATS = os.path.join(C.HERE, "results", "test1_stats.csv")
EMBED = os.path.join(C.HERE, "results", "test1_embed.csv")
ALPHABET = string.ascii_letters + string.digits + " .,!?"
SHARP = (0.5, 1.0, 2.0, 4.0)
NOISE = (2, 4, 8)
D = None


def init():
    global D
    D = C.load_v5()


def photo_names():
    return sorted(os.listdir(C.PHOTOS))


def make_cover(variant, i):
    """variant: finetuned, baseline, photo, sharp_<a>, noise_<s>; i: pair index (photo index for photo)."""
    if variant == "photo":
        return cv2.imread(os.path.join(C.PHOTOS, photo_names()[i]))
    if variant == "finetuned":
        return cv2.imread(os.path.join(C.CORP, "finetuned", f"{i:03d}.png"))
    base = cv2.imread(os.path.join(C.CORP, "baseline", f"{i:03d}.png"))
    if variant == "baseline":
        return base
    kind, val = variant.split("_"); val = float(val)
    b = base.astype(np.float64)
    if kind == "sharp":
        out = b + val * (b - cv2.GaussianBlur(b, (0, 0), 1.0))
    else:
        out = b + np.random.default_rng(700000 + i).normal(0, val, b.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def stats_work(task):
    key, variant, i = task
    img = make_cover(variant, i)
    row = dict(key=key, variant=variant, index=i, **C.cover_stats(D, img), **C.spectrum_stats(img), psnr_vs_baseline=np.nan)
    if variant.startswith(("sharp", "noise")):
        row["psnr_vs_baseline"] = C.quality(make_cover("baseline", i), img)["psnr_y"]
    return [row]


def embed_work(task):
    key, variant, i, K, n, pair = task
    msg = "".join(random.Random(424242 + pair).choice(ALPHABET) for _ in range(n // 8))
    bits = np.array([(b >> (7 - k)) & 1 for b in msg.encode() for k in range(8)])
    cover = make_cover(variant, i)
    row = dict(key=key, variant=variant, index=i, pair=pair, K=K, n_bits=len(bits), bpp=len(bits) / 65536,
               error="", fix_passes=np.nan, exact=False, psnr_y=np.nan, ssim_y=np.nan)
    try:
        stego, passes = C.embed_bits(D, cover, bits, K)
    except Exception as e:
        row.update(error=repr(e)); return [row]
    tmp = os.path.join(tempfile.gettempdir(), f"deds_t1_{os.getpid()}.png")
    stego = C.png_roundtrip(stego, tmp)
    got = C.read_bits(D, stego)
    row.update(fix_passes=passes, exact=got is not None and len(got) == len(bits) and bool(np.all(got == bits)),
               **C.quality(cover, stego))
    return [row]


def pick_matched_variants():
    s = pd.read_csv(STATS)
    target = s[s.variant == "finetuned"].e_bar.mean()
    m = s.groupby("variant").e_bar.mean()
    sharp = (m[[f"sharp_{a}" for a in SHARP]] - target).abs().idxmin()
    noise = (m[[f"noise_{x}" for x in NOISE]] - target).abs().idxmin()
    print(f"finetuned mean e_bar {target:.2f}; closest sharpening {sharp} ({m[sharp]:.2f}), closest noise {noise} ({m[noise]:.2f})")
    return sharp, noise


if __name__ == "__main__":
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("stats", "all"):
        variants = ["finetuned", "baseline"] + [f"sharp_{a}" for a in SHARP] + [f"noise_{x}" for x in NOISE]
        tasks = [(f"{v}:{i}", v, i) for v in variants for i in range(1000)]
        tasks += [(f"photo:{i}", "photo", i) for i in range(len(photo_names()))]
        if not C.run_resumable(tasks, stats_work, STATS, init=init):
            sys.exit(1)
    if phase in ("embed", "all"):
        mt = C.matched_tasks()
        sharp, noise = pick_matched_variants()
        tasks = []
        for v in ("finetuned", "baseline", sharp, noise):
            tasks += [(f"{v}:{i}", v, i, *mt[i], i) for i in range(0, 1000, 5)]
        tasks += [(f"photo:{j}", "photo", j, *mt[j], j) for j in range(len(photo_names()))]
        tasks += [(f"finetuned_photo_pairs:{j}", "finetuned", j, *mt[j], j) for j in range(len(photo_names())) if j % 5]
        C.run_resumable(tasks, embed_work, EMBED, init=init)
