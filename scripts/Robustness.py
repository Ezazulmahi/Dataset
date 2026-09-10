"""
robustness_pipeline.py
=======================
Channel-robustness evaluation for DEDS stego images (fine-tuned covers only).

For every fine-tuned cover image, this script:

  1. Embeds a random ASCII message using DCT_Adaptive.embed() at its natural
     adaptive K (no override -- this is the "real" operating condition, not
     the matched-K comparison used in thesis_comparison_pipeline.py).
  2. Records the ground-truth K, slot order and embedded bit list *before*
     any distortion is applied. This matters because the header itself can
     be corrupted by a distortion, and if we only relied on
     DCT_Adaptive.extract() we would get a binary pass/fail with no bit-level
     signal once the header breaks. Keeping the ground-truth K lets us read
     the payload bits at their known positions directly from the distorted
     image and compute a real bit-error-rate (BER) even when the header
     itself does not survive.
  3. Applies each distortion in DISTORTIONS to a copy of the stego image:
       - jpeg_standard   : real libjpeg re-encode via cv2 (the "main"/OS JPEG)
       - jpeg_custom_qtable : a block-DCT quantize/dequantize pass built from
         DCT_Adaptive's OWN luminance _Q_TABLE (IJG-scaled by quality), applied
         to the Y channel only -- i.e. "JPEG-like" compression in exactly the
         domain the embedder itself reasons about, chroma left untouched.
       - resize          : downscale then upscale back to the original size
       - gaussian_noise   : additive Gaussian noise at increasing sigma
       - gaussian_blur_mild : a single mild blur pass
       - crop_mild        : crop a border fraction, then resize back up
       - none             : sanity-check control (should read back ~0% BER)
  4. For each distorted image: reads the header (crc/magic ok or not), reads
     the payload bits at the ground-truth slot positions and computes BER
     against the originally embedded bits, and separately calls
     DCT_Adaptive.extract() to record whether the real-world pipeline
     (header + Huffman-free UTF-8 decode used by this module) still recovers
     the exact message.
  5. Aggregates mean/std/95% CI of BER and bit-accuracy per distortion type
     and severity level, plus the extraction-success ratio, and writes a
     literature-baseline comparison sheet (hard-coded from the papers already
     cited in your report -- these are NOT re-run on your pipeline, since we
     don't have their code/weights; they're included only as a side-by-side
     reference table).

Output: a single Excel workbook with per-image results, a summary/curve
sheet, and a literature-comparison sheet.

Run this on the machine where DCT_Adaptive.py and your finetuned image
folder live (not a sandbox) -- needs cv2, scipy, scikit-image, pandas,
openpyxl, tqdm.

    pip install opencv-python scipy scikit-image pandas openpyxl tqdm
"""

import os

# MUST be set before numpy/scipy/cv2 are imported anywhere in this file --
# BLAS/OpenCV read these once at library-load time. On Windows,
# multiprocessing uses "spawn", so every worker re-imports this whole file
# before its initializer runs, which is why these are set here at module
# level rather than inside _init_worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import io
import glob
import random
import string
import contextlib
import importlib.util
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)
from scipy import stats
from scipy.fftpack import dct, idct

try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------------
# CONFIG -- edit before running
# ------------------------------------------------------------------------
DCT_MODULE_PATH = r"F:\Results\DCT_Adaptive.py"
FINETUNED_DIR   = r"F:\Results\finetuned"
EXCEL_OUT_PATH  = r"F:\Results\robustness_results.xlsx"
CHECKPOINT_PATH = r"F:\Results\checkpoints\robustness_checkpoint.csv"

# Fraction of each cover's own capacity to embed. Kept well under 1.0 so the
# same message survives the resize/crop dimension changes below without
# hitting a "message too large" error on any image in the set.
PAYLOAD_FRACTION = 0.5

RANDOM_SEED = 42
CONFIDENCE  = 0.95
QUIET_EMBED_LOGS = True

N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# ------------------------------------------------------------------------
# Distortion severity levels. Each tuple is (dist_type, param, label).
# "label" is what shows up in the Excel sheets and controls x-axis ordering
# for the BER curves (severity increases top to bottom within each type).
# ------------------------------------------------------------------------
DISTORTIONS = [
    ("none",                None,  "none"),

    ("jpeg_standard",        90,    "jpeg_standard_QF90"),
    ("jpeg_standard",        70,    "jpeg_standard_QF70"),
    ("jpeg_standard",        50,    "jpeg_standard_QF50"),

    ("jpeg_custom_qtable",   90,    "jpeg_customQ_QF90"),
    ("jpeg_custom_qtable",   70,    "jpeg_customQ_QF70"),
    ("jpeg_custom_qtable",   50,    "jpeg_customQ_QF50"),

    ("resize",               0.90,  "resize_0.90x"),
    ("resize",               0.75,  "resize_0.75x"),
    ("resize",               0.50,  "resize_0.50x"),

    ("gaussian_noise",       5.0,   "gaussian_noise_sigma5"),
    ("gaussian_noise",       10.0,  "gaussian_noise_sigma10"),
    ("gaussian_noise",       15.0,  "gaussian_noise_sigma15"),

    ("gaussian_blur_mild",   None,  "gaussian_blur_mild"),

    ("crop_mild",            0.05,  "crop_mild_5pct"),
]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------
# Module loading
# ------------------------------------------------------------------------
def load_dct_module(path):
    spec = importlib.util.spec_from_file_location("DCT_Adaptive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_ascii_message(n_bytes):
    """ASCII-only so len(message.encode('utf-8')) == n_bytes exactly."""
    alphabet = string.ascii_letters + string.digits + " .,!?"
    return "".join(random.choice(alphabet) for _ in range(n_bytes))


# ------------------------------------------------------------------------
# Custom-quant-table "JPEG-like" pass, built from DCT_Adaptive's own
# luminance _Q_TABLE. Uses the standard IJG quality->scale formula so
# QF90/70/50 here are on the same quality scale as jpeg_standard, but the
# quantisation step sizes are exactly the embedder's own table (scaled),
# not libjpeg's. Applied to the Y channel only -- Cr/Cb pass through
# unchanged, matching the fact that embedding itself never touches them.
# ------------------------------------------------------------------------
def _ijg_scale(quality):
    quality = max(1, min(100, int(quality)))
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - quality * 2.0


def _blockify(channel):
    h, w = channel.shape
    h8, w8 = h - h % 8, w - w % 8
    nr, nc = h8 // 8, w8 // 8
    ch = channel[:h8, :w8]
    blocks = ch.reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3).astype(np.float64)
    return blocks, h8, w8


def _deblockify(blocks, h8, w8):
    nr, nc = blocks.shape[:2]
    ch = blocks.transpose(0, 2, 1, 3).reshape(h8, w8)
    return ch


def _dct2_batch(blocks):
    return dct(dct(blocks, axis=-1, norm='ortho'), axis=-2, norm='ortho')


def _idct2_batch(blocks):
    return idct(idct(blocks, axis=-1, norm='ortho'), axis=-2, norm='ortho')


def jpeg_custom_qtable(module, img_bgr, quality):
    scale = _ijg_scale(quality)
    Qs = np.floor((module._Q_TABLE * scale + 50.0) / 100.0)
    Qs = np.clip(Qs, 1, 255)

    Y, Cr, Cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = _blockify(Y)
    coeffs = _dct2_batch(blocks - 128.0)
    q = np.round(coeffs / Qs) * Qs
    recon = np.clip(np.round(_idct2_batch(q) + 128.0), 0, 255)
    Y_out = Y.copy()
    Y_out[:h8, :w8] = recon.astype(np.uint8)

    out = cv2.merge([Y_out, Cr, Cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


# ------------------------------------------------------------------------
# Distortion dispatch -- every function returns a BGR uint8 array with the
# SAME (h, w) as the input, since payload slot positions are indexed by
# block (i, j) coordinates that assume unchanged image dimensions.
# ------------------------------------------------------------------------
def apply_distortion(module, stego_bgr, dist_type, param):
    h, w = stego_bgr.shape[:2]

    if dist_type == "none":
        return stego_bgr.copy()

    if dist_type == "jpeg_standard":
        ok, enc = cv2.imencode(".jpg", stego_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(param)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if dist_type == "jpeg_custom_qtable":
        return jpeg_custom_qtable(module, stego_bgr, param)

    if dist_type == "resize":
        small_h = max(8, int(round(h * param)))
        small_w = max(8, int(round(w * param)))
        down = cv2.resize(stego_bgr, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

    if dist_type == "gaussian_noise":
        noise = np.random.normal(0.0, float(param), stego_bgr.shape)
        noisy = stego_bgr.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if dist_type == "gaussian_blur_mild":
        return cv2.GaussianBlur(stego_bgr, (3, 3), sigmaX=0.5)

    if dist_type == "crop_mild":
        ch = int(round(h * param))
        cw = int(round(w * param))
        cropped = stego_bgr[ch:h - ch, cw:w - cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    raise ValueError(f"Unknown distortion type: {dist_type}")


# ------------------------------------------------------------------------
# Ground-truth-position bit reading (bypasses the header so BER is
# meaningful even when the header itself is corrupted by the distortion).
# ------------------------------------------------------------------------
def read_payload_bits_at(module, image_bgr, slot_order, n_bits):
    Y = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = module._to_blocks(Y)
    bits = []
    for (i, j, u, v) in slot_order[:n_bits]:
        bits.append(module._lsb_read(blocks[i, j, u, v], u, v))
    return bits


def header_survives(module, image_bgr):
    Y = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = module._to_blocks(Y)
    n_bits, K, version, crc_ok = module._read_header(blocks)
    return bool(crc_ok), n_bits, K


def bit_error_rate(original_bits, read_bits):
    n = len(original_bits)
    if n == 0:
        return 0.0
    errors = sum(1 for a, b in zip(original_bits, read_bits) if a != b)
    return errors / n


# ------------------------------------------------------------------------
# Per-image worker: embed once, then run every distortion against that
# one stego image (avoids re-embedding per distortion).
# ------------------------------------------------------------------------
_worker_module = None


def _init_worker(dct_module_path):
    global _worker_module
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    _worker_module = load_dct_module(dct_module_path)


def _process_one_image(task):
    global _worker_module
    module = _worker_module

    name = task["name"]
    path = task["path"]

    cover = cv2.imread(path)
    if cover is None:
        return [{"image": name, "distortion": "ALL", "error": "could not read image"}]

    try:
        cap = module.get_capacity(cover)
    except Exception as e:
        return [{"image": name, "distortion": "ALL", "error": f"capacity error: {e}"}]

    n_bytes = max(8, int(cap["payload_bits"] * PAYLOAD_FRACTION) // 8)
    seed = (hash((name, "robustness_message")) ^ RANDOM_SEED) & 0xFFFFFFFF
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " .,!?"
    message = "".join(rng.choice(alphabet) for _ in range(n_bytes))

    raw = message.encode("utf-8")
    original_bits = [(b >> (7 - k)) & 1 for b in raw for k in range(8)]
    n_bits = len(original_bits)

    # Ground-truth K: the same value module.embed() will compute internally,
    # queried directly so we can rebuild the identical slot order afterwards.
    Y = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = module._to_blocks(Y)
    K = module._compute_K(blocks)
    nr, nc = blocks.shape[:2]
    slot_order = module._build_slot_order(K, nr, nc)

    try:
        if QUIET_EMBED_LOGS:
            with contextlib.redirect_stdout(io.StringIO()):
                stego = module.embed(cover, message)
        else:
            stego = module.embed(cover, message)
    except Exception as e:
        return [{"image": name, "distortion": "ALL", "error": f"embed error: {e}"}]

    rows = []
    for dist_type, param, label in DISTORTIONS:
        row = {
            "image": name, "distortion": label, "distortion_type": dist_type,
            "distortion_param": param, "K": K, "n_bits": n_bits,
            "target_bpp": cap["bpp"], "error": None,
        }
        try:
            distorted = apply_distortion(module, stego, dist_type, param)
        except Exception as e:
            row["error"] = f"distortion error: {e}"
            rows.append(row)
            continue

        try:
            header_ok, header_n_bits, header_K = header_survives(module, distorted)
        except Exception as e:
            header_ok, header_n_bits, header_K = False, None, None

        try:
            read_bits = read_payload_bits_at(module, distorted, slot_order, n_bits)
            ber = bit_error_rate(original_bits, read_bits)
        except Exception as e:
            ber = 1.0
            row["error"] = f"bit-read error: {e}"

        try:
            recovered = module.extract(distorted)
            extract_success = (recovered == message)
        except Exception:
            extract_success = False

        row.update({
            "header_survives": header_ok,
            "ber": ber,
            "bit_accuracy": 1.0 - ber,
            "extract_success": extract_success,
        })
        rows.append(row)

    return rows


# ------------------------------------------------------------------------
# Main sweep with checkpoint/resume (matches the resumability pattern used
# in thesis_comparison_pipeline.py -- a crash never loses more than the
# current batch of images).
# ------------------------------------------------------------------------
def run_sweep(dct_module_path, finetuned_dir, checkpoint_path, n_workers=N_WORKERS):
    paths = sorted(glob.glob(os.path.join(finetuned_dir, "*")))
    if not paths:
        raise RuntimeError(f"No images found in {finetuned_dir}")
    print(f"Found {len(paths)} fine-tuned cover images.")
    print(f"Using {n_workers} worker process(es).\n")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    done_rows = []
    done_names = set()
    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)
        done_rows = done_df.to_dict("records")
        done_names = set(done_df["image"].astype(str))
        print(f"Resuming from checkpoint: {len(done_names)} image(s) already done "
              f"-> {checkpoint_path}\n")

    tasks = [
        {"name": os.path.basename(p), "path": p}
        for p in paths
        if os.path.basename(p) not in done_names
    ]

    new_rows = []
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(dct_module_path,)
        ) as executor:
            futures = [executor.submit(_process_one_image, t) for t in tasks]
            for i, fut in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                          total=len(futures), desc="images")):
                result = fut.result()
                if result:
                    new_rows.extend(result)

                if (i + 1) % 25 == 0:
                    pd.DataFrame(done_rows + new_rows).to_csv(checkpoint_path, index=False)

    all_rows = done_rows + new_rows
    pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
    print(f"\nCheckpoint saved -> {checkpoint_path}")

    return pd.DataFrame(all_rows)


# ------------------------------------------------------------------------
# Stats: mean, std, 95% CI per distortion label (the BER/success "curve")
# ------------------------------------------------------------------------
def mean_std_ci(values, confidence=CONFIDENCE):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / np.sqrt(n)
        h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = 0.0
    return mean, std, mean - h, mean + h


# Manual severity ordering so the summary sheet reads as a curve, worst
# distortion last within each family.
_SEVERITY_ORDER = {label: i for i, (_, _, label) in enumerate(DISTORTIONS)}


def summarize(df):
    valid = df[df["error"].isna()] if "error" in df.columns else df
    rows = []
    for label, sub in valid.groupby("distortion"):
        if len(sub) == 0:
            continue
        ber_mean, ber_std, ber_lo, ber_hi = mean_std_ci(sub["ber"])
        acc_mean, acc_std, acc_lo, acc_hi = mean_std_ci(sub["bit_accuracy"])
        header_rate = float(sub["header_survives"].mean())
        success_rate = float(sub["extract_success"].mean())
        rows.append({
            "distortion": label,
            "severity_rank": _SEVERITY_ORDER.get(label, 999),
            "n_images": len(sub),
            "ber_mean": ber_mean, "ber_std": ber_std,
            "ber_ci_low": ber_lo, "ber_ci_high": ber_hi,
            "bit_accuracy_mean": acc_mean, "bit_accuracy_std": acc_std,
            "bit_accuracy_ci_low": acc_lo, "bit_accuracy_ci_high": acc_hi,
            "header_survival_rate": header_rate,
            "extraction_success_rate": success_rate,
        })
    out = pd.DataFrame(rows).sort_values("severity_rank").reset_index(drop=True)
    return out


# ------------------------------------------------------------------------
# Literature comparison sheet -- hard-coded reference values from the
# papers already cited in the report's bibliography. These are NOT re-run
# on your pipeline (no access to their code/weights); this sheet exists
# purely as a side-by-side reference table alongside your own measured
# numbers above. Fill in / trim rows as appropriate for your final table.
# ------------------------------------------------------------------------
def literature_comparison_table():
    return pd.DataFrame([
        {
            "method": "DEDS (this work, measured)",
            "channel_condition": "see summary_stats sheet",
            "reported_metric": "BER / bit-accuracy / extraction-success rate",
            "source": "this run",
        },
        {
            "method": "Long et al. 2026 [19] -- Robust SD steganography",
            "channel_condition": "real social-network transmission (e.g. Xiaohongshu)",
            "reported_metric": "100% zero-error at 0.0017 bpp; 93.4% at 0.0625 bpp",
            "source": "IEEE IoT Journal, early access 2026",
        },
        {
            "method": "Hu et al. 2024 [18] -- Robust generative steganography (SD)",
            "channel_condition": "JPEG90 + resize combined attack",
            "reported_metric": ">95% accuracy; JPEG50 76-82%; heavy blur 60-70%",
            "source": "IEEE TIFS 2024",
        },
        {
            "method": "Zhou et al. 2025 -- GSD (diffusion steganography)",
            "channel_condition": "JPEG compression",
            "reported_metric": "~60% recovery under JPEG",
            "source": "cited in this report, Ch.2",
        },
        {
            "method": "Gu et al. 2026 [18-bib] -- GAN JPEG steganography",
            "channel_condition": "JPEG-domain, adaptive modification loss",
            "reported_metric": "~0.50 bits/non-zero-AC-coeff (JPEG unit, not bpp); "
                                "~61% detection (39% detection error)",
            "source": "IEEE TCSVT, early access 2026",
        },
    ])


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    print("=== DEDS channel-robustness evaluation (fine-tuned covers) ===\n")
    print(f"Distortions: {[label for _, _, label in DISTORTIONS]}\n")

    df = run_sweep(DCT_MODULE_PATH, FINETUNED_DIR, CHECKPOINT_PATH, n_workers=N_WORKERS)

    if "error" in df.columns:
        n_errors = df["error"].notna().sum()
        if n_errors:
            print(f"\n[!] {n_errors} row(s) had errors -- see 'error' column in per_image_results.")

    summary_df = summarize(df)
    lit_df = literature_comparison_table()

    os.makedirs(os.path.dirname(EXCEL_OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(EXCEL_OUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_image_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_curve", index=False)
        lit_df.to_excel(writer, sheet_name="literature_comparison", index=False)

    print(f"\nDone. Results saved to: {EXCEL_OUT_PATH}")
    print(f"Checkpoint is at: {CHECKPOINT_PATH} "
          f"(safe to delete once robustness_results.xlsx looks correct)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()