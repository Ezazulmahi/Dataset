"""
robustness_customQ_adv_ecc_align.py
=====================================
Pilot robustness suite for ~400 stego images in F:\\Results\\stego_finetuned,
produced by DCT_Adaptive.py.  Ground truth for each image is obtained by
calling extract() on the UNMODIFIED stego (the scheme is self-contained --
header carries payload length + CRC -- so no external message log is needed).

Tests covered
-------------
1a) Custom-Q attack       : attacker recompresses the stego with a NON-STANDARD
                              (not a normal QFxx) 8x8 quantization table, then we
                              extract() and measure damage. Simulates an unusual
                              recompression pipeline, distinct from your existing
                              QF70/80/90 sweep.
1b) Custom-Q clean control: extract() on the UNMODIFIED stego, to catch any
                              residual bug in your own embedding's custom
                              Q-table logic (sanity/regression check, should be
                              ~100% across all 400 images).
3a) Adversarial attack     : WHITE-BOX worst-case perturbation. Using the
                              embedding module's own internals (header/payload
                              slot positions), we compute the exact DCT-domain
                              change that flips every embedded bit, then clip
                              that change to an L-inf pixel budget (epsilon).
                              Run at several epsilons to trace a degradation
                              curve (larger budget = more damage).
3b) ECC benefit simulation : Monte-Carlo simulation -- NOT a re-embed test.
                              Given the empirically observed raw bit-error-rate
                              (BER) for a given attack, we ask: "if this payload
                              had been protected with Hamming(7,4) and suffered
                              i.i.d. bit errors at that same BER, would it have
                              been recovered?" This avoids re-embedding all 400
                              images (which would also need capacity re-checks
                              for the ~1.75x larger ECC payload) while still
                              giving a defensible before/after-ECC comparison.
                              If you want a real embedded-ECC test later
                              (actually re-embedding ECC-coded payloads and
                              measuring true recovery), that's a separate,
                              bigger pipeline -- flag it and we'll build it.
3c) Alignment attack        : small pixel shifts, a small rotation, and a
                              crop+resize, applied to the stego before
                              extract(). This scheme has no re-synchronization,
                              so these are expected to fail catastrophically --
                              the point is to QUANTIFY that with real numbers.

Output
------
A NEW workbook (separate from your other results files), with:
    per_image_results   -- long format, one row per (image, test condition)
    summary_stats        -- mean BER / exact-match rate / 95% CI per condition

    pip install opencv-python scipy numpy pandas openpyxl tqdm
"""

import os
import sys
import glob
import random

import numpy as np
import pandas as pd
import cv2
from scipy import stats
from scipy.fftpack import dct, idct

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
STEGO_DIR       = r"F:\Results\stego_finetuned"
DSTG_MODULE_DIR = r"F:\Results"          # folder containing DCT_Adaptive.py
EXCEL_PATH      = r"F:\Results\robustness_customQ_adv_ecc_align.xlsx"
MAX_IMAGES      = 400
CONFIDENCE      = 0.95
RANDOM_SEED     = 42

ADV_EPSILONS       = [1, 2, 4, 8]          # L-inf pixel budgets for the adversarial attack
ECC_SOURCE_TESTS   = ["custom_q_attack", "adversarial_eps8"]  # which conditions get an ECC-benefit sim

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sys.path.insert(0, DSTG_MODULE_DIR)
import DCT_Adaptive as dstg  # noqa: E402  (embed/extract module, must be on DSTG_MODULE_DIR)


# ------------------------------------------------------------------------
# Non-standard custom quantization table (deliberately NOT a standard JPEG
# QFxx table -- irregular / skewed band weighting) used for the 1a attack.
# ------------------------------------------------------------------------
CUSTOM_Q_TABLE = np.array([
    [ 5,  3,  9,  6, 18, 11, 24, 15],
    [ 4,  8,  7, 14, 10, 22, 13, 26],
    [11,  6, 16,  9, 21, 12, 27, 14],
    [ 7, 15,  8, 19, 11, 25, 13, 30],
    [17, 10, 20, 12, 28, 14, 33, 16],
    [ 9, 23, 12, 26, 14, 31, 15, 38],
    [22, 13, 29, 15, 34, 16, 40, 18],
    [12, 27, 14, 32, 16, 37, 18, 44],
], dtype=np.float32)


# ------------------------------------------------------------------------
# Bit / string helpers
# ------------------------------------------------------------------------
def str_to_bits(s: str) -> list:
    raw = s.encode("utf-8")
    return [(b >> (7 - k)) & 1 for b in raw for k in range(8)]


def bits_to_str(bits: list):
    byte_arr = bytearray(
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, len(bits) - 7, 8)
    )
    try:
        return byte_arr.decode("utf-8")
    except UnicodeDecodeError:
        return None


def bit_diff_stats(gt_str: str, test_str) -> dict:
    """Compare extracted string against ground truth. Handles extract()
    failures and length mismatches gracefully."""
    if test_str is None:
        return {"exact_match": False, "ber": 1.0, "length_match": False, "note": "extract_failed"}
    gt_bits = str_to_bits(gt_str)
    t_bits = str_to_bits(test_str)
    n = max(len(gt_bits), len(t_bits), 1)
    gt_p = gt_bits + [0] * (n - len(gt_bits))
    t_p = t_bits + [0] * (n - len(t_bits))
    errors = sum(1 for a, b in zip(gt_p, t_p) if a != b)
    return {
        "exact_match": gt_str == test_str,
        "ber": errors / n,
        "length_match": len(gt_bits) == len(t_bits),
        "note": None,
    }


def safe_extract(img_bgr):
    try:
        return dstg.extract(img_bgr)
    except Exception:
        return None


# ------------------------------------------------------------------------
# 1a) Custom-Q attack (non-standard recompression table)
# ------------------------------------------------------------------------
def apply_custom_q_attack(img_bgr, q_table=CUSTOM_Q_TABLE):
    Y, Cr, Cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
    h, w = Y.shape
    h8, w8 = h - h % 8, w - w % 8
    Yf = Y[:h8, :w8].astype(np.float64)
    out = Yf.copy()
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = Yf[i:i + 8, j:j + 8] - 128.0
            d = dct(dct(block.T, norm="ortho").T, norm="ortho")
            q = np.round(d / q_table)
            dq = q * q_table
            rec = idct(idct(dq.T, norm="ortho").T, norm="ortho") + 128.0
            out[i:i + 8, j:j + 8] = rec
    Y_new = Y.copy()
    Y_new[:h8, :w8] = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Y_new, Cr, Cb]), cv2.COLOR_YCrCb2BGR)


# ------------------------------------------------------------------------
# 3a) White-box adversarial attack (flip every embedded bit, clip to epsilon)
# ------------------------------------------------------------------------
def adversarial_attack(stego_bgr, epsilon):
    Y, Cr, Cb = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = dstg._to_blocks(Y)

    n_bits, K, version, header_ok = dstg._read_header(blocks)
    if not header_ok:
        return None

    nr, nc = blocks.shape[:2]
    slot_order = dstg._build_slot_order(K, nr, nc)
    payload_slots = slot_order[:n_bits]

    for (bi, bj) in dstg.HEADER_BLOCKS:
        for (u, v) in dstg._HEADER_POSITIONS:
            cur_bit = dstg._lsb_read(blocks[bi, bj, u, v], u, v)
            blocks[bi, bj, u, v] = dstg._lsb_write(blocks[bi, bj, u, v], u, v, 1 - cur_bit, flip_dir=-1)

    for (i, j, u, v) in payload_slots:
        cur_bit = dstg._lsb_read(blocks[i, j, u, v], u, v)
        blocks[i, j, u, v] = dstg._lsb_write(blocks[i, j, u, v], u, v, 1 - cur_bit, flip_dir=-1)

    y_target = dstg._from_blocks(blocks, Y, h8, w8)

    delta = y_target.astype(np.int16) - Y.astype(np.int16)
    delta = np.clip(delta, -epsilon, epsilon)
    y_perturbed = np.clip(Y.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([y_perturbed, Cr, Cb]), cv2.COLOR_YCrCb2BGR)


# ------------------------------------------------------------------------
# 3c) Alignment attacks
# ------------------------------------------------------------------------
def shift_image(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def crop_pad(img, px):
    h, w = img.shape[:2]
    cropped = img[px:h - px, px:w - px]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


ALIGNMENT_TESTS = [
    ("shift_1px", lambda im: shift_image(im, 1, 0)),
    ("shift_2px", lambda im: shift_image(im, 2, 0)),
    ("rotate_1deg", lambda im: rotate_image(im, 1)),
    ("crop_pad_2px", lambda im: crop_pad(im, 2)),
]


# ------------------------------------------------------------------------
# 3b) ECC (Hamming 7,4) + Monte-Carlo benefit simulation
# ------------------------------------------------------------------------
def hamming74_encode(bits):
    bits = bits + [0] * ((-len(bits)) % 4)
    encoded = []
    for i in range(0, len(bits), 4):
        d1, d2, d3, d4 = bits[i:i + 4]
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        encoded += [p1, p2, d1, p3, d2, d3, d4]
    return encoded


def hamming74_decode(bits):
    decoded = []
    for i in range(0, len(bits) - 6, 7):
        block = list(bits[i:i + 7])
        p1, p2, d1, p3, d2, d3, d4 = block
        c1 = p1 ^ d1 ^ d2 ^ d4
        c2 = p2 ^ d1 ^ d3 ^ d4
        c3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = c1 + (c2 << 1) + (c3 << 2)
        if syndrome != 0:
            block[syndrome - 1] ^= 1
            p1, p2, d1, p3, d2, d3, d4 = block
        decoded += [d1, d2, d3, d4]
    return decoded


def simulate_ecc_benefit(gt_bits, observed_ber):
    """Monte-Carlo: encode gt payload with Hamming(7,4), inject i.i.d. bit
    flips at rate=observed_ber (calibrated to the empirically observed raw
    BER for this attack), decode, compare to gt. This is a channel-model
    simulation, not a real re-embed test -- see module docstring."""
    if observed_ber <= 0:
        return True, 0.0
    encoded = hamming74_encode(gt_bits)
    n_flip = int(round(observed_ber * len(encoded)))
    n_flip = min(n_flip, len(encoded))
    noisy = encoded.copy()
    if n_flip > 0:
        flip_idx = np.random.choice(len(encoded), size=n_flip, replace=False)
        for idx in flip_idx:
            noisy[idx] ^= 1
    decoded = hamming74_decode(noisy)[:len(gt_bits)]
    match = decoded == gt_bits
    ber_after = sum(a != b for a, b in zip(decoded, gt_bits)) / len(gt_bits)
    return match, ber_after


# ------------------------------------------------------------------------
# Stats helper (same convention as your other pipelines)
# ------------------------------------------------------------------------
def mean_std_ci(values, confidence=CONFIDENCE):
    values = np.asarray(values, dtype=float)
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


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    paths = sorted(glob.glob(os.path.join(STEGO_DIR, "*")))[:MAX_IMAGES]
    print(f"Found {len(paths)} stego images (using up to {MAX_IMAGES}).")

    rows = []

    for path in tqdm(paths, desc="Images"):
        name = os.path.basename(path)
        stego_bgr = cv2.imread(path)
        if stego_bgr is None:
            rows.append({"image": name, "test_type": "load", "condition": "load",
                         "exact_match": False, "ber": 1.0, "length_match": False, "note": "unreadable_file"})
            continue

        gt_msg = safe_extract(stego_bgr)
        if gt_msg is None:
            rows.append({"image": name, "test_type": "ground_truth", "condition": "clean",
                         "exact_match": False, "ber": 1.0, "length_match": False, "note": "no_gt_extractable"})
            continue
        gt_bits = str_to_bits(gt_msg)

        def record(test_type, condition, result_str, ecc_source=False):
            stats_d = bit_diff_stats(gt_msg, result_str)
            row = {"image": name, "test_type": test_type, "condition": condition, **stats_d}
            rows.append(row)
            if ecc_source and f"{test_type}_{condition}".replace("__", "_") in ECC_SOURCE_TESTS or condition in ECC_SOURCE_TESTS:
                match, ber_after = simulate_ecc_benefit(gt_bits, stats_d["ber"])
                rows.append({
                    "image": name, "test_type": "ecc_simulated", "condition": f"{condition}_with_hamming74",
                    "exact_match": match, "ber": ber_after, "length_match": True,
                    "note": f"channel_model_from_{condition}",
                })
            return stats_d

        # 1b) clean control -- catches residual Q-table/header bugs
        record("custom_q_control", "clean", gt_msg)

        # 1a) custom-Q non-standard-table attack
        cq_img = apply_custom_q_attack(stego_bgr)
        cq_msg = safe_extract(cq_img)
        record("custom_q_attack", "custom_q_attack", cq_msg, ecc_source=True)

        # 3a) adversarial attack at multiple epsilon budgets
        for eps in ADV_EPSILONS:
            adv_img = adversarial_attack(stego_bgr, eps)
            adv_msg = safe_extract(adv_img) if adv_img is not None else None
            record("adversarial", f"adversarial_eps{eps}", adv_msg, ecc_source=True)

        # 3c) alignment attacks
        for label, fn in ALIGNMENT_TESTS:
            al_img = fn(stego_bgr)
            al_msg = safe_extract(al_img)
            record("alignment", label, al_msg)

    per_image_df = pd.DataFrame(rows)

    # ---- summary stats ----
    summary_rows = []
    for (test_type, condition), grp in per_image_df.groupby(["test_type", "condition"]):
        ber_mean, ber_std, ber_lo, ber_hi = mean_std_ci(grp["ber"])
        exact_rate = grp["exact_match"].mean()
        summary_rows.append({
            "test_type": test_type,
            "condition": condition,
            "n_images": len(grp),
            "exact_match_rate": exact_rate,
            "ber_mean": ber_mean,
            "ber_std": ber_std,
            "ber_ci_low": ber_lo,
            "ber_ci_high": ber_hi,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["test_type", "condition"])

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        per_image_df.to_excel(writer, sheet_name="per_image_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_stats", index=False)

    print(f"\nSaved -> {EXCEL_PATH}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()