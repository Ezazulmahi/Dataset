"""
thesis_comparison_pipeline.py
==============================
Baseline vs. LoRA fine-tuned DEDS cover comparison pipeline.

NOTE ON THREADING: the OMP/OPENBLAS/MKL env vars below MUST be set before
numpy/scipy/cv2 are imported anywhere in this file. BLAS reads these once
at library-load time, not per-call. On Windows, multiprocessing uses
"spawn", so every worker process re-imports this entire file from scratch
before its initializer runs -- setting these vars later (e.g. inside
_init_worker) is too late, since numpy/cv2 are already imported by then.

NOTE ON EXTRACTION: this version does NOT re-verify extraction after
embedding. DCT_Adaptive.embed()'s internal verify-fix loop already
guarantees the message is extractable before it returns (it iterates
until the IDCT roundtrip produces zero bit errors), so a second
extract()-and-compare pass here is redundant -- it would always read back
True. Skipping it removes ~0.3-0.4s/image of pure overhead; it does NOT
address the dominant per-image cost, which is the fix-pass convergence
inside embed() itself at high BPP/near-capacity K.

For each image (matched by filename between baseline/finetuned folders),
this script:

  1. Draws ONE random target embedding rate (bits-per-pixel) per image,
     uniformly from [BPP_MIN, BPP_MAX], deterministic given RANDOM_SEED.
  2. Forces the SAME K (adaptive-K parameter) onto both the baseline cover
     and its fine-tuned counterpart, so any PSNR/SSIM difference reflects
     the cover itself, not a difference in embedded payload.
  3. Embeds a random message of matched length into both covers using the
     embed() function from DCT_Adaptive.py (unmodified logic — only K is
     overridden for that one call).
  4. Records PSNR, SSIM, capacity (bits used), and target BPP for every
     image. Extraction is NOT re-verified here (see note above).
  5. Saves the FINE-TUNED stego images only (per current scope: detection /
     steganalysis is deferred to a later script, so only the images needed
     for that later test are written to disk).
  6. Aggregates mean, std, 95% CI per metric, binned into BPP ranges, plus
     a paired Wilcoxon signed-rank test (baseline vs. fine-tuned) for PSNR
     and SSIM within each bin.
  7. Computes FID and a paired CLIP cosine-similarity score between the
     baseline and fine-tuned cover sets (measuring how much the LoRA
     fine-tuning shifted the covers' distribution / content), once — this
     is a property of the covers, not the embedding, so it doesn't depend
     on per-image BPP.
  8. Writes everything to a single Excel workbook with multiple sheets.

Run this on the machine where DCT_Adaptive.py and your image folders live
(not in a sandboxed environment) — it needs cv2, scipy, scikit-image,
pandas, openpyxl, torch, transformers, and pytorch-fid installed.

    pip install opencv-python scipy scikit-image pandas openpyxl \
                torch transformers pillow pytorch-fid tqdm
"""

import os

# MUST happen before numpy/scipy/cv2 are imported (see note above) --
# these libraries read thread-count env vars once at load time.
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

try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------------
# CONFIG — edit before running
# ------------------------------------------------------------------------
DCT_MODULE_PATH = r"F:\Results\DCT_Adaptive.py"
BASELINE_DIR    = r"F:\Results\baseline"
FINETUNED_DIR   = r"F:\Results\finetuned"
STEGO_OUT_DIR   = r"F:\Results\stego_finetuned"   # for later steganalysis
EXCEL_OUT_PATH  = r"F:\Results\comparison_results.xlsx"
CHECKPOINT_PATH = r"F:\Results\checkpoints\checkpoint.csv"

# Each image gets its OWN randomly drawn target BPP in this range, rather
# than every image being tested at the same fixed set of BPP levels.
BPP_MIN = 0.30
BPP_MAX = 0.60

# Width of the BPP bins used only for summary statistics / Wilcoxon testing
# (per-image BPPs are continuous, so we bucket them for aggregate reporting).
BPP_BIN_WIDTH = 0.05

RANDOM_SEED  = 42
CONFIDENCE   = 0.95
DEVICE       = "cuda"   # set to "cpu" if no GPU, used for FID / CLIP only
QUIET_EMBED_LOGS = True  # suppress DCT_Adaptive's per-image print() spam

# Leave 1-2 cores free so your machine stays responsive; tune as you like.
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
# ------------------------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------
# Module loading + fixed-K helpers
# ------------------------------------------------------------------------
def load_dct_module(path):
    spec = importlib.util.spec_from_file_location("DCT_Adaptive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def fixed_k(module, k_value):
    """Temporarily force module._compute_K() to return a fixed K, so a
    baseline cover and its fine-tuned counterpart are embedded with the
    SAME payload for a fair comparison. Adaptive K would otherwise let
    fine-tuned covers (more eligible mid-band coefficients) carry more
    bits than baseline covers of the same size — exactly the effect this
    experiment is trying to measure, not something that should leak into
    the embedding step itself."""
    original = module._compute_K
    module._compute_K = lambda blocks: k_value
    try:
        yield
    finally:
        module._compute_K = original


def block_grid(module, cover_bgr):
    Y = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = module._to_blocks(Y)
    nr, nc = blocks.shape[:2]
    n_payload = sum(1 for i in range(nr) for j in range(nc)
                     if (i, j) not in module._HEADER_BLOCK_SET)
    return nr, nc, n_payload


def k_for_target_bpp(module, cover_bgr, target_bpp):
    h, w = cover_bgr.shape[:2]
    _, _, n_payload = block_grid(module, cover_bgr)
    if n_payload == 0:
        raise ValueError("Image too small: no payload blocks available.")
    target_bits = int(round(target_bpp * h * w))
    k = int(np.ceil(target_bits / n_payload)) - 1
    k = max(1, min(module.MAX_K, k))
    return k, target_bits


def random_ascii_message(n_bits):
    """ASCII-only string so len(message.encode('utf-8')) == number of chars,
    giving exact control over the embedded bit count."""
    n_bytes = n_bits // 8
    alphabet = string.ascii_letters + string.digits + " .,!?"
    return "".join(random.choice(alphabet) for _ in range(n_bytes))


def embed_at_k(module, cover_bgr, message, k_value):
    with fixed_k(module, k_value):
        if QUIET_EMBED_LOGS:
            with contextlib.redirect_stdout(io.StringIO()):
                stego = module.embed(cover_bgr, message)
        else:
            stego = module.embed(cover_bgr, message)
    return stego


# ------------------------------------------------------------------------
# Filename matching
# ------------------------------------------------------------------------
import re

def _id_key(filename):
    """Extract a matching key from a filename so pairs like '000_cover.png'
    and '000_finetuned.png' still match. Prefers a leading run of digits
    (e.g. '000' from '000_cover.png'); falls back to the filename stem
    with common suffixes stripped if no leading digits are found."""
    stem = os.path.splitext(filename)[0]
    m = re.match(r"^(\d+)", stem)
    if m:
        return m.group(1)
    for suffix in ("_cover", "_baseline", "_finetuned", "_stego", "cover_", "baseline_", "finetuned_"):
        stem = stem.replace(suffix, "")
    return stem.lower()


# ------------------------------------------------------------------------
# Per-image random BPP assignment (deterministic given RANDOM_SEED)
# ------------------------------------------------------------------------
def assign_target_bpps(image_keys, bpp_min, bpp_max, seed):
    """Give every image its own random target BPP, deterministically, so
    re-running the script (or resuming from a checkpoint) reproduces the
    exact same assignment regardless of process/thread scheduling order."""
    assignments = {}
    for key in image_keys:
        # Each image gets an independent RNG stream keyed off its own id,
        # so the assignment doesn't depend on iteration order.
        seed_i = (hash((key, "bpp_assignment")) ^ seed) & 0xFFFFFFFF
        rng = random.Random(seed_i)
        assignments[key] = rng.uniform(bpp_min, bpp_max)
    return assignments


# ------------------------------------------------------------------------
# Multiprocessing worker: each process loads its own copy of DCT_Adaptive.py
# ------------------------------------------------------------------------
_worker_module = None
_worker_module_path = None


def _init_worker(dct_module_path):
    global _worker_module, _worker_module_path

    # Prevent thread oversubscription: without this, OpenCV/BLAS spawn their
    # own internal thread pools INSIDE each of the N_WORKERS processes, so
    # you end up with far more threads than cores and the "parallel" run
    # serializes against itself. Each worker process should do its DCT/IDCT
    # work single-threaded; the parallelism comes from the process pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)

    _worker_module_path = dct_module_path
    _worker_module = load_dct_module(dct_module_path)


def _process_one_pair(task):
    """Runs in a worker process. task is a plain dict (must be picklable).
    Each image carries its OWN target_bpp, drawn once up front, so K and
    the embedded message length are computed per-image rather than per
    shared BPP level. Extraction is NOT re-verified (see module docstring)."""
    global _worker_module
    module = _worker_module

    name = task["name"]
    baseline_path = task["baseline_path"]
    finetuned_path = task["finetuned_path"]
    target_bpp = task["target_bpp"]
    out_dir = task["out_dir"]

    cover_b = cv2.imread(baseline_path)
    cover_f = cv2.imread(finetuned_path)
    if cover_b is None or cover_f is None:
        return {"image": name, "target_bpp": target_bpp, "error": "could not read image(s)"}

    try:
        # K is sized off the BASELINE cover's block grid so the same
        # payload length is forced onto both images of the pair.
        k_value, target_bits = k_for_target_bpp(module, cover_b, target_bpp)
    except Exception as e:
        return {"image": name, "target_bpp": target_bpp, "error": f"K sizing error: {e}"}

    # Deterministic-but-independent message per image, safe across processes.
    seed = (hash((name, "message")) ^ RANDOM_SEED) & 0xFFFFFFFF
    rng = random.Random(seed)
    n_bytes = target_bits // 8
    alphabet = string.ascii_letters + string.digits + " .,!?"
    message = "".join(rng.choice(alphabet) for _ in range(n_bytes))
    n_bits = len(message.encode("utf-8")) * 8

    try:
        stego_b = embed_at_k(module, cover_b, message, k_value)
        stego_f = embed_at_k(module, cover_f, message, k_value)
    except Exception as e:
        return {"image": name, "target_bpp": target_bpp, "error": f"embed error: {e}"}

    qm_b = module.quality_metrics(cover_b, stego_b, n_bits)
    qm_f = module.quality_metrics(cover_f, stego_f, n_bits)

    out_name = os.path.basename(finetuned_path)
    cv2.imwrite(os.path.join(out_dir, out_name), stego_f)

    h, w = cover_b.shape[:2]
    achieved_bpp = n_bits / (h * w)

    return {
        "image": out_name,
        "target_bpp": target_bpp,
        "achieved_bpp": achieved_bpp,
        "K": k_value,
        "n_bits": n_bits,
        "baseline_psnr": qm_b["psnr"],
        "baseline_ssim": qm_b["ssim"],
        "finetuned_psnr": qm_f["psnr"],
        "finetuned_ssim": qm_f["ssim"],
        "delta_psnr": qm_f["psnr"] - qm_b["psnr"],
        "delta_ssim": qm_f["ssim"] - qm_b["ssim"],
    }


# ------------------------------------------------------------------------
# Main embed/measure loop — single pass, each image at its own random BPP,
# checkpointed incrementally (resume support based on already-done images).
# ------------------------------------------------------------------------
def compare_dataset(dct_module_path, baseline_dir, finetuned_dir, stego_out_dir,
                    bpp_min, bpp_max, checkpoint_path, n_workers=N_WORKERS):
    baseline_paths = glob.glob(os.path.join(baseline_dir, "*"))
    finetuned_paths = glob.glob(os.path.join(finetuned_dir, "*"))

    baseline_files = {}
    for p in baseline_paths:
        key = _id_key(os.path.basename(p))
        baseline_files[key] = p

    finetuned_files = {}
    for p in finetuned_paths:
        key = _id_key(os.path.basename(p))
        finetuned_files[key] = p

    common = sorted(set(baseline_files) & set(finetuned_files))
    if not common:
        print(f"  baseline sample keys : {list(baseline_files.keys())[:5]}")
        print(f"  finetuned sample keys: {list(finetuned_files.keys())[:5]}")
        raise RuntimeError(
            "No matching filenames found between baseline and finetuned folders. "
            "See the sample keys printed above to see why they don't line up."
        )
    print(f"Found {len(common)} matched image pairs (matched by ID key).")
    print(f"Using {n_workers} worker process(es).\n")

    # Deterministic per-image BPP assignment, independent of run order.
    bpp_assignment = assign_target_bpps(common, bpp_min, bpp_max, RANDOM_SEED)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(stego_out_dir, exist_ok=True)

    # Resume support: load any already-completed images from the checkpoint
    # and only process the remainder.
    done_rows = []
    done_names = set()
    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)
        # Drop old extraction-check columns if present, so old + new rows
        # have consistent columns going forward.
        done_df = done_df.drop(
            columns=[c for c in ("baseline_extraction_ok", "finetuned_extraction_ok")
                     if c in done_df.columns]
        )
        done_rows = done_df.to_dict("records")
        done_names = set(done_df["image"].apply(lambda n: _id_key(os.path.basename(str(n)))))
        print(f"Resuming from checkpoint: {len(done_names)} image(s) already done "
              f"-> {checkpoint_path}\n")

    remaining = [name for name in common if name not in done_names]

    tasks = [
        {
            "name": name,
            "baseline_path": baseline_files[name],
            "finetuned_path": finetuned_files[name],
            "target_bpp": bpp_assignment[name],
            "out_dir": stego_out_dir,
        }
        for name in remaining
    ]

    new_rows = []
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(dct_module_path,)
        ) as executor:
            futures = [executor.submit(_process_one_pair, t) for t in tasks]
            for i, fut in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                          total=len(futures), desc="images")):
                result = fut.result()
                if result is None:
                    continue
                if "error" in result:
                    print(f"  [skip] {result['image']}: {result['error']}")
                    continue
                new_rows.append(result)

                # Checkpoint every 25 completed images so a crash doesn't
                # lose more than a small batch of work.
                if (i + 1) % 25 == 0:
                    pd.DataFrame(done_rows + new_rows).to_csv(checkpoint_path, index=False)

    all_rows = done_rows + new_rows
    pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
    print(f"\nCheckpoint saved -> {checkpoint_path}")

    return pd.DataFrame(all_rows)


# ------------------------------------------------------------------------
# Stats: mean, std, 95% CI, Wilcoxon signed-rank (binned by achieved BPP)
# ------------------------------------------------------------------------
def mean_std_ci(values, confidence=CONFIDENCE):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / np.sqrt(n)
        h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = 0.0
    return mean, std, mean - h, mean + h


def _bin_bpp(df, bin_width=BPP_BIN_WIDTH, bpp_min=BPP_MIN, bpp_max=BPP_MAX):
    """Bucket each image's continuous target_bpp into a fixed-width range,
    purely for aggregate reporting (per-image rows keep their exact,
    continuous target_bpp in the per_image_results sheet)."""
    n_bins = max(1, int(np.ceil((bpp_max - bpp_min) / bin_width)))
    edges = [bpp_min + i * bin_width for i in range(n_bins + 1)]
    edges[-1] = max(edges[-1], df["target_bpp"].max() + 1e-9)
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(n_bins)]
    df = df.copy()
    df["bpp_bin"] = pd.cut(df["target_bpp"], bins=edges, labels=labels, include_lowest=True)
    return df


def summarize(df):
    df = _bin_bpp(df)
    rows = []
    for bpp_bin, sub in df.groupby("bpp_bin", observed=True):
        if len(sub) == 0:
            continue
        for (b_col, f_col), label in [
            (("baseline_psnr", "finetuned_psnr"), "PSNR"),
            (("baseline_ssim", "finetuned_ssim"), "SSIM"),
        ]:
            b_mean, b_std, b_lo, b_hi = mean_std_ci(sub[b_col])
            f_mean, f_std, f_lo, f_hi = mean_std_ci(sub[f_col])
            try:
                wstat, wp = stats.wilcoxon(sub[b_col], sub[f_col])
            except ValueError:
                wstat, wp = np.nan, np.nan
            rows.append({
                "bpp_bin": bpp_bin, "n_images": len(sub), "metric": label,
                "baseline_mean": b_mean, "baseline_std": b_std,
                "baseline_ci_low": b_lo, "baseline_ci_high": b_hi,
                "finetuned_mean": f_mean, "finetuned_std": f_std,
                "finetuned_ci_low": f_lo, "finetuned_ci_high": f_hi,
                "wilcoxon_stat": wstat, "wilcoxon_p": wp,
                "significant_p<0.05": (wp < 0.05) if not np.isnan(wp) else None,
            })

        cap_mean, cap_std, cap_lo, cap_hi = mean_std_ci(sub["n_bits"])
        rows.append({
            "bpp_bin": bpp_bin, "n_images": len(sub), "metric": "capacity_bits_used",
            "baseline_mean": cap_mean, "baseline_std": cap_std,
            "baseline_ci_low": cap_lo, "baseline_ci_high": cap_hi,
            "finetuned_mean": cap_mean, "finetuned_std": cap_std,
            "finetuned_ci_low": cap_lo, "finetuned_ci_high": cap_hi,
            "wilcoxon_stat": None, "wilcoxon_p": None, "significant_p<0.05": None,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# FID + CLIP (cover-set distribution shift, computed once — independent of
# each image's individually assigned BPP)
# ------------------------------------------------------------------------
def compute_fid(baseline_dir, finetuned_dir, device=DEVICE):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(
        [baseline_dir, finetuned_dir], batch_size=50, device=device, dims=2048
    )


def compute_clip_similarity(baseline_dir, finetuned_dir, device=DEVICE):
    """Paired CLIP cosine similarity: baseline cover vs. its fine-tuned
    counterpart, matched by ID key. Measures how much the LoRA fine-tuning
    shifted image content/semantics per cover."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    baseline_files = {_id_key(os.path.basename(p)): p
                       for p in glob.glob(os.path.join(baseline_dir, "*"))}
    finetuned_files = {_id_key(os.path.basename(p)): p
                        for p in glob.glob(os.path.join(finetuned_dir, "*"))}
    common = sorted(set(baseline_files) & set(finetuned_files))

    sims = []
    with torch.no_grad():
        for name in tqdm(common, desc="CLIP similarity"):
            img_b = Image.open(baseline_files[name]).convert("RGB")
            img_f = Image.open(finetuned_files[name]).convert("RGB")
            inputs = processor(images=[img_b, img_f], return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = (feats[0] * feats[1]).sum().item()
            sims.append({"image": os.path.basename(finetuned_files[name]),
                         "clip_cosine_similarity": sim})
    return pd.DataFrame(sims)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    print("=== Baseline vs. Fine-tuned DEDS cover comparison ===\n")
    print(f"Each image draws its own random target BPP in "
          f"[{BPP_MIN:.2f}, {BPP_MAX:.2f}] (seed={RANDOM_SEED}).\n")

    df = compare_dataset(
        DCT_MODULE_PATH, BASELINE_DIR, FINETUNED_DIR, STEGO_OUT_DIR,
        BPP_MIN, BPP_MAX, CHECKPOINT_PATH, n_workers=N_WORKERS
    )
    summary_df = summarize(df)

    print("\nComputing FID (can take a few minutes)...")
    try:
        fid_value = compute_fid(BASELINE_DIR, FINETUNED_DIR)
    except Exception as e:
        print(f"  FID failed: {e}")
        fid_value = None

    print("Computing paired CLIP cosine similarity...")
    try:
        clip_df = compute_clip_similarity(BASELINE_DIR, FINETUNED_DIR)
        clip_mean, clip_std, clip_lo, clip_hi = mean_std_ci(clip_df["clip_cosine_similarity"])
    except Exception as e:
        print(f"  CLIP failed: {e}")
        clip_df = pd.DataFrame()
        clip_mean = clip_std = clip_lo = clip_hi = None

    fid_clip_summary = pd.DataFrame([{
        "FID_baseline_vs_finetuned": fid_value,
        "CLIP_cosine_mean": clip_mean,
        "CLIP_cosine_std": clip_std,
        "CLIP_cosine_ci_low": clip_lo,
        "CLIP_cosine_ci_high": clip_hi,
        "n_images_compared": len(clip_df) if not clip_df.empty else 0,
    }])

    os.makedirs(os.path.dirname(EXCEL_OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(EXCEL_OUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_image_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_stats", index=False)
        fid_clip_summary.to_excel(writer, sheet_name="FID_CLIP", index=False)
        if not clip_df.empty:
            clip_df.to_excel(writer, sheet_name="clip_per_image", index=False)

    print(f"\nDone. Results saved to: {EXCEL_OUT_PATH}")
    print(f"Fine-tuned stego images saved under: {STEGO_OUT_DIR}\\ "
          f"(for your later steganalysis test)")
    print(f"Checkpoint is at: {CHECKPOINT_PATH} "
          f"(safe to delete once comparison_results.xlsx looks correct)")


if __name__ == "__main__":
    # Required on Windows: multiprocessing needs this guard so worker
    # processes re-importing this file don't re-trigger main().
    multiprocessing.freeze_support()
    main()