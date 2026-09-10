"""
fid_clip_only.py
==================
Standalone FID + CLIP step. Run this on its own -- it does NOT redo the
embedding sweep (your PSNR/SSIM comparison is already done and untouched).

This version computes FID and CLIP cosine similarity between the
FINE-TUNED COVER images and their STEGO (finetuned) counterparts, i.e. it
measures how much the embedding step itself shifts the fine-tuned covers,
rather than baseline-vs-finetuned (which measures how much the LoRA
fine-tuning itself shifted the covers). Both comparisons matter and are
kept separate: this script writes to NEW sheets
(FID_CLIP_finetuned_vs_stego, clip_per_image_finetuned_vs_stego) in your
existing comparison_results.xlsx, leaving per_image_results,
summary_stats, FID_CLIP, and clip_per_image (baseline vs finetuned)
exactly as they are.

Fix vs. the original script: newer transformers versions can return a
BaseModelOutputWithPooling from get_image_features() instead of a raw
tensor depending on call signature -- that's what caused
'BaseModelOutputWithPooling' object has no attribute 'norm'. This version
normalizes to a plain tensor either way before calling .norm().

    pip install torch transformers pillow pytorch-fid pandas openpyxl scipy tqdm
"""

import os
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------------
# CONFIG -- must match what you used in comparison.py / steganalysis_pipeline.py
# ------------------------------------------------------------------------
FINETUNED_DIR  = r"F:\Results\finetuned"
STEGO_DIR      = r"F:\Results\stego_finetuned"   # flat folder, same basenames as finetuned covers
EXCEL_PATH     = r"F:\Results\comparison_results.xlsx"
DEVICE         = "cuda"   # set to "cpu" if no GPU
CONFIDENCE     = 0.95

# Sheet names -- kept distinct from the baseline-vs-finetuned sheets so
# nothing in the existing workbook gets overwritten.
FID_CLIP_SHEET_NAME  = "FID_CLIP_finetuned_vs_stego"
CLIP_PER_IMG_SHEET_NAME = "clip_per_image_finetuned_vs_stego"


def _id_key(filename):
    """Same matching logic as comparison.py, so pairs line up identically.
    Since stego_finetuned/ uses the same basename as its source finetuned
    cover (per steganalysis_pipeline.py's out_name convention), this will
    typically just match on the full stem, but we keep the same fallback
    logic for consistency with your other scripts."""
    stem = os.path.splitext(filename)[0]
    m = re.match(r"^(\d+)", stem)
    if m:
        return m.group(1)
    for suffix in ("_cover", "_baseline", "_finetuned", "_stego", "cover_", "baseline_", "finetuned_"):
        stem = stem.replace(suffix, "")
    return stem.lower()


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


def compute_fid(dir_a, dir_b, device=DEVICE):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(
        [dir_a, dir_b], batch_size=50, device=device, dims=2048
    )


def compute_clip_similarity(dir_a, dir_b, device=DEVICE):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    a_files = {_id_key(os.path.basename(p)): p
               for p in glob.glob(os.path.join(dir_a, "*"))}
    b_files = {_id_key(os.path.basename(p)): p
               for p in glob.glob(os.path.join(dir_b, "*"))}
    common = sorted(set(a_files) & set(b_files))
    if not common:
        raise RuntimeError(
            f"No matching filename pairs found between {dir_a} and {dir_b}. "
            f"Check STEGO_DIR uses the same basenames as FINETUNED_DIR."
        )
    print(f"  {len(common)} matched pair(s) found.")

    sims = []
    with torch.no_grad():
        for name in tqdm(common, desc="CLIP similarity"):
            img_a = Image.open(a_files[name]).convert("RGB")
            img_b = Image.open(b_files[name]).convert("RGB")
            inputs = processor(images=[img_a, img_b], return_tensors="pt").to(device)

            feats = model.get_image_features(**inputs)
            # Fix: normalize to a plain tensor regardless of what this
            # transformers version returns from get_image_features().
            if not isinstance(feats, torch.Tensor):
                unwrapped = getattr(feats, "image_embeds", None)
                if unwrapped is None:
                    unwrapped = getattr(feats, "pooler_output", None)
                if unwrapped is None:
                    raise RuntimeError(
                        f"get_image_features() returned {type(feats)}, which has "
                        f"neither .image_embeds nor .pooler_output -- print(dir(feats)) "
                        f"to see what's available on your transformers version."
                    )
                feats = unwrapped

            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = (feats[0] * feats[1]).sum().item()
            sims.append({
                "image": os.path.basename(b_files[name]),
                "clip_cosine_similarity": sim,
            })

    return pd.DataFrame(sims)


def main():
    print("Computing FID: finetuned covers vs stego (can take a few minutes)...")
    try:
        fid_value = compute_fid(FINETUNED_DIR, STEGO_DIR, device=DEVICE)
        print(f"  FID = {fid_value:.4f}")
    except Exception as e:
        print(f"  FID failed: {e}")
        fid_value = None

    print("\nComputing paired CLIP cosine similarity: finetuned covers vs stego...")
    try:
        clip_df = compute_clip_similarity(FINETUNED_DIR, STEGO_DIR, device=DEVICE)
        clip_mean, clip_std, clip_lo, clip_hi = mean_std_ci(clip_df["clip_cosine_similarity"])
        print(f"  n={len(clip_df)}  mean={clip_mean:.4f}  std={clip_std:.4f}  "
              f"95% CI=[{clip_lo:.4f}, {clip_hi:.4f}]")
    except Exception as e:
        print(f"  CLIP failed: {e}")
        clip_df = pd.DataFrame()
        clip_mean = clip_std = clip_lo = clip_hi = None

    fid_clip_summary = pd.DataFrame([{
        "FID_finetuned_vs_stego": fid_value,
        "CLIP_cosine_mean": clip_mean,
        "CLIP_cosine_std": clip_std,
        "CLIP_cosine_ci_low": clip_lo,
        "CLIP_cosine_ci_high": clip_hi,
        "n_images_compared": len(clip_df) if not clip_df.empty else 0,
    }])

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"{EXCEL_PATH} not found -- update EXCEL_PATH above to point at "
            f"your existing comparison_results.xlsx."
        )

    # Read every existing sheet so per_image_results / summary_stats /
    # FID_CLIP / clip_per_image (baseline vs finetuned) all survive the
    # rewrite untouched; only the two new sheets below are added/replaced.
    existing = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
    existing[FID_CLIP_SHEET_NAME] = fid_clip_summary
    if not clip_df.empty:
        existing[CLIP_PER_IMG_SHEET_NAME] = clip_df

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet_name, sheet_df in existing.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nUpdated {EXCEL_PATH} -- {FID_CLIP_SHEET_NAME} sheet added/refreshed"
          + (f", {CLIP_PER_IMG_SHEET_NAME} sheet added" if not clip_df.empty else ""))


if __name__ == "__main__":
    main()