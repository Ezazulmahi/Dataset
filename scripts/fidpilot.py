"""
fid_clip_pilot5.py
====================
Pilot-scale FID + CLIP step for a small 5-image test set. Computes BOTH
comparisons in one run:

  1) baseline vs finetuned   -> measures how much LoRA fine-tuning shifted covers
  2) finetuned vs stego      -> measures how much embedding shifted the covers

For each comparison it computes FID and paired CLIP cosine similarity, and
writes results to 4 sheets in EXCEL_PATH:
    FID_CLIP_baseline_vs_finetuned
    clip_per_image_baseline_vs_finetuned
    FID_CLIP_finetuned_vs_stego
    clip_per_image_finetuned_vs_stego

Any other existing sheets in EXCEL_PATH are read first and rewritten
untouched, so this is safe to run against a workbook that already has
per_image_results / summary_stats / etc. If EXCEL_PATH does not exist yet,
a new workbook is created with just these 4 sheets.

NOTE on n=5: pytorch-fid's FID score relies on estimating a covariance
matrix from Inception features and is only statistically meaningful with
much larger samples (typically 1000+ images per side). With 5 images the
FID number here is a rough pilot signal only -- treat it as a smoke test,
not a reportable metric.

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
# CONFIG
# ------------------------------------------------------------------------
BASELINE_DIR   = r"F:\Results\Baselin5"
FINETUNED_DIR  = r"F:\Results\finetunecover"
STEGO_DIR      = r"F:\Results\stego_pilot"
EXCEL_PATH     = r"F:\Results\comparison_results_pilot5.xlsx"
DEVICE         = "cuda"   # set to "cpu" if no GPU
CONFIDENCE     = 0.95

# (sheet_prefix, dir_a, dir_b) -- dir_a is the "reference" side, dir_b is
# the "compared" side; clip_per_image filename column reflects dir_b.
COMPARISONS = [
    ("baseline_vs_finetuned", BASELINE_DIR, FINETUNED_DIR),
    ("finetuned_vs_stego",    FINETUNED_DIR, STEGO_DIR),
]


def _id_key(filename):
    """Same matching logic as comparison.py, so pairs line up identically."""
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


def compute_clip_similarity(dir_a, dir_b, device=DEVICE, model=None, processor=None):
    import torch
    from PIL import Image

    a_files = {_id_key(os.path.basename(p)): p
               for p in glob.glob(os.path.join(dir_a, "*"))}
    b_files = {_id_key(os.path.basename(p)): p
               for p in glob.glob(os.path.join(dir_b, "*"))}
    common = sorted(set(a_files) & set(b_files))
    if not common:
        raise RuntimeError(
            f"No matching filename pairs found between {dir_a} and {dir_b}. "
            f"Check that both folders use the same basenames."
        )
    print(f"  {len(common)} matched pair(s) found.")

    sims = []
    with torch.no_grad():
        for name in tqdm(common, desc="CLIP similarity"):
            img_a = Image.open(a_files[name]).convert("RGB")
            img_b = Image.open(b_files[name]).convert("RGB")
            inputs = processor(images=[img_a, img_b], return_tensors="pt").to(device)

            feats = model.get_image_features(**inputs)
            # Newer transformers versions can return BaseModelOutputWithPooling
            # instead of a raw tensor -- normalize to a plain tensor either way.
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


def run_comparison(label, dir_a, dir_b, model, processor):
    print(f"\n=== {label}: {dir_a}  vs  {dir_b} ===")

    print("Computing FID...")
    try:
        fid_value = compute_fid(dir_a, dir_b, device=DEVICE)
        print(f"  FID = {fid_value:.4f}")
    except Exception as e:
        print(f"  FID failed: {e}")
        fid_value = None

    print("Computing paired CLIP cosine similarity...")
    try:
        clip_df = compute_clip_similarity(dir_a, dir_b, device=DEVICE,
                                           model=model, processor=processor)
        clip_mean, clip_std, clip_lo, clip_hi = mean_std_ci(clip_df["clip_cosine_similarity"])
        print(f"  n={len(clip_df)}  mean={clip_mean:.4f}  std={clip_std:.4f}  "
              f"95% CI=[{clip_lo:.4f}, {clip_hi:.4f}]")
    except Exception as e:
        print(f"  CLIP failed: {e}")
        clip_df = pd.DataFrame()
        clip_mean = clip_std = clip_lo = clip_hi = None

    summary = pd.DataFrame([{
        f"FID_{label}": fid_value,
        "CLIP_cosine_mean": clip_mean,
        "CLIP_cosine_std": clip_std,
        "CLIP_cosine_ci_low": clip_lo,
        "CLIP_cosine_ci_high": clip_hi,
        "n_images_compared": len(clip_df) if not clip_df.empty else 0,
    }])

    return summary, clip_df


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP model once for both comparisons...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load existing workbook (if any) so unrelated sheets survive the rewrite.
    if os.path.exists(EXCEL_PATH):
        existing = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
    else:
        existing = {}

    for label, dir_a, dir_b in COMPARISONS:
        summary_df, clip_df = run_comparison(label, dir_a, dir_b, model, processor)
        existing[f"FID_CLIP_{label}"] = summary_df
        if not clip_df.empty:
            existing[f"clip_per_image_{label}"] = clip_df

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet_name, sheet_df in existing.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nUpdated {EXCEL_PATH} with sheets for: "
          + ", ".join(label for label, _, _ in COMPARISONS))


if __name__ == "__main__":
    main()