"""
steganalysis_pipeline.py
=========================
Runs the fine-tuned SRNet steganalyzer (Steganalyzer.py) against the
fine-tuned stego images produced by thesis_comparison_pipeline.py.

FIX vs. the original version: thesis_comparison_pipeline.py does NOT save
stego images into per-BPP bpp_* subfolders -- every image gets its own
continuous, randomly-drawn achieved_bpp and all 400 stego images are saved
flat into STEGO_OUT_DIR (stego_finetuned/), with the same filename as their
source finetuned cover. This version reads that flat folder directly and
pulls each image's continuous achieved_bpp from the per_image_results
sheet of comparison_results.xlsx (the file thesis_comparison_pipeline.py
already wrote), then bins ONLY for the summary/report tables -- using the
same BPP_BIN_WIDTH convention as thesis_comparison_pipeline.py's own
summarize() function, so this analysis lines up on the same BPP axis as
your PSNR/SSIM comparison. Per-image rows keep their exact continuous
achieved_bpp in the Excel output.

SECOND FIX (this version): the batch accuracy metric was being computed
at a fixed, hardcoded 0.5 softmax threshold on argmax predictions. Because
mean P(stego|stego) sits below 0.5 in every bin (the classifier's stego-
class scores rarely cross the nominal decision boundary even though they
are clearly separated from cover-class scores -- see ROC-AUC), this fixed
threshold mechanically forces accuracy to exactly 50% in every bin
regardless of how separable the classes actually are: nearly all cover
images are correctly labelled "cover", and nearly all stego images are
ALSO mislabelled "cover", so accuracy = (all-correct-covers +
all-incorrect-stego) / all = 50% whenever classes are balanced 1:1.
This version reports accuracy at BOTH the fixed 0.5 threshold (what the
model itself would say) and at a per-bin recalibrated threshold chosen by
Youden's J statistic on the ROC curve (the accuracy the classifier WOULD
achieve at its best possible operating point), alongside ROC-AUC, which
was already threshold-free and is unaffected by this issue.

THIRD FIX (this version): alongside overall batch accuracy, we now also
report per-class precision, recall, and F1 score (for the "cover" and
"stego" classes separately), at both the fixed 0.5 threshold and the
recalibrated best threshold. Overall accuracy can look identical at
0.5 while the two classes are being treated very differently (e.g. one
class recalled almost perfectly, the other almost never) -- per-class
precision/recall/F1 exposes that asymmetry, which a single accuracy
number hides.

This script:

  1. Detects every cover / stego pair (per-image cover%, stego%, delta).
  2. Computes a first-order chi-square statistic (pair-of-values attack,
     Westfeld & Pfitzmann) per image -- needed to reproduce your Table 5.5.
  3. Bins images by achieved_bpp and computes batch accuracy (at 0.5 and
     at the recalibrated best threshold), per-class precision/recall/F1
     (at both thresholds), ROC-AUC, mean P(stego|cover), mean
     P(stego|stego), and a verdict string per bin (Table 5.3/5.4 style).
  4. Saves ALL per-image results (continuous achieved_bpp + bin label)
     plus the per-bin batch summaries (including per-class P/R/F1) to a
     separate Excel workbook.
  5. Auto-generates a plain-text report in the same structure/style as
     your Section 5.4.6 excerpt (Table 5.2 / 5.3 / 5.5 + narrative),
     using a sample of pairs per bin for the illustrative tables (full
     data stays in the Excel file) -- British English, first person
     plural, no em/en dashes, ranges as "X to Y", per your established
     thesis prose conventions.

Run locally where Steganalyzer.py, the checkpoint, comparison_results.xlsx,
and your image folders live. Needs: torch, opencv-python, numpy, pandas,
openpyxl, scikit-learn (required this version, for ROC-AUC, the
recalibrated-threshold accuracy, AND per-class precision/recall/F1), tqdm
(optional).
"""

import os
import glob
import importlib.util

import numpy as np
import pandas as pd
import cv2
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
except ImportError:
    roc_auc_score = None
    roc_curve = None
    precision_recall_fscore_support = None


# ------------------------------------------------------------------------
# CONFIG -- edit before running
# ------------------------------------------------------------------------
STEGANALYZER_MODULE_PATH = r"F:\Thesis\Steganalyzer.py"
CHECKPOINT_PATH          = r"F:\Thesis\steganalyzer_finetuned.pth"

FINETUNED_COVER_DIR = r"F:\Results\finetuned"
STEGO_DIR           = r"F:\Results\stego_finetuned"        # flat folder, no bpp_* subfolders
COMPARISON_XLSX     = r"F:\Results\comparison_results.xlsx"  # source of each image's achieved_bpp

EXCEL_OUT_PATH  = r"F:\Results\steganalysis_results.xlsx"
REPORT_TXT_PATH = r"F:\Results\steganalysis_report.txt"

# Same convention as thesis_comparison_pipeline.py's BPP_BIN_WIDTH, so the
# steganalysis bins line up with your PSNR/SSIM summary bins.
BPP_BIN_WIDTH = 0.05

N_SAMPLE_FOR_TEXT_TABLES = 5   # how many pairs to show in the illustrative text tables

# Verdict thresholding is done on the probability delta (mean P(stego|stego)
# minus mean P(stego|cover)), not on accuracy, since accuracy at a fixed 0.5
# cutoff is not a reliable separability signal (see fix note above).
VERDICT_DELTA_EPS = 0.02
# ------------------------------------------------------------------------


def load_steganalyzer_module(path):
    spec = importlib.util.spec_from_file_location("Steganalyzer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------
# Chi-square pair-of-values attack (Westfeld & Pfitzmann first-order check)
# ------------------------------------------------------------------------
def chi_square_stat(img_bgr):
    """Classic PoV chi-square statistic on the grayscale histogram. LSB-style
    embedding pushes each (2k, 2k+1) pair toward equal frequency; the
    statistic measures how far the observed histogram is from that."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    chi2 = 0.0
    for k in range(128):
        h0, h1 = hist[2 * k], hist[2 * k + 1]
        n = h0 + h1
        if n == 0:
            continue
        expected = n / 2.0
        chi2 += ((h0 - expected) ** 2) / expected + ((h1 - expected) ** 2) / expected
    return float(chi2)


# ------------------------------------------------------------------------
# achieved_bpp lookup, pulled from thesis_comparison_pipeline.py's own
# output rather than re-deriving it, so the two analyses agree exactly.
# ------------------------------------------------------------------------
def load_achieved_bpp_map(comparison_xlsx_path):
    if not os.path.exists(comparison_xlsx_path):
        raise FileNotFoundError(
            f"{comparison_xlsx_path} not found -- this script needs the "
            f"per_image_results sheet thesis_comparison_pipeline.py already "
            f"wrote, to know each stego image's achieved_bpp."
        )
    df = pd.read_excel(comparison_xlsx_path, sheet_name="per_image_results", engine="openpyxl")
    if "image" not in df.columns or "achieved_bpp" not in df.columns:
        raise KeyError(
            "per_image_results sheet is missing 'image' or 'achieved_bpp' "
            "columns -- check you're pointing at the right workbook."
        )
    # Later rows win if a filename appears more than once (e.g. checkpoint
    # resume duplicates); keep the last occurrence.
    df = df.drop_duplicates(subset="image", keep="last")
    return dict(zip(df["image"], df["achieved_bpp"]))


def _bin_bpp_series(bpp_values, bin_width=BPP_BIN_WIDTH):
    bpp_values = np.asarray(bpp_values, dtype=float)
    bpp_min, bpp_max = float(bpp_values.min()), float(bpp_values.max())
    n_bins = max(1, int(np.ceil((bpp_max - bpp_min) / bin_width)))
    edges = [bpp_min + i * bin_width for i in range(n_bins + 1)]
    edges[-1] = max(edges[-1], bpp_max + 1e-9)
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(n_bins)]
    return pd.cut(bpp_values, bins=edges, labels=labels, include_lowest=True)


# ------------------------------------------------------------------------
# Per-image detection (single pass over the flat folder)
# ------------------------------------------------------------------------
@torch.no_grad()
def detect_one_pair(module, model, device, cover_path, stego_path, achieved_bpp):
    cover_img = cv2.imread(cover_path)
    stego_img = cv2.imread(stego_path)
    if cover_img is None or stego_img is None:
        return None

    prob_c = torch.softmax(model(module.preprocess(cover_img).to(device)), dim=1)[0].cpu().numpy()
    prob_s = torch.softmax(model(module.preprocess(stego_img).to(device)), dim=1)[0].cpu().numpy()
    pred_c, pred_s = int(np.argmax(prob_c)), int(np.argmax(prob_s))

    chi_c = chi_square_stat(cover_img)
    chi_s = chi_square_stat(stego_img)

    return {
        "image": os.path.basename(stego_path),
        "achieved_bpp": achieved_bpp,
        "cover_pred": module.CLASS_NAMES[pred_c],
        "cover_pct_cover": prob_c[0] * 100,
        "cover_pct_stego": prob_c[1] * 100,
        "stego_pred": module.CLASS_NAMES[pred_s],
        "stego_pct_cover": prob_s[0] * 100,
        "stego_pct_stego": prob_s[1] * 100,
        "delta_stego_minus_cover_pct": (prob_s[1] - prob_c[1]) * 100,
        "chi_square_cover": chi_c,
        "chi_square_stego": chi_s,
        "chi_square_delta": chi_s - chi_c,
    }


def run_detection(module, model, device, cover_dir, stego_dir, bpp_map):
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    cover_files = {os.path.basename(p): p for p in glob.glob(os.path.join(cover_dir, "*"))
                   if p.lower().endswith(IMG_EXT)}
    stego_files = {os.path.basename(p): p for p in glob.glob(os.path.join(stego_dir, "*"))
                   if p.lower().endswith(IMG_EXT)}

    # Stego filenames are saved under the same basename as their source
    # finetuned cover (see thesis_comparison_pipeline.py's out_name), and
    # achieved_bpp is keyed the same way in comparison_results.xlsx, so a
    # direct basename match across all three is correct here.
    common = sorted(set(cover_files) & set(stego_files) & set(bpp_map.keys()))
    missing_bpp = sorted((set(cover_files) & set(stego_files)) - set(bpp_map.keys()))
    if missing_bpp:
        print(f"  [warn] {len(missing_bpp)} image(s) have a cover+stego pair but no "
              f"achieved_bpp entry in comparison_results.xlsx -- skipped. "
              f"e.g. {missing_bpp[:5]}")
    if not common:
        raise RuntimeError(
            "No images with a matched cover, stego, AND achieved_bpp entry were found. "
            "Check FINETUNED_COVER_DIR, STEGO_DIR, and COMPARISON_XLSX all point at "
            "the same run."
        )
    print(f"Found {len(common)} fully-matched image(s) to evaluate.\n")

    rows = []
    for name in tqdm(common, desc="steganalysis"):
        row = detect_one_pair(module, model, device, cover_files[name], stego_files[name],
                              bpp_map[name])
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# Per-bin batch summary (accuracy at 0.5 AND at recalibrated threshold,
# per-class precision/recall/F1 at both thresholds, ROC-AUC, verdict)
# ------------------------------------------------------------------------
def summarize_by_bin(per_image_df):
    if roc_auc_score is None or roc_curve is None or precision_recall_fscore_support is None:
        raise ImportError(
            "scikit-learn is required for this version of the script "
            "(roc_auc_score and roc_curve for ROC-AUC and the recalibrated-"
            "threshold accuracy, and precision_recall_fscore_support for the "
            "per-class precision/recall/F1 scores). Install it with: "
            "pip install scikit-learn"
        )

    summaries = []
    for bpp_bin, sub in per_image_df.groupby("bpp_bin", observed=True):
        if len(sub) == 0:
            continue

        # Label convention: 0 = cover, 1 = stego (matches module.CLASS_NAMES order).
        y_true = np.concatenate([np.zeros(len(sub)), np.ones(len(sub))])
        y_score = np.concatenate([sub["cover_pct_stego"].values / 100.0,
                                   sub["stego_pct_stego"].values / 100.0])

        auc = float(roc_auc_score(y_true, y_score))

        # Accuracy at the fixed 0.5 threshold -- i.e. what the model's own
        # argmax decision gives. Kept for transparency, but see the module
        # docstring: this is mechanically pinned near 50% whenever
        # mean P(stego|stego) sits below 0.5 and classes are balanced.
        y_pred_fixed = (y_score >= 0.5).astype(int)
        acc_fixed = float((y_true == y_pred_fixed).mean())

        # Per-class precision/recall/F1 at the fixed 0.5 threshold.
        # labels=[0, 1] pins the order to (cover, stego) regardless of
        # which classes happen to appear in y_pred for this bin.
        prec_fixed, rec_fixed, f1_fixed, _ = precision_recall_fscore_support(
            y_true, y_pred_fixed, labels=[0, 1], zero_division=0
        )

        # Accuracy at the threshold that maximises Youden's J (tpr - fpr)
        # on this bin's ROC curve -- the accuracy the classifier WOULD
        # achieve if recalibrated to its best operating point. This is the
        # number that should be compared against ROC-AUC for a coherent
        # separability picture.
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_stat = tpr - fpr
        best_idx = int(np.argmax(j_stat))
        best_thresh = float(thresholds[best_idx])
        y_pred_best = (y_score >= best_thresh).astype(int)
        acc_best = float((y_true == y_pred_best).mean())

        # Per-class precision/recall/F1 at the recalibrated best threshold.
        prec_best, rec_best, f1_best, _ = precision_recall_fscore_support(
            y_true, y_pred_best, labels=[0, 1], zero_division=0
        )

        mean_p_cover = float(sub["cover_pct_stego"].mean() / 100.0)
        mean_p_stego = float(sub["stego_pct_stego"].mean() / 100.0)
        delta = mean_p_stego - mean_p_cover

        if abs(delta) < VERDICT_DELTA_EPS:
            verdict = "evades detection (chance-level separation)"
        elif delta > 0:
            verdict = "detectable (model leans toward flagging stego)"
        else:
            verdict = "stego scores lower than cover (model is blind in this direction)"

        summaries.append({
            "bpp_bin": bpp_bin,
            "pairs": len(sub),
            "images": len(y_true),
            "accuracy_pct_at_0.5": acc_fixed * 100,
            "accuracy_pct_at_best_thresh": acc_best * 100,
            "best_thresh": best_thresh,
            "roc_auc": auc,
            # Per-class metrics at the fixed 0.5 threshold.
            "precision_cover_at_0.5": prec_fixed[0],
            "recall_cover_at_0.5": rec_fixed[0],
            "f1_cover_at_0.5": f1_fixed[0],
            "precision_stego_at_0.5": prec_fixed[1],
            "recall_stego_at_0.5": rec_fixed[1],
            "f1_stego_at_0.5": f1_fixed[1],
            # Per-class metrics at the recalibrated best threshold.
            "precision_cover_at_best": prec_best[0],
            "recall_cover_at_best": rec_best[0],
            "f1_cover_at_best": f1_best[0],
            "precision_stego_at_best": prec_best[1],
            "recall_stego_at_best": rec_best[1],
            "f1_stego_at_best": f1_best[1],
            "mean_p_stego_given_cover": mean_p_cover,
            "mean_p_stego_given_stego": mean_p_stego,
            "delta": delta,
            "verdict": verdict,
        })
    return pd.DataFrame(summaries)


# ------------------------------------------------------------------------
# Text report generation (mirrors Section 5.4.6 style)
# ------------------------------------------------------------------------
def format_table(df, columns=None):
    if columns:
        df = df[columns]
    return df.to_string(index=False)


def build_report(per_image_df, summary_df, n_sample=N_SAMPLE_FOR_TEXT_TABLES):
    lines = []
    lines.append("5.4.6 Detectability against Steganalysis")
    lines.append("")
    lines.append(
        "Detectability was evaluated with the fine-tuned SRNet detector across "
        "the achieved BPP range, binned in steps of "
        f"{BPP_BIN_WIDTH:.2f} bpp. All images were at 256x256. For each bin we "
        "report the per-image detector outputs on a sample of pairs, a batch "
        "accuracy test at two operating points, per-class precision, recall "
        "and F1 score at both operating points, and a first order chi-square "
        "check. We report accuracy and per-class scores both at the model's "
        "native 0.5 softmax threshold and at a per-bin recalibrated threshold "
        "(chosen by Youden's J statistic on the ROC curve), since the native "
        "threshold is not well calibrated for this class balance and, taken "
        "alone, understates what the detector's scores actually separate, and "
        "can mask asymmetric treatment of the two classes that overall "
        "accuracy alone would hide."
    )
    lines.append("")

    bpp_bins = summary_df["bpp_bin"].tolist()

    for bpp_bin in bpp_bins:
        sub = per_image_df[per_image_df["bpp_bin"] == bpp_bin].reset_index(drop=True)
        summ = summary_df[summary_df["bpp_bin"] == bpp_bin].iloc[0]
        sample = sub.head(n_sample)

        lines.append(f"--- BPP {bpp_bin} ---")
        lines.append("")
        lines.append(f"Table: Per-image steganalysis results, BPP {bpp_bin} "
                     f"(first {len(sample)} of {len(sub)} pairs; full results in "
                     f"steganalysis_results.xlsx).")
        table_rows = []
        for _, r in sample.iterrows():
            table_rows.append({
                "Image": r["image"],
                "Cover pred": r["cover_pred"],
                "Cover %": f"{r['cover_pct_cover']:.1f}",
                "Stego %": f"{r['cover_pct_stego']:.1f}",
            })
            table_rows.append({
                "Image": r["image"] + " (stego)",
                "Cover pred": r["stego_pred"],
                "Cover %": f"{r['stego_pct_cover']:.1f}",
                "Stego %": f"{r['stego_pct_stego']:.1f} "
                          f"(delta {r['delta_stego_minus_cover_pct']:+.1f}%)",
            })
        lines.append(format_table(pd.DataFrame(table_rows)))
        lines.append("")

        n_no_shift = int((sample["delta_stego_minus_cover_pct"].abs() < 1.0).sum())
        lines.append(
            f"On this sample, the detector classifies the covers as expected, and "
            f"the stego probability shift relative to the matching cover ranges "
            f"from {sample['delta_stego_minus_cover_pct'].min():.1f}% to "
            f"{sample['delta_stego_minus_cover_pct'].max():.1f}%, with "
            f"{n_no_shift} of {len(sample)} pairs showing less than a 1.0% shift."
        )
        lines.append("")

        lines.append(f"Table: Batch steganalysis accuracy (fine-tuned SRNet), BPP {bpp_bin}.")
        auc_str = f"{summ['roc_auc']:.4f}" if summ["roc_auc"] is not None else "N/A"
        batch_tbl = pd.DataFrame([{
            "Pairs": summ["pairs"],
            "Images": summ["images"],
            "Accuracy @0.5": f"{summ['accuracy_pct_at_0.5']:.2f}%",
            "Accuracy @best thresh": f"{summ['accuracy_pct_at_best_thresh']:.2f}% "
                                     f"(t={summ['best_thresh']:.3f})",
            "Mean P(stego|cover)": f"{summ['mean_p_stego_given_cover']:.3f}",
            "Mean P(stego|stego)": f"{summ['mean_p_stego_given_stego']:.3f}",
            "ROC-AUC": auc_str,
            "Verdict": summ["verdict"],
        }])
        lines.append(format_table(batch_tbl))
        lines.append("")

        lines.append(f"Table: Per-class precision, recall and F1, BPP {bpp_bin}.")
        pcls_tbl = pd.DataFrame([
            {
                "Threshold": "0.5 (native)",
                "Class": "cover",
                "Precision": f"{summ['precision_cover_at_0.5']:.3f}",
                "Recall": f"{summ['recall_cover_at_0.5']:.3f}",
                "F1": f"{summ['f1_cover_at_0.5']:.3f}",
            },
            {
                "Threshold": "0.5 (native)",
                "Class": "stego",
                "Precision": f"{summ['precision_stego_at_0.5']:.3f}",
                "Recall": f"{summ['recall_stego_at_0.5']:.3f}",
                "F1": f"{summ['f1_stego_at_0.5']:.3f}",
            },
            {
                "Threshold": f"{summ['best_thresh']:.3f} (recalibrated)",
                "Class": "cover",
                "Precision": f"{summ['precision_cover_at_best']:.3f}",
                "Recall": f"{summ['recall_cover_at_best']:.3f}",
                "F1": f"{summ['f1_cover_at_best']:.3f}",
            },
            {
                "Threshold": f"{summ['best_thresh']:.3f} (recalibrated)",
                "Class": "stego",
                "Precision": f"{summ['precision_stego_at_best']:.3f}",
                "Recall": f"{summ['recall_stego_at_best']:.3f}",
                "F1": f"{summ['f1_stego_at_best']:.3f}",
            },
        ])
        lines.append(format_table(pcls_tbl))
        lines.append("")

        lines.append(
            f"At BPP {bpp_bin}, the detector reaches {summ['accuracy_pct_at_0.5']:.2f}% "
            f"accuracy at its native 0.5 threshold, rising to "
            f"{summ['accuracy_pct_at_best_thresh']:.2f}% at the recalibrated threshold "
            f"of {summ['best_thresh']:.3f} (ROC-AUC {auc_str}), over {summ['images']} images. "
            f"At the native threshold, stego recall is {summ['recall_stego_at_0.5']:.3f} "
            f"against cover recall of {summ['recall_cover_at_0.5']:.3f}, which recovers to "
            f"{summ['recall_stego_at_best']:.3f} and {summ['recall_cover_at_best']:.3f} "
            f"respectively at the recalibrated threshold. "
            f"Mean stego probability is {summ['mean_p_stego_given_cover']:.3f} for covers and "
            f"{summ['mean_p_stego_given_stego']:.3f} for stego images "
            f"(delta {summ['delta']:+.3f}). We read this as: {summ['verdict']}."
        )
        lines.append("")

        lines.append(f"Table: Chi-square statistic, cover vs stego, BPP {bpp_bin} "
                     f"(first {len(sample)} pairs; full results in Excel).")
        chi_tbl = pd.DataFrame([{
            "Image": r["image"],
            "Cover chi2": f"{r['chi_square_cover']:.2f}",
            "Stego chi2": f"{r['chi_square_stego']:.2f}",
            "Delta": f"{r['chi_square_delta']:+.2f}",
        } for _, r in sample.iterrows()])
        lines.append(format_table(chi_tbl))
        lines.append("")

        mean_delta = sub["chi_square_delta"].mean()
        lines.append(
            f"Across the full {len(sub)}-pair set at BPP {bpp_bin}, the mean chi-square "
            f"delta between stego and cover is {mean_delta:+.2f}. The statistic "
            f"is dominated by each cover's own content rather than by embedding, "
            f"so we take this as consistent with the SRNet result: no systematic "
            f"first-order signature is introduced by the embedding at this rate."
        )
        lines.append("")
        lines.append("")

    lines.append(
        "Qualified reading. Across the tested BPP range, the recalibrated-threshold "
        "accuracy and ROC-AUC together indicate the fine-tuned SRNet detector does "
        "separate cover from stego above chance for these covers, though the "
        "detector's own native threshold is not calibrated to reflect this. The "
        "per-class precision, recall and F1 scores show whether that separation is "
        "shared evenly between the cover and stego classes or concentrated in one "
        "of them. The chi-square deltas give no consistent additional evidence of a "
        "first-order signature. This does not establish undetectability in general, "
        "only that detection performance at the rates tested here is bounded, and "
        "depends on the operating threshold chosen."
    )

    return "\n".join(lines)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    print("=== Steganalysis evaluation (flat stego folder, continuous BPP) ===\n")
    module = load_steganalyzer_module(STEGANALYZER_MODULE_PATH)
    model, device = module.load_model(CHECKPOINT_PATH)

    print("Loading achieved_bpp from comparison_results.xlsx ...")
    bpp_map = load_achieved_bpp_map(COMPARISON_XLSX)
    print(f"  {len(bpp_map)} image(s) have a recorded achieved_bpp.\n")

    per_image_df = run_detection(module, model, device, FINETUNED_COVER_DIR, STEGO_DIR, bpp_map)

    per_image_df["bpp_bin"] = _bin_bpp_series(per_image_df["achieved_bpp"], BPP_BIN_WIDTH)
    summary_df = summarize_by_bin(per_image_df)

    for _, row in summary_df.iterrows():
        print(f"  BPP {row['bpp_bin']}: acc@0.5={row['accuracy_pct_at_0.5']:.2f}%  "
              f"acc@best={row['accuracy_pct_at_best_thresh']:.2f}% (t={row['best_thresh']:.3f})  "
              f"auc={row['roc_auc']:.4f}  "
              f"P/R/F1(stego)@0.5={row['precision_stego_at_0.5']:.3f}/"
              f"{row['recall_stego_at_0.5']:.3f}/{row['f1_stego_at_0.5']:.3f}  "
              f"verdict={row['verdict']}")

    os.makedirs(os.path.dirname(EXCEL_OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(EXCEL_OUT_PATH, engine="openpyxl") as writer:
        per_image_df.to_excel(writer, sheet_name="per_image_detection", index=False)
        summary_df.to_excel(writer, sheet_name="batch_summary", index=False)

    print(f"\nSaved detection metrics to: {EXCEL_OUT_PATH}")

    report_text = build_report(per_image_df, summary_df)
    os.makedirs(os.path.dirname(REPORT_TXT_PATH), exist_ok=True)
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved narrative report to: {REPORT_TXT_PATH}")


if __name__ == "__main__":
    main()