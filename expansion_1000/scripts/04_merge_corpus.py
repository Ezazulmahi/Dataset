r"""Expansion, step 4: assemble the 1,000-pair corpus and the merged spreadsheets.

corpus1000\ has the layout of F:\Results: baseline\, finetuned\, stego_finetuned\ with images 000 to 999
(000 to 399 copied from F:\Results, 400 to 999 from new600\), and copies of the authors' spreadsheets
with the 600 new rows appended and the summary sheets recomputed with the authors' own summary
functions. F:\Results itself is not modified. Every per-image sheet gets a column corpus_part."""
import os, shutil, importlib.util, io, contextlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); NEW = os.path.join(HERE, "new600"); R6 = os.path.join(HERE, "results_new600")
C1000 = os.path.join(HERE, "corpus1000"); A1000 = os.path.join(HERE, "analysis1000"); RES = r"F:\Results"
OLD_AN = r"F:\Publication\scirep_submission\analysis"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m
def part(df, label):
    df = df.copy(); df["corpus_part"] = label; return df
OLD, EXP = "original 400", "expansion 600"

if __name__ == "__main__":
    # ---------------------------------------------------------------- images
    for sub, src_old, src_new in (("baseline", os.path.join(RES, "baseline"), os.path.join(NEW, "baseline")),
                                  ("finetuned", os.path.join(RES, "finetuned"), os.path.join(NEW, "finetuned")),
                                  ("stego_finetuned", os.path.join(RES, "stego_finetuned"), os.path.join(NEW, "stego_finetuned"))):
        d = os.path.join(C1000, sub); os.makedirs(d, exist_ok=True)
        for i in range(1000):
            n = f"{i:03d}.png"; dst = os.path.join(d, n)
            if not os.path.exists(dst): shutil.copy2(os.path.join(src_old if i < 400 else src_new, n), dst)
        assert len(os.listdir(d)) == 1000, sub
    d = os.path.join(A1000, "stego_base"); os.makedirs(d, exist_ok=True)      # baseline stego: re-embedding run for 0..399, expansion run for 400..999
    for i in range(1000):
        n = f"{i:03d}.png"; dst = os.path.join(d, n)
        if not os.path.exists(dst): shutil.copy2(os.path.join(os.path.join(OLD_AN, "stego_base") if i < 400 else os.path.join(NEW, "stego_baseline"), n), dst)
    print("images assembled", flush=True)

    # ---------------------------------------------------------------- comparison_results.xlsx
    CMP = load("comparison_authors", os.path.join(RES, "comparison.py"))
    xo = pd.ExcelFile(os.path.join(RES, "comparison_results.xlsx")); xn = pd.ExcelFile(os.path.join(R6, "comparison_results_400_999.xlsx"))
    per = pd.concat([part(xo.parse("per_image_results").sort_values("image"), OLD), part(xn.parse("per_image_results"), EXP)], ignore_index=True)
    summary = CMP.summarize(per)
    clip = pd.concat([part(xo.parse("clip_per_image"), OLD), part(xn.parse("clip_per_image"), EXP)], ignore_index=True)
    clip2 = pd.concat([part(xo.parse("clip_per_image_finetuned_vs_stego"), OLD), part(xn.parse("clip_per_image_finetuned_vs_stego"), EXP)], ignore_index=True)
    fid_bf = CMP.compute_fid(os.path.join(C1000, "baseline"), os.path.join(C1000, "finetuned"))
    fid_fs = CMP.compute_fid(os.path.join(C1000, "finetuned"), os.path.join(C1000, "stego_finetuned"))
    def fc(name, fid, c):
        m, s, lo, hi = CMP.mean_std_ci(c["clip_cosine_similarity"])
        return pd.DataFrame([{name: fid, "CLIP_cosine_mean": m, "CLIP_cosine_std": s, "CLIP_cosine_ci_low": lo, "CLIP_cosine_ci_high": hi, "n_images_compared": len(c)}])
    parts = pd.DataFrame([{"corpus_part": OLD, "FID_baseline_vs_finetuned": float(xo.parse("FID_CLIP").iloc[0, 0]), "FID_finetuned_vs_stego": float(xo.parse("FID_CLIP_finetuned_vs_stego").iloc[0, 0])},
                          {"corpus_part": EXP, "FID_baseline_vs_finetuned": float(xn.parse("FID_CLIP").iloc[0, 0]), "FID_finetuned_vs_stego": float(xn.parse("FID_CLIP_finetuned_vs_stego").iloc[0, 0])},
                          {"corpus_part": "all 1000", "FID_baseline_vs_finetuned": fid_bf, "FID_finetuned_vs_stego": fid_fs}])
    with pd.ExcelWriter(os.path.join(C1000, "comparison_results.xlsx"), engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_image_results", index=False); summary.to_excel(w, sheet_name="summary_stats", index=False)
        fc("FID_baseline_vs_finetuned", fid_bf, clip).to_excel(w, sheet_name="FID_CLIP", index=False); clip.to_excel(w, sheet_name="clip_per_image", index=False)
        fc("FID_finetuned_vs_stego", fid_fs, clip2).to_excel(w, sheet_name="FID_CLIP_finetuned_vs_stego", index=False)
        clip2.to_excel(w, sheet_name="clip_per_image_finetuned_vs_stego", index=False)
        parts.to_excel(w, sheet_name="FID_by_corpus_part", index=False)
        xn.parse("recovery_and_runtime").to_excel(w, sheet_name="recovery_runtime_400_999", index=False)
    print("comparison_results.xlsx merged:", len(per), "rows"); print(parts.round(3).to_string(index=False), flush=True)

    # ---------------------------------------------------------------- robustness_results.xlsx
    ROB = load("Robustness_authors", os.path.join(RES, "Robustness.py"))
    xo = pd.ExcelFile(os.path.join(RES, "robustness_results.xlsx")); xn = pd.ExcelFile(os.path.join(R6, "robustness_results_400_999.xlsx"))
    per = pd.concat([part(xo.parse("per_image_results"), OLD), part(xn.parse("per_image_results"), EXP)], ignore_index=True)
    with pd.ExcelWriter(os.path.join(C1000, "robustness_results.xlsx"), engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_image_results", index=False); ROB.summarize(per).to_excel(w, sheet_name="summary_curve", index=False)
        xo.parse("literature_comparison").to_excel(w, sheet_name="literature_comparison", index=False)
    print("robustness_results.xlsx merged:", len(per), "rows", flush=True)

    # ---------------------------------------------------------------- robustness_customQ_adv_ecc_align.xlsx
    NR_NAME = "robustness_customQ_adv_ecc_align"
    xo = pd.ExcelFile(os.path.join(RES, NR_NAME + ".xlsx")); xn = pd.ExcelFile(os.path.join(R6, NR_NAME + "_400_999.xlsx"))
    per = pd.concat([part(xo.parse("per_image_results"), OLD), part(xn.parse("per_image_results"), EXP)], ignore_index=True)
    rows = []
    for (tt, cond), g in per.groupby(["test_type", "condition"]):       # same statistics as main() of newrobust.py
        m, s, lo, hi = CMP.mean_std_ci(g["ber"]); rows.append({"test_type": tt, "condition": cond, "n_images": len(g), "exact_match_rate": g["exact_match"].mean(),
                                                                "ber_mean": m, "ber_std": s, "ber_ci_low": lo, "ber_ci_high": hi})
    with pd.ExcelWriter(os.path.join(C1000, NR_NAME + ".xlsx"), engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_image_results", index=False); pd.DataFrame(rows).sort_values(["test_type", "condition"]).to_excel(w, sheet_name="summary_stats", index=False)
    print(NR_NAME + ".xlsx merged:", len(per), "rows", flush=True)

    # ---------------------------------------------------------------- steganalysis_results.xlsx (the authors' earlier detector and the chi-square statistic)
    STG = load("Steganalysis_authors", os.path.join(RES, "Steganalysis.py"))
    xo = pd.ExcelFile(os.path.join(RES, "steganalysis_results.xlsx")); xn = pd.ExcelFile(os.path.join(R6, "steganalysis_results_400_999.xlsx"))
    per = pd.concat([part(xo.parse("per_image_detection"), OLD), part(xn.parse("per_image_detection"), EXP)], ignore_index=True)
    with pd.ExcelWriter(os.path.join(C1000, "steganalysis_results.xlsx"), engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_image_detection", index=False); STG.summarize_by_bin(per).to_excel(w, sheet_name="batch_summary", index=False)
    print("steganalysis_results.xlsx merged:", len(per), "rows", flush=True)

    # ---------------------------------------------------------------- embed_rerun.csv for the analysis scripts (recovery, verify-fix passes, runtime)
    old = pd.read_csv(os.path.join(OLD_AN, "embed_rerun.csv")); e = pd.read_csv(os.path.join(HERE, "embed_400_999.csv")); rows = []
    for _, r in e.iterrows():
        for tag, pre in (("ft", "finetuned"), ("base", "baseline")):
            rows.append({"image": r["image"], "set": tag, "K": r["K"], "n_bits": r["n_bits"], "bpp": r["achieved_bpp"], "embed_s": r[pre + "_embed_s"], "fix_passes": r[pre + "_fix_passes"],
                         "error": r.get(pre + "_error", np.nan), "extract_s": r[pre + "_extract_s"], "exact": r[pre + "_exact"], "ber": 0.0 if r[pre + "_exact"] else np.nan,
                         "psnr_luma": r[pre + "_psnr"], "ssim_luma": r[pre + "_ssim"], "psnr_color": r[pre + "_psnr_color"], "ssim_color": r[pre + "_ssim_color"], "changed_px_frac": r[pre + "_changed_px_frac"]})
    pd.concat([part(old, OLD + " (re-embedding run)"), part(pd.DataFrame(rows), EXP)], ignore_index=True).to_csv(os.path.join(A1000, "embed_rerun.csv"), index=False)
    print("DONE", flush=True)
