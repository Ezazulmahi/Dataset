r"""Expansion, step 3: write comparison_results_400_999.xlsx for the 600 new pairs in the layout of the
authors' comparison_results.xlsx, using the authors' own summarize(), compute_fid() and
compute_clip_similarity() from F:\Results\comparison.py. The copy of fid.py is run afterwards and adds
the fine-tuned versus stego sheets, exactly as it did for the first 400 pairs."""
import os, io, contextlib, importlib.util
import pandas as pd
import clip_compat  # noqa: F401  (transformers 5 return type, see clip_compat.py)
HERE = os.path.dirname(os.path.abspath(__file__)); NEW = os.path.join(HERE, "new600"); OUT = os.path.join(HERE, "results_new600")

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("comparison_authors", r"F:\Results\comparison.py")
    CMP = importlib.util.module_from_spec(spec); spec.loader.exec_module(CMP)
    e = pd.read_csv(os.path.join(HERE, "embed_400_999.csv")).sort_values("image")
    cols = ["image", "target_bpp", "achieved_bpp", "K", "n_bits", "baseline_psnr", "baseline_ssim", "finetuned_psnr", "finetuned_ssim", "delta_psnr", "delta_ssim"]
    df = e[cols].reset_index(drop=True); summary = CMP.summarize(df)
    fid = CMP.compute_fid(os.path.join(NEW, "baseline"), os.path.join(NEW, "finetuned"))
    clip = CMP.compute_clip_similarity(os.path.join(NEW, "baseline"), os.path.join(NEW, "finetuned"))
    m, s, lo, hi = CMP.mean_std_ci(clip["clip_cosine_similarity"])
    fc = pd.DataFrame([{"FID_baseline_vs_finetuned": fid, "CLIP_cosine_mean": m, "CLIP_cosine_std": s, "CLIP_cosine_ci_low": lo, "CLIP_cosine_ci_high": hi, "n_images_compared": len(clip)}])
    extra = e[["image"] + [c for c in e.columns if c not in cols]]
    with pd.ExcelWriter(os.path.join(OUT, "comparison_results_400_999.xlsx"), engine="openpyxl") as w:
        df.to_excel(w, sheet_name="per_image_results", index=False); summary.to_excel(w, sheet_name="summary_stats", index=False)
        fc.to_excel(w, sheet_name="FID_CLIP", index=False); clip.to_excel(w, sheet_name="clip_per_image", index=False)
        extra.to_excel(w, sheet_name="recovery_and_runtime", index=False)
    print(fc.T.to_string()); print("exact recovery: fine-tuned %d, baseline %d of %d" % (e.finetuned_exact.sum(), e.baseline_exact.sum(), len(e)))
