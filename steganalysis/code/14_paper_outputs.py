r"""Step 14 - the manuscript's Fig. 4 (with the ROC panels) and Table 3 from the saved detector scores.

Reads the score files of 03_test.py (SRNet-KV), 11_srm_classify.py (SRM + FLD) and 13_jpeg_classify.py
(DCTR + FLD); never retrains anything. Test pairs: the paper's 1,000 fine-tuned covers against
    DEDS        the 1,000 DEDS-v5 fine-tuned stego of F:\scirep submission 2.0 (set v5_paper_ft)
    S-UNIWARD, WOW, HILL, MiPOD   that method's 0.4 bpp stego of the same covers (set paper_ft)
    J-UNIWARD, J-MiPOD            the covers compressed at QF 75 and their 0.4 bpnzac stego (DCTR table rows only)
each scored by the detector trained on that method (matched protocol).
Output in paper_outputs\:
    fig4_distortion_detection.pdf/.png   full Fig. 4: a, b as in 04_figures.py of the submission, c ROC SRNet-KV,
                                         d ROC SRM+FLD (final copies also in figures\ when all 12 rows exist)
    fig4cd_roc.pdf/.png                  the two ROC panels alone
    roc_points.csv                       detector, method, fpr, tpr (every curve in the figure)
    table3_detectability.csv             all rows with P_E at the validation threshold, P_E min, AUC and 95% CI
    table3_rows.tex                      the rows in the format of Table 3 of main.tex
Missing results are skipped and listed, so this can be run at any time; run it again when everything is done.
    python 14_paper_outputs.py
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = C.ROOT / "paper_outputs"; R = C.RESULTS
EVAL = Path(r"F:\scirep submission 2.0\4_evaluation")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"                       # as Fig. 4a,b of the submission
NAMES = {"deds_v4": "DEDS", "suniward": "S-UNIWARD", "wow": "WOW", "hill": "HILL", "mipod": "MiPOD",
         "juniward": "J-UNIWARD", "jmipod": "J-MiPOD"}
COLORS = {"deds_v4": ORANGE, "suniward": BLUE, "wow": "#2e9e6b", "hill": "#8a5cc7", "mipod": "#7d838a"}
SPATIAL = ["deds_v4", "suniward", "wow", "hill", "mipod"]
missing = []

def own_set(m):
    return "v5_paper_ft" if m == "deds_v4" else "paper_ft"

def entry(detector, m, scores_csv, score_col, summary_csv, sel, training):
    """One table row + its ROC from a scores file and the matching summary row."""
    if not (Path(scores_csv).exists() and Path(summary_csv).exists()):
        missing.append(f"{detector} / {NAMES[m]}: {scores_csv}"); return None
    sm = pd.read_csv(summary_csv); sm = sm[np.logical_and.reduce([sm[k] == v for k, v in sel.items()])]
    if sm.empty:
        missing.append(f"{detector} / {NAMES[m]}: no row {sel} in {summary_csv}"); return None
    s = sm.iloc[0]; d = pd.read_csv(scores_csv)
    fpr, tpr, _ = roc_curve(d.label, d[score_col])
    return {"embedder": NAMES[m], "key": m, "detector": detector, "training": training, "pairs": int(s.pairs),
            "pe_at_val_thr": 1 - float(s.acc_at_val_thr), "pe_min": float(s.pe_min), "auc": float(s.auc),
            "auc_ci_low": float(s.auc_ci_low), "auc_ci_high": float(s.auc_ci_high), "fpr": fpr, "tpr": tpr}

def collect():
    rows = []
    for m in SPATIAL:
        run = f"srnet_kv_matched-{m}_tta"; st = own_set(m)
        rows.append(entry("SRNet-KV", m, R / run / f"{st}_scores.csv", "margin", R / run / "summary.csv", {"set": st}, NAMES[m]))
    for m in SPATIAL:
        st = own_set(m)
        rows.append(entry("SRM+FLD", m, R / "srm_fld" / f"{m}_paper_ft_scores.csv", "score", R / "srm_fld" / "summary.csv",
                          {"method": m, "set": "paper_ft"}, NAMES[m]))
    for m in ("juniward", "jmipod"):
        rows.append(entry("DCTR+FLD", m, R / "dctr_fld" / f"{m}_paper_ft_scores.csv", "score", R / "dctr_fld" / "summary.csv",
                          {"method": m, "set": "paper_ft"}, NAMES[m]))
    return [r for r in rows if r is not None]          # WSERNet dropped 19 Oct 2026 (mismatched detector)

def roc_panel(ax, rows, detector, letter):
    ax.plot([0, 1], [0, 1], color="#b8bec4", lw=0.6, ls=":")
    for r in [r for r in rows if r["detector"] == detector]:
        ax.plot(r["fpr"], r["tpr"], color=COLORS[r["key"]], lw=1.4 if r["key"] == "deds_v4" else 1.0,
                label=f"{r['embedder']} (AUC {r['auc']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("detection rate"); ax.grid(True, color="#d9dde1", lw=0.4)
    ax.set_title(letter, loc="left", fontweight="bold"); ax.set_title(detector, fontsize=7)
    ax.legend(fontsize=5.5, loc="lower right", handlelength=1.6)

def distortion_panels(axes):
    """Fig. 4a,b exactly as 04_figures.py of the submission draws them."""
    d = pd.read_csv(EVAL / "results" / "normal_per_image.csv"); edges = np.arange(0.30, 0.6001, 0.05)
    for ax, col, lab in ((axes[0], "psnr_y", "luminance PSNR (dB)"), (axes[1], "ssim_y", "luminance SSIM")):
        for cov, colr, name in (("baseline", BLUE, "baseline covers"), ("finetuned", ORANGE, "fine-tuned covers")):
            g = d[(d.condition == "none") & (d.covers == cov)]
            ax.scatter(g.bpp, g[col], s=2, color=colr, alpha=0.25, lw=0)
            mids = (edges[:-1] + edges[1:]) / 2
            means = [g[(g.bpp >= lo) & (g.bpp < hi)][col].mean() for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(mids, means, color=colr, marker="o", ms=2.5, lw=1.2, label=name)
        ax.set_xlabel("embedded payload (bits per pixel)"); ax.set_ylabel(lab); ax.grid(True, color="#d9dde1", lw=0.4)
    axes[0].legend(fontsize=6); axes[0].set_title("a", loc="left", fontweight="bold"); axes[1].set_title("b", loc="left", fontweight="bold")

def tex_rows(t):
    lines = []
    for det in ("SRNet-KV", "SRM+FLD", "DCTR+FLD"):
        g = t[t.detector == det]
        if g.empty:
            continue
        if lines:
            lines.append("\\midrule")
        for _, r in g.iterrows():
            cov = "fine-tuned (JPEG QF 75)" if det == "DCTR+FLD" else "fine-tuned"
            lines.append(f"{r.embedder} & {cov} & {det} ({r.training}) & {r.pe_at_val_thr:.3f} & "
                         f"{r.auc:.3f} ({r.auc_ci_low:.3f} to {r.auc_ci_high:.3f}) \\\\")
    return "\n".join(lines) + "\n"

def main():
    OUT.mkdir(exist_ok=True); rows = collect()
    if rows:
        t = pd.DataFrame([{k: v for k, v in r.items() if k not in ("fpr", "tpr", "key")} for r in rows])
        t.to_csv(OUT / "table3_detectability.csv", index=False); (OUT / "table3_rows.tex").write_text(tex_rows(t))
        pts = []
        for r in rows:
            k = np.unique(np.linspace(0, len(r["fpr"]) - 1, min(len(r["fpr"]), 400)).astype(int))   # thinned, shape kept
            pts.append(pd.DataFrame({"detector": r["detector"], "method": r["embedder"], "fpr": r["fpr"][k], "tpr": r["tpr"][k]}))
        pd.concat(pts).to_csv(OUT / "roc_points.csv", index=False)
        print(t.drop(columns=["training"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if any(r["detector"] in ("SRNet-KV", "SRM+FLD") for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
        roc_panel(axes[0], rows, "SRNet-KV", "c"); roc_panel(axes[1], rows, "SRM+FLD", "d")
        fig.tight_layout(); fig.savefig(OUT / "fig4cd_roc.pdf", dpi=300); fig.savefig(OUT / "fig4cd_roc.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.7))
        distortion_panels(axes[0]); roc_panel(axes[1, 0], rows, "SRNet-KV", "c"); roc_panel(axes[1, 1], rows, "SRM+FLD", "d")
        fig.tight_layout(); fig.savefig(OUT / "fig4_distortion_detection.pdf", dpi=300)
        fig.savefig(OUT / "fig4_distortion_detection.png", dpi=130); plt.close(fig)
        print(f"\nfigures -> {OUT}")
    if missing:
        print("\nnot available yet (skipped):\n  " + "\n  ".join(missing))
    print(f"\n{len(rows)} of 12 rows written -> {OUT}")
    if (not missing or "--copy-partial" in sys.argv) and (OUT / "fig4_distortion_detection.pdf").exists():
        save_figures()
    else:
        print("figures NOT saved to figures\\ yet (results incomplete; --copy-partial forces it)")

def save_figures():
    """Final figures in this project's figures\\ folder (the submission folder is not touched)."""
    import shutil
    FIGS = C.ROOT / "figures"; FIGS.mkdir(exist_ok=True)
    for f in ("fig4_distortion_detection.pdf", "fig4_distortion_detection.png", "fig4cd_roc.pdf", "fig4cd_roc.png"):
        shutil.copy2(OUT / f, FIGS / f); print(f"saved {FIGS / f}")

if __name__ == "__main__":
    main()
