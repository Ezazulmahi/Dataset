"""
04_figures.py  -  figures of the manuscript remade with the v5 (normal mode) stego images.

  fig3_example_pairs_v5.pdf   Figure 3: the SAME five author-selected pairs as the
                              submitted paper (images 299, 114, 096, 328, 216),
                              same layout and row labels, v5 stego images.
  figS1_corpus_examples_v5.pdf  Supplementary Fig. S1: the same rule-selected pairs
                              (min, quartiles, max of fine-tuned e-bar), v5 stego.
  fig4ab_distortion_v5.pdf    Figure 4a,b: luminance PSNR and SSIM against payload
                              (points per image, means in 0.05 bpp bins), baseline
                              and fine-tuned covers, n = 1,000 pairs.  Panel c (ROC)
                              is left for the steganalysis rerun.
Usage: python 04_figures.py
"""
import os, sys
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = r"F:\scirep results\09_expansion_1000\corpus1000"
COVSTATS = r"F:\scirep results\09_expansion_1000\analysis1000\cover_stats.csv"
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"

cs = pd.read_csv(COVSTATS); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
d = pd.read_csv(os.path.join(HERE, "results", "normal_per_image.csv"))
q = d[d.condition == "none"].set_index(["image", "covers"])


def read_y(path):
    return cv2.split(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YCrCb))[0]


def panel(picks, labels, out_pdf, out_png):
    titles = ["baseline cover", "fine-tuned cover", "fine-tuned stego", "luminance difference x 8"]
    fig, axes = plt.subplots(5, 4, figsize=(5.6, 7.0))
    for r, n in enumerate(picks):
        st_path = os.path.join(HERE, "stego_finetuned", n)
        bc = cv2.imread(os.path.join(CORP, "baseline", n)); fc = cv2.imread(os.path.join(CORP, "finetuned", n)); st = cv2.imread(st_path)
        diff = np.abs(read_y(st_path).astype(int) - read_y(os.path.join(CORP, "finetuned", n)).astype(int))
        for c, im in enumerate([bc[..., ::-1], fc[..., ::-1], st[..., ::-1], np.clip(diff * 8, 0, 255)]):
            ax = axes[r, c]; ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0: ax.set_title(titles[c], fontsize=7)
        row = q.loc[(n, "finetuned")]
        axes[r, 0].set_ylabel(f"{labels[r]}\n(image {n[:3]})\ne: {b.loc[n,'mean_eligible']:.1f} to {f.loc[n,'mean_eligible']:.1f}\n"
                              f"{row.bpp:.2f} bpp\nPSNR {row.psnr_y:.1f} dB\nSSIM {row.ssim_y:.3f}",
                              fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
        print(out_pdf, n, labels[r], f"{row.bpp:.3f} bpp  PSNR {row.psnr_y:.2f}  SSIM {row.ssim_y:.4f}")
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.2, right=0.995, top=0.965, bottom=0.005)
    fig.savefig(os.path.join(FIG, out_pdf), dpi=300); fig.savefig(os.path.join(FIG, out_png), dpi=120); plt.close(fig)


# Figure 3: the paper's five pairs
panel(["299.png", "114.png", "096.png", "328.png", "216.png"],
      ["Driftwood grain", "Sand dunes", "Ivy leaves", "Spider web", "Snake skin"],
      "fig3_example_pairs_v5.pdf", "fig3_example_pairs_v5.png")

# Supplementary Fig. S1: rule-selected (min, 25th, 50th, 75th percentile, max of fine-tuned e-bar)
order = f.mean_eligible.sort_values()
picks = [order.index[int(round(p * (len(order) - 1)))] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
panel(picks, ["minimum", "25th percentile", "median", "75th percentile", "maximum"],
      "figS1_corpus_examples_v5.pdf", "figS1_corpus_examples_v5.png")

# Figure 4a,b
fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
edges = np.arange(0.30, 0.6001, 0.05)
for ax, col, lab in ((axes[0], "psnr_y", "luminance PSNR (dB)"), (axes[1], "ssim_y", "luminance SSIM")):
    for cov, colr, name in (("baseline", BLUE, "baseline covers"), ("finetuned", ORANGE, "fine-tuned covers")):
        g = d[(d.condition == "none") & (d.covers == cov)]
        ax.scatter(g.bpp, g[col], s=2, color=colr, alpha=0.25, lw=0)
        mids = (edges[:-1] + edges[1:]) / 2
        means = [g[(g.bpp >= lo) & (g.bpp < hi)][col].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mids, means, color=colr, marker="o", ms=2.5, lw=1.2, label=name)
    ax.set_xlabel("embedded payload (bits per pixel)"); ax.set_ylabel(lab); ax.grid(True, color="#d9dde1", lw=0.4)
axes[0].legend(fontsize=6); axes[0].set_title("a", loc="left", fontweight="bold"); axes[1].set_title("b", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4ab_distortion_v5.pdf"), dpi=300); fig.savefig(os.path.join(FIG, "fig4ab_distortion_v5.png"), dpi=130)
print("n pairs in Fig 4a,b:", d[d.condition == "none"].image.nunique())
