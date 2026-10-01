"""Figure 3: five author-selected pairs from the 400-pair corpus (original 256x256 files)."""
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
cs = pd.read_csv("cover_stats.csv"); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
cmp_ = pd.read_excel(C.RES + "/comparison_results.xlsx", sheet_name="per_image_results").set_index("image")
picks = [("299.png", "Driftwood grain"), ("114.png", "Sand dunes"), ("096.png", "Ivy leaves"), ("328.png", "Spider web"), ("216.png", "Snake skin")]
titles = ["baseline cover", "fine-tuned cover", "fine-tuned stego", "luminance difference x 8"]
fig, axes = plt.subplots(5, 4, figsize=(5.6, 7.0))
for r, (n, lab) in enumerate(picks):
    bc = cv2.imread(C.BASE_DIR + "/" + n); fc = cv2.imread(C.FT_DIR + "/" + n); st = cv2.imread(C.STEGO_FT_DIR + "/" + n)
    diff = np.abs(C.read_y(C.STEGO_FT_DIR + "/" + n).astype(int) - C.read_y(C.FT_DIR + "/" + n).astype(int))
    for c, im in enumerate([bc[..., ::-1], fc[..., ::-1], st[..., ::-1], np.clip(diff * 8, 0, 255)]):
        ax = axes[r, c]; ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(titles[c], fontsize=7)
    axes[r, 0].set_ylabel(f"{lab}\n(image {n[:3]})\ne: {b.loc[n,'mean_eligible']:.1f} to {f.loc[n,'mean_eligible']:.1f}\n{cmp_.loc[n,'achieved_bpp']:.2f} bpp\nPSNR {cmp_.loc[n,'finetuned_psnr']:.1f} dB\nSSIM {cmp_.loc[n,'finetuned_ssim']:.3f}",
                          fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
    print(n, lab, round(b.loc[n,'mean_eligible'],2), round(f.loc[n,'mean_eligible'],2), round(cmp_.loc[n,'achieved_bpp'],3), round(cmp_.loc[n,'finetuned_psnr'],2), round(cmp_.loc[n,'finetuned_ssim'],4))
fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.2, right=0.995, top=0.965, bottom=0.005)
fig.savefig("../figures/fig3_example_pairs.pdf", dpi=300); fig.savefig("fig3_selected.png", dpi=120)
