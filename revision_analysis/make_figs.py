"""Figures 2-4 from the raw evaluation data."""
import numpy as np, pandas as pd, cv2, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import common as C
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BLUE, ORANGE, AQUA, VIOLET, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#30363d", "#d9dde1"
cs = pd.read_csv("cover_stats.csv"); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
cmp_ = pd.read_excel(C.RES + r"\comparison_results.xlsx", sheet_name="per_image_results").set_index("image")

# ------------------------------------------------------------------ Fig 2: examples chosen by a stated rule
order = f.mean_eligible.sort_values()
picks = [order.index[int(round(q * (len(order) - 1)))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
fig, axes = plt.subplots(5, 4, figsize=(5.6, 7.0))
titles = ["baseline cover", "fine-tuned cover", "fine-tuned stego", r"$|Y_{\mathrm{stego}}-Y_{\mathrm{cover}}|\times 8$"]
for r, n in enumerate(picks):
    bc = cv2.imread(C.BASE_DIR + "\\" + n); fc = cv2.imread(C.FT_DIR + "\\" + n); st = cv2.imread(C.STEGO_FT_DIR + "\\" + n)
    diff = np.abs(C.read_y(C.STEGO_FT_DIR + "\\" + n).astype(int) - C.read_y(C.FT_DIR + "\\" + n).astype(int))
    for c, im in enumerate([bc[..., ::-1], fc[..., ::-1], st[..., ::-1], np.clip(diff * 8, 0, 255)]):
        ax = axes[r, c]; ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(titles[c], fontsize=7)
    axes[r, 0].set_ylabel(f"image {n[:3]}\n" + r"$\bar e$" + f" {b.loc[n,'mean_eligible']:.1f} → {f.loc[n,'mean_eligible']:.1f}\n{cmp_.loc[n,'achieved_bpp']:.2f} bpp, "
                          f"{cmp_.loc[n,'finetuned_psnr']:.1f} dB", fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.2, right=0.995, top=0.965, bottom=0.005)
fig.savefig("../figures/figS1_corpus_examples.pdf", dpi=300); fig.savefig("fig2_examples.png", dpi=130)
print("fig2 picks", picks)

# ------------------------------------------------------------------ Fig 3: DCT statistics and spectra
m = np.load("elig_maps.npz")
def radial_psd(y):
    y = y.astype(float); y = (y - y.mean()) * np.outer(np.hanning(256), np.hanning(256))
    P = np.abs(np.fft.fftshift(np.fft.fft2(y))) ** 2
    yy, xx = np.indices(P.shape); r = np.hypot(yy - 128, xx - 128).astype(int)
    return (np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[1:128]
if not os.path.exists("psd.npz"):
    nat = sorted(glob.glob(r"F:\Thesis\finetune_extra\covers\*.png"))
    pb = np.array([radial_psd(C.read_y(C.BASE_DIR + "\\" + f"{i:03d}.png")) for i in range(400)])
    pf = np.array([radial_psd(C.read_y(C.FT_DIR + "\\" + f"{i:03d}.png")) for i in range(400)])
    pn = np.array([radial_psd(C.read_y(p)) for p in nat])
    np.savez("psd.npz", pb=pb, pf=pf, pn=pn)
p = np.load("psd.npz")
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.48, wspace=0.42)
for k, (key, ttl) in enumerate((("base", "baseline covers"), ("ft", "fine-tuned covers"))):
    ax = fig.add_subplot(gs[0, k]); im = ax.imshow(m[key], cmap="Blues", vmin=0, vmax=0.85)
    for u in range(8):
        for v in range(8):
            if C.STABLE[u, v]:
                ax.text(v, u, f"{m[key][u, v]:.2f}"[1:], ha="center", va="center", fontsize=4.6, color="white" if m[key][u, v] > 0.45 else INK)
            else:
                ax.add_patch(Rectangle((v - .5, u - .5), 1, 1, fc="#eeeeec", ec="none"))
    ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.tick_params(length=0, labelsize=6)
    ax.set_xlabel("horizontal frequency index $v$"); ax.set_ylabel("vertical frequency index $u$")
    ax.set_title(("a  " if k == 0 else "b  ") + ttl, loc="left", fontsize=7.5, fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
ax = fig.add_subplot(gs[0, 2]); bins = np.linspace(0, 24, 49)
ax.hist(b.mean_eligible, bins=bins, color=BLUE, alpha=0.85, label=f"baseline (mean {b.mean_eligible.mean():.2f})")
ax.hist(f.mean_eligible, bins=bins, color=ORANGE, alpha=0.85, label=f"fine-tuned (mean {f.mean_eligible.mean():.2f})")
ax.set_xlabel(r"mean eligible coefficients per block, $\bar e$"); ax.set_ylabel("number of covers (of 400)"); ax.legend(fontsize=6, loc="upper center")
ax.set_title("c  eligible-coefficient count", loc="left", fontsize=7.5, fontweight="bold"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[1, :2]); fr = np.arange(1, 128) / 256.0
for arr, col, lab, ls in ((p["pb"], BLUE, "baseline SD v1.5 covers (n = 400)", "-"), (p["pf"], ORANGE, "fine-tuned covers (n = 400)", "-"), (p["pn"], AQUA, "photographs, Lorem Picsum (n = 195)", "--")):
    lg = np.log10(arr); ax.plot(fr, lg.mean(0), color=col, lw=1.4, ls=ls, label=lab)
    ax.fill_between(fr, np.percentile(lg, 25, 0), np.percentile(lg, 75, 0), color=col, alpha=0.15, lw=0)
ax.set_xscale("log"); ax.set_xlabel("spatial frequency (cycles per pixel)"); ax.set_ylabel(r"$\log_{10}$ radially averaged power of $Y$")
ax.legend(fontsize=6, loc="lower left"); ax.grid(color=GRID, lw=0.5); ax.set_title("d  luminance power spectrum (mean, interquartile band)", loc="left", fontsize=7.5, fontweight="bold")
ax = fig.add_subplot(gs[1, 2])
ax.hist(b.capacity_bpp[~b.fallback], bins=np.linspace(0, 0.6, 31), color=BLUE, alpha=0.85, label="baseline (n = %d)" % (~b.fallback).sum())
ax.hist(f.capacity_bpp, bins=np.linspace(0, 0.6, 31), color=ORANGE, alpha=0.85, label="fine-tuned (n = 400)")
ax.set_xlabel("raw slot capacity (bits per pixel)"); ax.set_ylabel("number of covers"); ax.legend(fontsize=6, loc="upper right"); ax.set_ylim(0, 150)
ax.set_title("e  adaptive capacity", loc="left", fontsize=7.5, fontweight="bold"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.savefig("../figures/fig2_dct_statistics.pdf", bbox_inches="tight"); fig.savefig("fig3.png", dpi=150, bbox_inches="tight")
# numbers for the text
sl = lambda arr: np.polyfit(np.log10(fr[4:100]), np.log10(arr).mean(0)[4:100], 1)[0]
print("spectral slopes (log-log, 0.02-0.39 c/px): base %.2f ft %.2f natural %.2f" % (sl(p["pb"]), sl(p["pf"]), sl(p["pn"])))
hf = lambda arr: (arr[:, 64:].sum(1) / arr.sum(1))
print("fraction of AC power above 0.25 c/px (median): base %.4f ft %.4f natural %.4f" % (np.median(hf(p["pb"])), np.median(hf(p["pf"])), np.median(hf(p["pn"]))))
