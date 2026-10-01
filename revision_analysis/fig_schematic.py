"""Figure 1: technical schematic of the training objective and the embedding/extraction pipeline."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
FROZEN = "#e8eef7"; TRAIN = "#fde6da"; OP = "#f3f3f1"; LOSS = "#e3f4ec"; EDGE = "#3c4650"; GRAD = "#c2410c"
fig, ax = plt.subplots(figsize=(7.2, 7.0)); ax.set_xlim(0, 100); ax.set_ylim(-4, 92); ax.axis("off")

def box(x, y, w, h, text, fc=OP, fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=EDGE, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text.replace("|", "\n"), ha="center", va="center", fontsize=fs, linespacing=1.3)
    return (x, y, w, h)

def arrow(p, q, label=None, off=(0, 1.3), color=EDGE, head=True):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>" if head else "-", lw=0.8, color=color,
                shrinkA=0, shrinkB=0, mutation_scale=7))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", va="center", fontsize=6.2, color=color)

R = lambda b: (b[0] + b[2] + 0.2, b[1] + b[3] / 2); L = lambda b: (b[0] - 0.2, b[1] + b[3] / 2)
T = lambda b: (b[0] + b[2] / 2, b[1] + b[3] + 0.2); B = lambda b: (b[0] + b[2] / 2, b[1] - 0.2)

# ---------------- panel a: training ----------------
ax.text(0, 90, "a", fontsize=10, fontweight="bold"); ax.text(3.5, 90, "Reward-guided LoRA fine-tuning (training)", fontsize=8, fontweight="bold")
b_x0 = box(0, 77, 12.5, 8, r"image $x_0$, caption $c$|(256×256 RGB)", fs=5.9)
b_enc = box(15.5, 77, 11, 8, "VAE encoder|(frozen)", FROZEN)
b_noise = box(31, 77, 19, 8, r"forward noising, Eq. (2)|$z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$")
b_unet = box(55, 77, 18, 8, r"U-Net $f_\theta$|frozen weights + LoRA|(rank 4, trainable)", TRAIN)
b_text = box(31, 64, 19, 8, r"CLIP text encoder|$\tau(c)$ (frozen)", FROZEN)
b_anchor = box(55, 64, 18, 8, r"anchor U-Net $f_{\mathrm{anchor}}$|(frozen copy, no LoRA)|same inputs $z_t,\,t,\,\tau(c)$", FROZEN)
b_ldiff = box(80, 78.2, 20, 5.6, r"$L_{\mathrm{diff}}=\Vert \varepsilon-\hat\varepsilon\Vert _2^2$   Eq. (3)", LOSS)
b_lanch = box(80, 65.2, 20, 5.6, r"$L_{\mathrm{anchor}}=\Vert \hat\varepsilon-\varepsilon_{\mathrm{a}}\Vert _2^2$   Eq. (9)", LOSS)
arrow(R(b_x0), L(b_enc)); arrow(R(b_enc), L(b_noise), r"$z_0$"); arrow(R(b_noise), L(b_unet), r"$z_t,\,t$")
arrow(R(b_unet), L(b_ldiff), r"$\hat\varepsilon$"); arrow(R(b_anchor), L(b_lanch), r"$\varepsilon_{\mathrm{a}}$")
arrow((6, 76.8), (6, 68), head=False); arrow((6, 68), L(b_text), r"$c$")
arrow(R(b_text), L(b_anchor), r"$\tau(c)$"); arrow((50.2, 70.5), (55, 76.8))

b_z0 = box(55, 49, 18, 8, r"clean-latent estimate|$\hat z_0=(z_t-\sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$", fs=5.8)
b_dec = box(33, 49, 17, 8, "VAE decoder|(frozen, differentiable)", FROZEN)
b_y = box(5, 49, 23, 8, r"luminance, Eq. (4)|$Y=0.299R+0.587G+0.114B$|$Y_c=Y-128$", fs=5.8)
b_dct = box(0, 35, 22, 8, r"8×8 block DCT-II, Eqs. (5)–(6)|$C=D\,Y_c\,D^{\top}$,  $q=C/Q$|(continuous, not rounded)", fs=5.8)
b_sig = box(27, 35, 23, 8, r"soft eligibility on $\mathcal{M}$|(39 positions with $Q\geq 8$)|$\sigma(q)=\min(\vert q\vert /2,\,1)$, Eq. (7)", fs=5.8)
b_lmid = box(55, 36.2, 20, 5.6, r"$L_{\mathrm{mid}}=-\mathbb{E}\,[\sum_{\mathcal{M}}\sigma(q)]$   Eq. (8)", LOSS, fs=5.8)
b_tot = box(80, 47.5, 20, 11, r"total loss, Eq. (1)|$L=L_{\mathrm{diff}}+\lambda_{\mathrm{mid}}L_{\mathrm{mid}}$|$+\lambda_{\mathrm{anchor}}L_{\mathrm{anchor}}$|$\lambda_{\mathrm{mid}}=0.005,\ \lambda_{\mathrm{anchor}}=1$", LOSS, fs=5.9)
arrow((76.5, 81), (76.5, 60), head=False); arrow((76.5, 60), (64, 60), head=False); arrow((64, 60), T(b_z0))
arrow(L(b_z0), R(b_dec)); arrow(L(b_dec), R(b_y), r"$\hat x_0$")
arrow((14, 48.8), (14, 43.2)); arrow(R(b_dct), L(b_sig)); arrow(R(b_sig), L(b_lmid))
arrow(R(b_lmid), (90, 39), head=False); arrow((90, 39), (90, 47.3))
arrow((90, 78), (90, 71)); arrow((90, 65), (90, 58.7))
ax.text(40, 45.9, r"gradient of $L_{\mathrm{mid}}$ flows back through the decoder and $\hat z_0$ to the LoRA factors only",
        fontsize=6.0, color=GRAD, ha="center", style="italic")

ax.plot([0, 100], [31, 31], color="#b9c0c7", lw=0.6)
# ---------------- panel b: inference ----------------
ax.text(0, 28, "b", fontsize=10, fontweight="bold"); ax.text(3.5, 28, "Cover generation, embedding and extraction (inference)", fontsize=8, fontweight="bold")
c1 = box(0, 14.5, 12, 8, r"prompt $c$|$z_T\sim\mathcal{N}(0,I)$|(seeded)")
c2 = box(15.5, 14.5, 16, 8, "fine-tuned U-Net|DDIM, 50 steps|→ VAE decoder", TRAIN)
c3 = box(35, 14.5, 13, 8, r"cover $X$|256×256 RGB|→ $Y$, Cr, Cb")
c4 = box(51.5, 14.5, 17, 8, r"block DCT of $Y$|$q=\mathrm{round}(C/Q)$|$K$, slot order (Eq. 10)", fs=5.8)
c5 = box(72, 14.5, 13, 8, "64-bit header +|payload parities|Eq. (12)", fs=5.8)
c6 = box(88.5, 14.5, 11.5, 8, "verify-fix loop|IDCT→uint8|→DCT (≤ 20×)", fs=5.8)
c7 = box(88.5, 1, 11.5, 8, r"exact-$Y$ colour|repair → stego|PNG $X'$", fs=5.8)
c8 = box(57, 1, 27, 8, r"extraction: $Y$ of $X'$ → block DCT →|check header (magic, CRC-14) →|read $n$ parities in slot order", fs=5.8)
c9 = box(35, 1, 17, 8, r"recovered bits $m$|(byte-exact on a|lossless channel)", LOSS, fs=5.8)
c0 = box(51.5, 24.3, 17, 3.6, r"message bits $m$ ($n$ bits)", fs=5.8)
for a, b in ((c1, c2), (c2, c3), (c3, c4), (c4, c5), (c5, c6)):
    arrow(R(a), L(b))
arrow(B(c6), T(c7)); arrow(L(c7), R(c8)); arrow(L(c8), R(c9))
arrow(R(c0), (78.5, 26.1), head=False); arrow((78.5, 26.1), (78.5, 22.7))
for fc, lab, x in ((FROZEN, "frozen module", 0), (TRAIN, "contains LoRA parameters", 16), (OP, "deterministic operation", 42), (LOSS, "loss / output", 66)):
    ax.add_patch(FancyBboxPatch((x, -3.6), 2.2, 2.0, boxstyle="round,pad=0.1,rounding_size=0.4", fc=fc, ec=EDGE, lw=0.6, clip_on=False))
    ax.text(x + 3.3, -2.6, lab, fontsize=6.2, va="center")
fig.savefig("../figures/fig1_schematic.pdf", bbox_inches="tight"); fig.savefig("fig1_schematic.png", dpi=170, bbox_inches="tight")
