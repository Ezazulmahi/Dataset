"""
08_validate_theory.py  -  empirical checks of every derived result in the v5 Methods.

V1  Expected distortion  D(mu) = (1/2 + mu)^3 Q^2 / 3
      (a) synthetic coefficients with uniform in-bin offset (the model's assumption)
      (b) the real first write pass on 100 fine-tuned + 100 baseline covers at the
          Table 2 payloads, coefficient domain, before rounding; and the predicted vs
          measured gain over the bin-centre rule (mu = 1/2) in dB
V2  Parseval: squared coefficient change = squared pixel change before rounding
V3  Rounding noise per coefficient: s.d. 1/sqrt(12) = 0.289
V4  Margin guarantee: a parity written with margin mu survives requantisation with the
    own table scaled by s whenever the step change bound Q_s/2 < mu Q (coefficient
    domain, no pixel clipping) - grid over mu and s
V5  Reliability weight a = 1 - 2|f|: accuracy of single reads against a, robust-mode
    stego images under JPEG 90 / noise 5
V6  Soft vote vs plain majority vote, same reads, robust-mode subset
Output: results/validation/*.csv, results/validation/validation_summary.txt,
        figures/figS2_validation.png/.pdf
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, random, string, importlib.util
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "results", "validation"); os.makedirs(OUTD, exist_ok=True)
CORP = r"F:\scirep results\09_expansion_1000\corpus1000"
spec = importlib.util.spec_from_file_location("D", os.path.join(HERE, "embedder", "DCT_Adaptive_v5.py"))
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
D.PREFER_ELIGIBLE = False
MUS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
ALPHABET = string.ascii_letters + string.digits + " .,!?"
log = []
def out(s=""):
    print(s, flush=True); log.append(str(s))

def pred(mu):
    return (0.5 + mu) ** 3 / 3.0

def write_vec(C, Q, bits, mu):
    """vectorised _parity_write (nearest bin, margin mu) - checked against the scalar version below"""
    q = np.round(C / Q)
    m = mu * Q
    proj = lambda qc: np.clip(C, (qc - 0.5) * Q + m, (qc + 0.5) * Q - m)
    same = proj(q); up = proj(q + 1); dn = proj(q - 1)
    flip = np.where(np.abs(up - C) <= np.abs(dn - C), up, dn)
    return np.where((q.astype(np.int64) & 1) == bits, same, flip)


# ── V1a synthetic ────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
N = 2_000_000
rows = []
for Qv in (8.0, 13.0, 19.0):
    q0 = rng.integers(-40, 41, N).astype(float)
    C = q0 * Qv + rng.uniform(-Qv / 2, Qv / 2, N)
    bits = rng.integers(0, 2, N)
    for mu in MUS:
        e = write_vec(C, np.full(N, Qv), bits, mu) - C
        rows.append(dict(Q=Qv, mu=mu, measured=np.mean(e ** 2) / Qv ** 2, predicted=pred(mu)))
v1a = pd.DataFrame(rows); v1a["rel_error_pct"] = 100 * (v1a.measured / v1a.predicted - 1)
v1a.to_csv(os.path.join(OUTD, "V1a_synthetic.csv"), index=False)
out("V1a synthetic (uniform offsets), E[e^2]/Q^2 measured vs (1/2+mu)^3/3:")
out(v1a.groupby("mu")[["measured", "predicted", "rel_error_pct"]].mean().round(5).to_string())
# scalar/vector agreement
idx = rng.integers(0, N, 2000)
u, v = D._PAYLOAD_CANDIDATES[5]; Qs = float(D._Q_TABLE[u, v])
Cs = rng.normal(0, 30, 2000); bs = rng.integers(0, 2, 2000)
agree = all(abs(D._parity_write(c, u, v, int(b), margin=0.15) - x) < 1e-9 for c, b, x in zip(Cs, bs, write_vec(Cs, np.full(2000, Qs), bs, 0.15)))
out(f"vectorised write == embedder's _parity_write on 2,000 random cases: {agree}")


# ── V1b / V2 / V3 / V4 on real covers ───────────────────────────────────────
def real_cover(task):
    i, cov, K, n = task
    name = f"{i:03d}.png"
    Y = cv2.split(cv2.cvtColor(cv2.imread(os.path.join(CORP, cov, name)), cv2.COLOR_BGR2YCrCb))[0]
    blocks, h8, w8 = D._to_blocks(Y)
    msg = "".join(random.Random(424242 + i).choice(ALPHABET) for _ in range(n // 8))
    bits = np.array([(b >> (7 - k)) & 1 for b in msg.encode() for k in range(8)])
    I, J, U, V, Q = D._slot_arrays(D._build_slot_order(K, 32, 32)[:len(bits)])
    C = blocks[I, J, U, V]
    res = dict(image=name, covers=cov, n=len(bits), q_le1=float(np.mean(np.abs(np.round(C / Q)) <= 1)),
               offset_ks=float(np.max(np.abs(np.sort(np.abs(C / Q - np.round(C / Q))) - np.linspace(0, 0.5, len(C))))))
    for mu in MUS:
        Cw = write_vec(C, Q, bits, mu)
        res[f"D_{mu}"] = float(np.mean(((Cw - C) / Q) ** 2))           # E[e^2]/Q^2 (per slot, normalised)
        res[f"E_{mu}"] = float(np.sum((Cw - C) ** 2))                    # total coefficient-domain energy
        if mu in (0.15, 0.25, 0.5):
            # V4: requantise written coefficients with the scaled own table, coefficient domain only
            for s_q, s in ((50, 1.0), (70, 0.6), (80, 0.4), (90, 0.2), (95, 0.1)):
                Qsc = np.clip(np.floor((Q * (200 - 2 * s_q) + 50) / 100), 1, 255)
                Cr = np.round(Cw / Qsc) * Qsc
                res[f"surv_{mu}_{s}"] = float(np.mean((np.round(Cr / Q).astype(np.int64) & 1) == bits))
                res[f"bound_{mu}_{s}"] = float(np.mean(Qsc / 2 < mu * Q))  # share of slots where the bound applies
    # V2 Parseval at mu = 0.15: write all slots into a copy of the blocks, inverse DCT without rounding
    b2 = blocks.copy(); b2[I, J, U, V] = write_vec(C, Q, bits, 0.15)
    pix = D._idct_tiles(b2) - D._idct_tiles(blocks)
    res["parseval_coeff"] = float(np.sum((b2 - blocks) ** 2)); res["parseval_pixel"] = float(np.sum(pix ** 2))
    # V3 rounding noise: DCT of (round(x) - x) for the unrounded written image
    x = D._idct_tiles(b2); err = D._to_blocks(np.zeros((256, 256)))[0]  # shape helper
    r = np.round(x) - x
    tiles = r                                                            # already in block layout
    from scipy.fft import dct
    R = dct(dct(tiles, axis=2, norm="ortho"), axis=3, norm="ortho")
    res["round_noise_sd"] = float(R[I, J, U, V].std())
    return res


def reliability(task):
    name, lvl = task
    i = int(name[:3])
    nb = min(max(1, int(round(lvl * 65536 / 8))), (1022 * 39 - 32) // 8)
    msg = "".join(random.Random(777000 + i).choice(ALPHABET) for _ in range(nb))
    F = D._frame(msg.encode()); L = len(F)
    st = cv2.imread(os.path.join(HERE, "stego_robust", name))
    target = np.resize(F, 1022 * 39)
    rows, votes = [], []
    seed = i * 2 + 1
    for cond, fn in (("jpeg_90", lambda s: cv2.imdecode(cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1)),
                     ("jpeg_95", lambda s: cv2.imdecode(cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)),
                     ("noise_5", lambda s: np.clip(s + np.random.default_rng(seed).normal(0, 5, s.shape), 0, 255).astype(np.uint8)),
                     ("noise_10", lambda s: np.clip(s + np.random.default_rng(seed).normal(0, 10, s.shape), 0, 255).astype(np.uint8))):
        _, soft = D._soft_reads(fn(st), D.DEFAULT_KEY)
        b = (soft > 0).astype(np.int64); a = np.abs(soft)
        hist = np.histogram(a, bins=10, range=(0, 1))[0]
        corr = np.histogram(a[b == target], bins=10, range=(0, 1))[0]
        rows.append(dict(image=name, level=lvl, cond=cond, **{f"n{k}": hist[k] for k in range(10)}, **{f"c{k}": corr[k] for k in range(10)}))
        hard = np.where(soft > 0, 1.0, -1.0)
        dec_soft = (D._votes(soft, L) > 0).astype(np.int64)
        dec_hard = (D._votes(hard, L) > 0).astype(np.int64)
        votes.append(dict(image=name, level=lvl, cond=cond, raw_pct=100 * np.mean(b == target),
                          soft_pct=100 * np.mean(dec_soft[16:-16] == F[16:-16]), hard_pct=100 * np.mean(dec_hard[16:-16] == F[16:-16]),
                          soft_exact=bool(np.all(dec_soft == F)), hard_exact=bool(np.all(dec_hard == F))))
    return rows, votes


if __name__ == "__main__":
    cmp_ = pd.read_excel(os.path.join(CORP, "comparison_results.xlsx"), sheet_name="per_image_results")
    sel = cmp_.iloc[::10]
    tasks = [(int(r.image[:3]), cov, int(r.K), int(r.n_bits)) for r in sel.itertuples() for cov in ("finetuned", "baseline")]
    with Pool(3) as p:
        real = pd.DataFrame(p.map(real_cover, tasks))
    real.to_csv(os.path.join(OUTD, "V1b_V2_V3_V4_real_covers.csv"), index=False)
    out(f"\nV1b real covers ({len(real)} covers, Table 2 payloads, first write pass, coefficient domain):")
    for cov, g in real.groupby("covers"):
        out(f"  {cov}: share of slots with |q|<=1: {g.q_le1.mean():.3f}")
        for mu in MUS:
            gain_meas = 10 * np.log10(g["E_0.5"].sum() / g[f"E_{mu}"].sum())
            gain_pred = 10 * np.log10(pred(0.5) / pred(mu))
            out(f"    mu={mu:.2f}: E[e^2]/Q^2 measured {g[f'D_{mu}'].mean():.4f}, uniform-model {pred(mu):.4f}; "
                f"gain over mu=0.5: measured {gain_meas:.2f} dB, predicted {gain_pred:.2f} dB")
    out(f"\nV2 Parseval: max relative difference coefficient vs pixel energy = "
        f"{np.max(np.abs(real.parseval_coeff / real.parseval_pixel - 1)):.2e}")
    out(f"V3 rounding noise per coefficient: s.d. {real.round_noise_sd.mean():.4f} (theory 1/sqrt(12) = {1/np.sqrt(12):.4f})")
    out("\nV4 parity survival after requantisation with the scaled own table (coefficient domain):")
    rows = []
    for mu in (0.15, 0.25, 0.5):
        for s in (1.0, 0.6, 0.4, 0.2, 0.1):
            rows.append(dict(mu=mu, scale=s, predicted_safe=f"{100*real[f'bound_{mu}_{s}'].mean():.0f}% of slots",
                             survival_pct=100 * real[f"surv_{mu}_{s}"].mean(), min_image_pct=100 * real[f"surv_{mu}_{s}"].min()))
    v4 = pd.DataFrame(rows); v4.to_csv(os.path.join(OUTD, "V4_margin_guarantee.csv"), index=False); out(v4.round(3).to_string(index=False))

    asg = pd.read_csv(os.path.join(HERE, "results", "robust_subset_assignment.csv"))
    with Pool(3) as p:
        res = p.map(reliability, list(zip(asg.image, asg.level_bpp)))
    rel = pd.DataFrame([r for rr, _ in res for r in rr]); vot = pd.DataFrame([v for _, vv in res for v in vv])
    rel.to_csv(os.path.join(OUTD, "V5_reliability.csv"), index=False); vot.to_csv(os.path.join(OUTD, "V6_soft_vs_hard.csv"), index=False)
    out("\nV5 single-read accuracy by reliability a (all 200 robust stego images):")
    acc = {}
    for cond, g in rel.groupby("cond"):
        n_ = g[[f"n{k}" for k in range(10)]].sum().values; c_ = g[[f"c{k}" for k in range(10)]].sum().values
        acc[cond] = 100 * c_ / np.maximum(n_, 1)
        out(f"  {cond:8s}: " + " ".join(f"{x:5.1f}" for x in acc[cond]) + "   (a bins 0-0.1 ... 0.9-1.0)")
    pd.DataFrame(acc, index=[f"{k/10:.1f}-{(k+1)/10:.1f}" for k in range(10)]).to_csv(os.path.join(OUTD, "V5_accuracy_by_a.csv"))
    out("\nV6 soft vote vs majority vote (same reads): mean % of message bits / exact recoveries")
    s6 = vot.groupby(["cond", "level"]).agg(raw=("raw_pct", "mean"), hard=("hard_pct", "mean"), soft=("soft_pct", "mean"),
                                           hard_exact=("hard_exact", "sum"), soft_exact=("soft_exact", "sum"), n=("image", "size"))
    out(s6.round(1).to_string()); s6.to_csv(os.path.join(OUTD, "V6_summary.csv"))
    out(f"  overall exact recoveries: soft {int(vot.soft_exact.sum())} vs majority {int(vot.hard_exact.sum())} of {len(vot)} decodes")
    open(os.path.join(OUTD, "validation_summary.txt"), "w", encoding="utf-8").write("\n".join(log))

    # ── figure S2 ─────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
    mm = np.linspace(0.01, 0.5, 100)
    ax[0].plot(mm, pred(mm), color="#30363d", lw=1, label="theory $(1/2+\\mu)^3/3$")
    g = v1a.groupby("mu").measured.mean(); ax[0].plot(g.index, g.values, "o", color="#2a78d6", ms=3, label="synthetic")
    for cov, col in (("finetuned", "#eb6834"), ("baseline", "#1baf7a")):
        ax[0].plot(MUS, [real[real.covers == cov][f"D_{m}"].mean() for m in MUS], "s", color=col, ms=3, label=f"{cov} covers")
    ax[0].set_xlabel("margin $\\mu$"); ax[0].set_ylabel("$E[e^2]/Q^2$"); ax[0].legend(fontsize=5); ax[0].set_title("a", loc="left", fontweight="bold")
    for mu, col in ((0.15, "#eb6834"), (0.25, "#2a78d6"), (0.5, "#30363d")):
        sub = v4[v4.mu == mu]; ax[1].plot(sub.scale, sub.survival_pct, "o-", color=col, ms=3, lw=1, label=f"$\\mu$={mu}")
        ax[1].axvline(2 * mu, color=col, lw=0.6, ls=":")
    ax[1].set_xlabel("table scale s"); ax[1].set_ylabel("parities kept (%)"); ax[1].legend(fontsize=5); ax[1].set_title("b", loc="left", fontweight="bold")
    centers = np.arange(0.05, 1.0, 0.1)
    for cond, col in (("jpeg_95", "#2a78d6"), ("jpeg_90", "#eb6834"), ("noise_5", "#1baf7a"), ("noise_10", "#4a3aa7")):
        ax[2].plot(centers, acc[cond], "o-", color=col, ms=2.5, lw=1, label=cond.replace("_", " "))
    ax[2].set_xlabel("reliability $a$"); ax[2].set_ylabel("single-read accuracy (%)"); ax[2].legend(fontsize=5); ax[2].set_title("c", loc="left", fontweight="bold")
    vv = vot.groupby("level")[["hard_pct", "soft_pct"]].mean()
    ax[3].plot(vv.index, vv.hard_pct, "o-", color="#30363d", ms=3, lw=1, label="majority vote")
    ax[3].plot(vv.index, vv.soft_pct, "o-", color="#eb6834", ms=3, lw=1, label="soft vote")
    ax[3].set_xscale("log"); ax[3].set_xlabel("net payload (bpp)"); ax[3].set_ylabel("message bits recovered (%)")
    ax[3].legend(fontsize=5); ax[3].set_title("d", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "figS2_validation.pdf"), dpi=300)
    fig.savefig(os.path.join(HERE, "figures", "figS2_validation.png"), dpi=150)
