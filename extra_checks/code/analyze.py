"""Summarises whatever results exist so far into results/summary.txt (safe to run at any time)."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import common as C

R = os.path.join(C.HERE, "results")
out = []


def say(s=""):
    print(s); out.append(s)


def paired(a, b):
    d = (a - b).dropna()
    return f"mean diff {d.mean():+.3f}, median {d.median():+.3f}, p = {wilcoxon(d).pvalue:.2g} (n = {len(d)})" if len(d) > 5 else "n too small"


# ---------------- test 3
p = os.path.join(R, "test3_random_bits.csv")
if os.path.exists(p):
    t = pd.read_csv(p)
    a = pd.read_csv(C.NORMAL_CSV)
    a = a[a.condition == "none"][["image", "covers", "psnr_y", "ssim_y", "fix_passes", "exact"]]
    m = t.merge(a, on=["image", "covers"], suffixes=("_rand", "_ascii"))
    say("== TEST 3: random bits versus ASCII text (same covers, K and payload length)")
    say(f"pairs done: {t.key.nunique()} of 100")
    for cs in ("finetuned", "baseline"):
        s = m[m.covers == cs]
        say(f"{cs:9s}: exact recovery random {int(s.exact_rand.sum())}/{len(s)} (ASCII {int(s.exact_ascii.sum())}/{len(s)}); "
            f"PSNR_Y random {s.psnr_y_rand.mean():.2f} vs ASCII {s.psnr_y_ascii.mean():.2f} dB, {paired(s.psnr_y_rand, s.psnr_y_ascii)}; "
            f"SSIM random {s.ssim_y_rand.mean():.4f} vs ASCII {s.ssim_y_ascii.mean():.4f}")
    w = m.pivot_table(index="image", columns="covers", values=["psnr_y_rand", "psnr_y_ascii", "ssim_y_rand"])
    say(f"fine-tuned minus baseline PSNR_Y: random bits {(w.psnr_y_rand.finetuned - w.psnr_y_rand.baseline).mean():+.2f} dB "
        f"(higher on fine-tuned in {int(((w.psnr_y_rand.finetuned - w.psnr_y_rand.baseline) > 0).sum())}/{len(w)}), "
        f"ASCII {(w.psnr_y_ascii.finetuned - w.psnr_y_ascii.baseline).mean():+.2f} dB")
    say(f"  {paired(w.psnr_y_rand.finetuned, w.psnr_y_rand.baseline)}")
    say("")

# ---------------- test 1
p = os.path.join(R, "test1_stats.csv")
if os.path.exists(p):
    s = pd.read_csv(p)
    say("== TEST 1a: cover statistics before embedding")
    say(f"{'variant':12s} {'n':>5s} {'e_bar':>7s} {'fallback':>9s} {'cap bpp (cover-derived)':>24s} {'>=0.30 bpp':>11s} "
        f"{'HF share %':>11s} {'slope':>6s} {'PSNR vs base':>13s}")
    order = ["finetuned", "baseline", "photo"] + sorted(v for v in s.variant.unique() if v.startswith("sharp")) + \
            sorted(v for v in s.variant.unique() if v.startswith("noise"))
    for v in order:
        g = s[s.variant == v]
        if g.empty:
            continue
        fb = g.K.isna().sum()
        cap = g.capacity_bpp.dropna()
        pv = f"{g.psnr_vs_baseline.mean():.1f} dB" if "psnr_vs_baseline" in g and g.psnr_vs_baseline.notna().any() else ""
        say(f"{v:12s} {len(g):5d} {g.e_bar.mean():7.2f} {fb:9d} {cap.mean():12.3f} (n={len(cap):4d})   {int((cap >= 0.30).sum()):5d}/{len(g):<5d}"
            f" {100*g.hf_share.median():10.2f} {g.slope.median():6.2f} {pv:>13s}")
    say("  (HF share: median share of luminance power above 0.25 cycles/pixel; slope: median log-log slope, 0.02 to 0.39)")
    say("")

p = os.path.join(R, "test1_embed.csv")
if os.path.exists(p):
    e = pd.read_csv(p)
    e["set"] = e.key.str.split(":").str[0]
    say("== TEST 1b: matched payload (same K and message as the paper)")
    for v in e.set.unique():
        g = e[e.set == v]
        ex = int(g.exact.fillna(False).sum()) if "exact" in g else 0
        say(f"{v:22s} n={len(g):4d}  exact {ex}/{len(g)}  PSNR_Y {g.psnr_y.mean():.2f} dB  SSIM {g.ssim_y.mean():.4f}")
    ft = e[e.set == "finetuned"].set_index("pair")
    for v in e.set.unique():
        if v in ("finetuned", "finetuned_photo_pairs", "photo"):
            continue
        g = e[e.set == v].set_index("pair")
        say(f"  finetuned minus {v}: PSNR {paired(ft.psnr_y, g.psnr_y)}")
    ph = e[e.set == "photo"].set_index("pair")
    ftp = e[e.set.isin(["finetuned", "finetuned_photo_pairs"]) & (e.pair < len(ph))].drop_duplicates("pair").set_index("pair")
    if len(ph):
        say(f"  finetuned (pairs 0..{len(ph)-1}) minus photos at the same K and payload: PSNR {paired(ftp.psnr_y, ph.psnr_y)}")
    say("")

p = os.path.join(R, "test2_generation.csv")
if os.path.exists(p):
    g = pd.read_csv(p)
    say("== TEST 2: normal generation settings")
    for _, r in g[g.setting == "check"].iterrows():
        say(f"check: {r.model} corpus image 000 regenerated identically: {r.identical_to_corpus}")
    st = pd.read_csv(os.path.join(C.HERE, "results", "test1_stats.csv")) if os.path.exists(os.path.join(R, "test1_stats.csv")) else None
    for setting in ("cfg256", "cfg512"):
        for model in ("baseline", "finetuned"):
            h = g[(g.setting == setting) & (g.model == model)]
            if h.empty:
                continue
            cap = h.capacity_bpp.dropna()
            line = (f"{setting} {model:9s} n={len(h):3d}  e_bar {h.e_bar.mean():6.2f}  fallback {h.K.isna().sum():3d}  "
                    f"cap {cap.mean():.3f} bpp (n={len(cap)})  >=0.30: {int((cap >= 0.3).sum())}/{len(h)}  HF {100*h.hf_share.median():.2f}%")
            if setting == "cfg512" and "down256_e_bar" in h:
                line += f"  | downscaled to 256: e_bar {h.down256_e_bar.mean():.2f}, fallback {h.down256_K.isna().sum()}"
            say(line)
            if st is not None:
                ref = st[(st.variant == model) & st["index"].isin(h["index"])]
                say(f"    same prompts in the corpus (no CFG, 256): e_bar {ref.e_bar.mean():.2f}, fallback {ref.K.isna().sum()}, "
                    f"cap {ref.capacity_bpp.dropna().mean():.3f} bpp")

open(os.path.join(R, "summary.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
