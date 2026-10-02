r"""Stage 2 lightweight tests (T1-T10). Read-only on the research data; writes only into this folder.
The embedder under test is the unmodified F:\Results\DCT_Adaptive.py."""
import os, io, sys, json, zlib, contextlib, importlib.util, random, string
import numpy as np, pandas as pd, cv2
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("DCT_Adaptive", r"F:\Results\DCT_Adaptive.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
RES = r"F:\Results"; OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)
def forced_K(K):
    @contextlib.contextmanager
    def cm():
        o = D._compute_K; D._compute_K = lambda b: K
        try: yield
        finally: D._compute_K = o
    return cm()

# ------------------------------------------------------------------ T1
log("== T1 DCT / inverse DCT round trip")
rng = np.random.default_rng(0)
imgs = {"constant 128": np.full((64, 64), 128.0), "constant 0": np.zeros((64, 64)), "constant 255": np.full((64, 64), 255.0),
        "impulse": np.pad(np.array([[255.0]]), ((31, 32), (31, 32))), "horizontal gradient": np.tile(np.linspace(0, 255, 64), (64, 1)),
        "vertical gradient": np.tile(np.linspace(0, 255, 64), (64, 1)).T, "checkerboard": (np.indices((64, 64)).sum(0) % 2) * 255.0,
        "random": rng.integers(0, 256, (64, 64)).astype(float)}
t1 = []
for name, y in imgs.items():
    y8 = np.rint(y).astype(np.uint8)
    blocks, h8, w8 = D._to_blocks(y8)
    rec = np.stack([[D._idct2(blocks[i, j]) + 128.0 for j in range(blocks.shape[1])] for i in range(blocks.shape[0])])
    rec = rec.transpose(0, 2, 1, 3).reshape(64, 64)
    back = D._from_blocks(blocks, y8, h8, w8)
    energy_ok = np.isclose((blocks ** 2).sum(), ((y8.astype(float) - 128) ** 2).sum())
    t1.append({"image": name, "max_abs_error_float": float(np.abs(rec - y8).max()), "uint8_identical": bool((back == y8).all()), "parseval_holds": bool(energy_ok)})
t1 = pd.DataFrame(t1); log(t1.to_string(index=False))

# ------------------------------------------------------------------ T2
log("\n== T2 coefficient positions")
pos = pd.DataFrame([{"zigzag_rank_within_set": r, "u": u, "v": v, "u_plus_v": u + v, "Q": int(D._Q_TABLE[u, v]), "zigzag_index": D._ZZ_IDX[(u, v)],
                     "used_for_header": (u, v) in D._HEADER_POSITIONS} for r, (u, v) in enumerate(D._PAYLOAD_CANDIDATES)])
pos.to_csv(os.path.join(HERE, "T2_positions.csv"), index=False)
log("payload positions:", len(pos), "| DC included:", bool(((pos.u == 0) & (pos.v == 0)).any()), "| min u+v:", int(pos.u_plus_v.min()), "| max u+v:", int(pos.u_plus_v.max()),
    "| positions with u+v<=5:", int((pos.u_plus_v <= 5).sum()), "| header positions:", int(pos.used_for_header.sum()))
log("u+v histogram:", pos.u_plus_v.value_counts().sort_index().to_dict())

# ------------------------------------------------------------------ T3
log("\n== T3 parity rule over q = -5..5, three offsets, both bits, all 39 positions")
bad_parity = bad_step = small = total = 0
for (u, v) in D._PAYLOAD_CANDIDATES:
    Q = float(D._Q_TABLE[u, v])
    for q in range(-5, 6):
        for off in (-0.4, 0.0, 0.4):
            c = (q + off) * Q
            for bit in (0, 1):
                c2 = D._lsb_write(c, u, v, bit); q0 = D._qi(c, u, v); q2 = D._qi(c2, u, v); total += 1
                bad_parity += (q2 & 1) != bit; bad_step += abs(q2 - q0) > 1; small += (abs(q2) < 2)
log(f"cases {total} | wrong parity {bad_parity} | change larger than one step {bad_step} | results with |q'|<2: {small} (zero or +-1 kept or produced)")

# ------------------------------------------------------------------ T4, T5, T6
log("\n== T4 payload patterns (five fine-tuned covers, forced K=5, 600 bytes)")
covers = [cv2.imread(os.path.join(RES, "finetuned", f"{i:03d}.png")) for i in (0, 80, 160, 240, 320)]
pats = {"all zero bits": "\x00" * 600, "0x7F bytes (seven ones)": "\x7f" * 600, "alternating 0x55": "U" * 600,
        "random ASCII": "".join(random.Random(1).choice(string.ascii_letters + string.digits + " .,!?") for _ in range(600)), "one byte": "A"}
t4 = []
for pname, msg in pats.items():
    ok = 0
    for c in covers:
        with forced_K(5):
            s = quiet(D.embed, c, msg)
        cv2.imwrite(os.path.join(HERE, "_tmp.png"), s)
        try: ok += quiet(D.extract, cv2.imread(os.path.join(HERE, "_tmp.png"))) == msg
        except Exception: pass
    t4.append({"pattern": pname, "exact_of_5": ok})
log(pd.DataFrame(t4).to_string(index=False)); log("note: the API accepts text only, so an all-ones byte stream (0xFF) cannot be passed; it is not valid UTF-8.")

log("\n== T5 limits")
c = covers[0]; cap = quiet(D.get_capacity, c); nb = cap["payload_bits"] // 8
with forced_K(5):
    capK5 = (1022 * 6) // 8
    m_full = "x" * capK5; s_full = quiet(D.embed, c, m_full); cv2.imwrite(os.path.join(HERE, "_tmp.png"), s_full)
    r_full = quiet(D.extract, cv2.imread(os.path.join(HERE, "_tmp.png"))) == m_full
    try: quiet(D.embed, c, "x" * (capK5 + 1)); over = "NO ERROR (unexpected)"
    except ValueError as e: over = "ValueError raised"
    s_empty = quiet(D.embed, c, "")
    try: quiet(D.extract, s_empty); empty = "extracted"
    except ValueError as e: empty = "extract raises ValueError: " + str(e)[:60]
log(f"payload exactly at capacity (K=5, {capK5} bytes): exact recovery {r_full}"); log("one byte over capacity:", over); log("empty message: embed returns an image;", empty)

log("\n== T6 corrupted header / payload")
with forced_K(5):
    st = quiet(D.embed, c, pats["random ASCII"])
Y = cv2.split(cv2.cvtColor(st, cv2.COLOR_BGR2YCrCb))[0]; blocks, _, _ = D._to_blocks(Y)
hb = [D._lsb_read(blocks[bi, bj, u, v], u, v) for bi, bj in D.HEADER_BLOCKS for (u, v) in D._HEADER_POSITIONS]
accepted = sum(D._unpack_header(hb[:k] + [1 - hb[k]] + hb[k + 1:])[3] for k in range(64))
log(f"single flipped header bit accepted in {accepted} of 64 positions (0 means every single-bit header error is detected)")
rngh = np.random.default_rng(1); fp = sum(D._unpack_header([int(b) for b in rngh.integers(0, 2, 64)])[3] for _ in range(200000))
log(f"random 64-bit headers accepted: {fp} of 200000 (expected 2^-30 each)")
slot = D._build_slot_order(5, 32, 32); n = len(pats["random ASCII"]) * 8
bits = [D._lsb_read(blocks[i, j, u, v], u, v) for (i, j, u, v) in slot[:n]]
def decode(b):
    ba = bytearray(sum(b[i + k] << (7 - k) for k in range(8)) for i in range(0, len(b) - 7, 8))
    try: return ba.decode("utf-8")
    except UnicodeDecodeError: return None
res = {"wrong text returned silently": 0, "UTF-8 error raised": 0}
for k in rngh.choice(n, 400, replace=False):
    b2 = bits.copy(); b2[k] ^= 1; d = decode(b2)
    res["UTF-8 error raised" if d is None else "wrong text returned silently"] += 1
log("single flipped payload bit (400 trials):", res, "-> the payload itself carries no checksum")

# ------------------------------------------------------------------ T7, T8
log("\n== T7 extract() on 800 clean covers (false positives)")
fp7 = 0
for folder in ("baseline", "finetuned"):
    for i in range(400):
        try: quiet(D.extract, cv2.imread(os.path.join(RES, folder, f"{i:03d}.png"))); fp7 += 1
        except ValueError: pass
log(f"covers reported as containing a message: {fp7} of 800")

log("\n== T8 headers of the 400 original stego images against comparison_results.xlsx")
cmp_ = pd.read_excel(os.path.join(RES, "comparison_results.xlsx"), sheet_name="per_image_results").set_index("image")
alphabet = set(string.ascii_letters + string.digits + " .,!?"); okh = okn = okk = oka = 0
for i in range(400):
    n_ = f"{i:03d}.png"; st_ = cv2.imread(os.path.join(RES, "stego_finetuned", n_))
    Y = cv2.split(cv2.cvtColor(st_, cv2.COLOR_BGR2YCrCb))[0]; b, _, _ = D._to_blocks(Y); nb_, K_, _, ok_ = D._read_header(b)
    okh += ok_; okn += nb_ == int(cmp_.loc[n_, "n_bits"]); okk += K_ == int(cmp_.loc[n_, "K"])
    try: oka += set(quiet(D.extract, st_)) <= alphabet
    except Exception: pass
log(f"header valid {okh}/400 | length matches spreadsheet {okn}/400 | K matches {okk}/400 | decoded text entirely in the message alphabet {oka}/400")

# ------------------------------------------------------------------ T9
log("\n== T9 unusual inputs")
c250 = covers[1][:250, :250].copy()
with forced_K(5): s250 = quiet(D.embed, c250, "border test " * 20)
log("250x250 input: exact", quiet(D.extract, s250) == "border test " * 20, "| untouched border pixels identical:", bool((s250[248:, :] == c250[248:, :]).all() and (s250[:, 248:] == c250[:, 248:]).all()))
gray = cv2.cvtColor(covers[2], cv2.COLOR_BGR2GRAY); cv2.imwrite(os.path.join(HERE, "_gray.png"), gray); g3 = cv2.imread(os.path.join(HERE, "_gray.png"))
with forced_K(5): sg = quiet(D.embed, g3, "gray test " * 20)
log("greyscale PNG (read by OpenCV as 3 channels): exact", quiet(D.extract, sg) == "gray test " * 20)
rgba = cv2.cvtColor(covers[3], cv2.COLOR_BGR2BGRA); rgba[..., 3] = 128; cv2.imwrite(os.path.join(HERE, "_rgba.png"), rgba)
log("RGBA PNG read with cv2.imread default -> shape", cv2.imread(os.path.join(HERE, "_rgba.png")).shape, "(alpha dropped silently)")
try: quiet(D.embed, gray, "x"); log("2-D array passed directly: no error (unexpected)")
except Exception as e: log("2-D array passed directly:", type(e).__name__)
for f in ("_tmp.png", "_gray.png", "_rgba.png"):
    try: os.remove(os.path.join(HERE, f))
    except OSError: pass

# ------------------------------------------------------------------ T10
log("\n== T10 subject-level statistics (40 prompt subjects)")
d = cmp_.reset_index(); d["subject"] = d.image.str[:3].astype(int) // 10
cs = pd.read_csv(r"F:\scirep results\03_revision_analysis\cover_stats.csv"); cs["subject"] = cs.image.str[:3].astype(int) // 10
e = cs.pivot_table(index="subject", columns="set", values="mean_eligible", aggfunc="mean")
log("eligible count, subject means: base %.2f ft %.2f | Wilcoxon n=40 p=%.2e | ft>base in %d/40" % (e.base.mean(), e.ft.mean(), stats.wilcoxon(e.base, e.ft).pvalue, (e.ft > e.base).sum()))
g = d.groupby("subject")[["baseline_psnr", "finetuned_psnr", "baseline_ssim", "finetuned_ssim"]].mean()
log("SSIM subject means: diff %.4f | p=%.2e | ft>base in %d/40" % ((g.finetuned_ssim - g.baseline_ssim).mean(), stats.wilcoxon(g.baseline_ssim, g.finetuned_ssim).pvalue, (g.finetuned_ssim > g.baseline_ssim).sum()))
log("PSNR subject means: diff %.3f dB | p=%.3f" % ((g.finetuned_psnr - g.baseline_psnr).mean(), stats.wilcoxon(g.baseline_psnr, g.finetuned_psnr).pvalue))
for lab, sel in (("payload < 0.45", d.target_bpp < 0.45), ("payload >= 0.50", d.target_bpp >= 0.50)):
    gg = d[sel].groupby("subject")[["baseline_psnr", "finetuned_psnr"]].mean()
    log(f"PSNR {lab}: subjects {len(gg)} | diff %.3f dB | p=%.2e" % ((gg.finetuned_psnr - gg.baseline_psnr).mean(), stats.wilcoxon(gg.baseline_psnr, gg.finetuned_psnr).pvalue))
def cluster_ci(sc_c, sc_s, B=2000):
    subj = np.arange(len(sc_c)) // 10; rng = np.random.default_rng(0); out = []
    for _ in range(B):
        pick = rng.integers(0, 40, 40); idx = np.concatenate([np.where(subj == s)[0] for s in pick])
        out.append(roc_auc_score(np.r_[np.zeros(len(idx)), np.ones(len(idx))], np.r_[sc_c[idx], sc_s[idx]]))
    return np.percentile(out, [2.5, 97.5])
ud = pd.read_excel(r"F:\scirep results\04_universal_dct_steganalyser\results\universal_dct_steganalysis_results.xlsx", sheet_name="per_image_detection")
u = ud[ud.set == "DEDS on fine-tuned covers"].sort_values("image")
lo, hi = cluster_ci(u.cover_p_stego.values, u.stego_p_stego.values)
log("CNN (universal DCT), fine-tuned covers: AUC %.3f | image-level CI 0.832-0.886 | subject-clustered CI %.3f-%.3f" % (roc_auc_score(np.r_[np.zeros(400), np.ones(400)], np.r_[u.cover_p_stego, u.stego_p_stego]), lo, hi))
z = np.load(r"F:\scirep results\03_revision_analysis\steganalysis_spam_scores.npz", allow_pickle=True); names = list(z["names"])
for key in ("S-UNIWARD, fine-tuned covers (matched detector)", "DEDS embedder, fine-tuned covers (detector trained on S-UNIWARD)"):
    i = names.index(key); c_, s_ = z[f"e{i}_c"], z[f"e{i}_s"]; lo, hi = cluster_ci(c_, s_)
    log(f"SPAM | {key}: AUC %.3f | subject-clustered CI %.3f-%.3f" % (roc_auc_score(np.r_[np.zeros(400), np.ones(400)], np.r_[c_, s_]), lo, hi))

open(os.path.join(HERE, "stage2_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
t1.to_csv(os.path.join(HERE, "T1_dct_roundtrip.csv"), index=False)
