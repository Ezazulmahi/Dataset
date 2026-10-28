"""Checks every formula and numerical claim of the v5 Methods with worked examples and Monte Carlo.
Writes results/derivations_report.txt. Each check prints PASS or FAIL."""
import os, math, zlib
import numpy as np
import common as C

D = C.load_v5()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
lines, fails = [], []


def say(s=""):
    print(s); lines.append(s)


def check(name, ok, detail=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


rng = np.random.default_rng(0)

say("== 1. Write rule, worked example (Q = 10, mu = 0.15, so the margin is 1.5)")
# position with Q = 10: (0,7)
u, v = 0, 7
assert D._Q_TABLE[u, v] == 10
cases = [(23.7, 0, 23.5, "q = 2 even, bit 0 matches, C is 0.2 above the admissible part [16.5, 23.5], pulled in by 0.2"),
         (21.0, 0, 21.0, "q = 2 even, bit 0 matches, C already admissible, not moved"),
         (23.7, 1, 26.5, "q = 2, bit 1 needs odd: q+1 = 3 gives 26.5 (move 2.8), q-1 = 1 gives 13.5 (move 10.2), nearer wins"),
         (16.2, 1, 13.5, "q = 2, bit 1: q-1 gives 13.5 (move 2.7), q+1 gives 26.5 (move 10.3)"),
         (-4.6, 1, -6.5, "q = 0 even, bit 1: q-1 = -1 gives -6.5 (move 1.9), q+1 = 1 gives 6.5 (move 11.1)")]
for c, bit, want, why in cases:
    got = D._parity_write(c, u, v, bit, margin=0.15)
    check(f"C = {c}, bit {bit} -> {want}: {why}", abs(got - want) < 1e-9, f"code gives {got:.4f}")
say("  Same coefficients with the bin-centre rule (mu = 0.5): 23.7/bit0 -> 20 (move 3.7), 23.7/bit1 -> 30 (move 6.3)")
check("bin-centre rule from the same function", abs(D._parity_write(23.7, u, v, 0, margin=0.5) - 20) < 1e-9
      and abs(D._parity_write(23.7, u, v, 1, margin=0.5) - 30) < 1e-9)

say("\n== 2. Expected distortion D_mu = (1/2 + mu)^3 Q^2 / 3 (Eq. distortion)")
say("  Derivation, matching parity: moves only if |x| > (1/2 - mu)Q, probability 2mu; move t ~ U[0, mu Q], E[t^2] = mu^2 Q^2 / 3")
say("  Derivation, parity change: s = Q/2 - |x| ~ U[0, Q/2], move s + mu Q, E = (2/Q) int_0^{Q/2} (s + mu Q)^2 ds = (2/3)[(1/2+mu)^3 - mu^3] Q^2")
say("  Average: (1/2)(2/3)mu^3 Q^2 + (1/2)(2/3)[(1/2+mu)^3 - mu^3] Q^2 = (1/2+mu)^3 Q^2 / 3")
for mu in (0.05, 0.10, 0.15, 0.25, 0.35, 0.5):
    analytic_match = 2 * mu * mu ** 2 / 3
    analytic_change = (2 / 3) * ((0.5 + mu) ** 3 - mu ** 3)
    closed = (0.5 + mu) ** 3 / 3
    check(f"mu = {mu}: algebra of the two cases equals the closed form", abs(0.5 * analytic_match + 0.5 * analytic_change - closed) < 1e-12)
Q = 10.0
N = 400_000
x = rng.uniform(-Q / 2, Q / 2, N); q0 = rng.integers(-6, 7, N); C0 = q0 * Q + x; bits = rng.integers(0, 2, N)
for mu in (0.05, 0.15, 0.25, 0.5):
    new = np.array([D._parity_write(c, u, v, int(b), margin=mu) for c, b in zip(C0[:60000], bits[:60000])])
    mc = np.mean((new - C0[:60000]) ** 2) / Q ** 2
    th = (0.5 + mu) ** 3 / 3
    check(f"Monte Carlo through the embedder's own write function, mu = {mu}", abs(mc - th) / th < 0.02, f"measured {mc:.4f}, theory {th:.4f}")
say("  Bin-centre special case: Q^2/12 if parity matches, 7Q^2/12 if it changes, mean Q^2/3")
check("Q^2/12 and 7Q^2/12 average to Q^2/3", abs((1 / 12 + 7 / 12) / 2 - 1 / 3) < 1e-12)
for mu, quoted in ((0.25, 3.7), (0.15, 5.6), (1e-9, 9.0)):
    G = -30 * math.log10(0.5 + mu)
    check(f"G(mu) = -30 log10(1/2 + mu) at mu = {mu:g} is {quoted} dB as quoted", abs(G - quoted) < 0.06, f"{G:.3f} dB")
check("G follows from 10 log10[(1/3) / ((1/2+mu)^3/3)]", abs(10 * math.log10((1 / 3) / ((0.65 ** 3) / 3)) + 30 * math.log10(0.65)) < 1e-12)

say("\n== 3. Parseval (orthonormal 8x8 DCT): coefficient change energy equals pixel change energy before rounding")
blk = rng.normal(0, 30, (8, 8)); d = rng.normal(0, 2, (8, 8))
dc = D._dct2(blk + d) - D._dct2(blk)
check("sum of squared coefficient changes equals sum of squared pixel changes", abs((dc ** 2).sum() - (d ** 2).sum()) < 1e-9)

say("\n== 4. Tolerance |e| < mu Q (Eq. tolerance), worked example and Monte Carlo")
c = D._parity_write(23.7, u, v, 1, margin=0.15)  # 26.5, odd bin 3 spans [25, 35]
check("written value 26.5 survives e = -1.49 (just below mu Q = 1.5)", round((c - 1.49) / 10) % 2 == 1)
check("and can flip at e = -1.51 (just above mu Q)", round((c - 1.51) / 10) % 2 == 0)
for mu in (0.05, 0.15, 0.3):
    new = np.array([D._parity_write(cc, u, v, int(b), margin=mu) for cc, b in zip(C0[:30000], bits[:30000])])
    e = rng.uniform(-1, 1, new.size) * mu * Q * 0.999
    check(f"no parity change under any |e| < mu Q, mu = {mu}", np.all((np.round((new + e) / Q).astype(int) & 1) == bits[:30000]))

say("\n== 5. Rounding noise: each coefficient gets variance 1/12, sd 0.289; mu = 0.15 with Q >= 8 leaves >= 1.2, about 4 sd")
y = rng.uniform(0, 255, (4000, 8, 8))
err = np.array([D._dct2(np.round(t)) - D._dct2(t) for t in y])
check("measured sd of coefficient error after pixel rounding", abs(err.std() - math.sqrt(1 / 12)) < 0.005, f"{err.std():.4f} vs {math.sqrt(1/12):.4f}")
check("smallest margin 0.15 x 8 = 1.2 is about 4 sd", abs(1.2 / math.sqrt(1 / 12) - 4.16) < 0.01, f"{1.2/math.sqrt(1/12):.2f} sd")

say("\n== 6. Scaled-table requantisation: parities survive if mu > s/2 (coefficient domain)")
for mu, s in ((0.15, 0.2), (0.15, 0.4), (0.25, 0.4), (0.5, 0.6)):
    new = np.array([D._parity_write(cc, u, v, int(b), margin=mu) for cc, b in zip(C0[:30000], bits[:30000])])
    Qs = s * Q
    rq = np.round(new / Qs) * Qs
    surv = np.mean((np.round(rq / Q).astype(int) & 1) == bits[:30000])
    if mu > s / 2:
        check(f"mu = {mu}, s = {s}: guaranteed, all survive", surv == 1.0, f"{100*surv:.1f}%")
    else:
        say(f"  mu = {mu}, s = {s}: not guaranteed, {100*surv:.1f}% survive (no claim made)")
say("  NOTE: the guarantee is in the coefficient domain. After requantisation the image is rounded to pixels again,")
say("  which adds sd 0.29 per coefficient. At mu = 0.15, s = 0.2 the margin left is (mu - s/2) Q = 0.05 Q (0.4 at Q = 8),")
say("  so pixel-domain survival can be well below 100% (the paper reports 93.5% bits at scale 0.2). The text should say so.")

say("\n== 7. Margin escalation mu_f = min(1/2, mu 2^(f-1)); bin centre after at most 1 + ceil(log2(1/(2 mu))) failures")
for mu, quoted in ((0.15, 3), (0.25, 2)):
    f = 1
    while min(0.5, mu * 2 ** (f - 1)) < 0.5:
        f += 1
    check(f"mu = {mu}: bin centre reached at failure {f}, paper says {quoted}", f == quoted and f == 1 + math.ceil(math.log2(1 / (2 * mu))))

say("\n== 8. Adaptive K (Eq. K) and capacity (Eq. cap), checked against the code")
for cbar, want in ((16.64, 24), (1.57, 1), (0.95, 24), (30.0, 38), (2.4, 3)):
    blocks = np.zeros((32, 32, 8, 8))
    f = int(cbar)
    # build blocks whose eligible count averages cbar: give each block a set number of eligible positions
    counts = np.full(1022, f); counts[: int(round((cbar - f) * 1022))] += 1
    k = 0
    for i in range(32):
        for j in range(32):
            if (i, j) in D._HEADER_BLOCK_SET:
                continue
            for (uu, vv) in D._PAYLOAD_CANDIDATES[: counts[k]]:
                blocks[i, j, uu, vv] = 3 * D._Q_TABLE[uu, vv]
            k += 1
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        K = D._compute_K(blocks)
    check(f"mean eligible count {cbar} gives K = {want}", K == want, f"code gives {K}")
check("non-header blocks of a 256x256 image = 32*32 - 2 = 1022", 32 * 32 - len(D.HEADER_BLOCKS) == 1022)
check("Table 1: mean K 23.92 gives 1022 x 24.92 = 25,468 bits = 0.389 bpp", abs(1022 * 24.92 / 65536 - 0.389) < 0.001, f"{1022*24.92:.0f} bits")
check("Table 1: baseline all covers K 11.88 gives 13,163 bits = 0.201 bpp", abs(1022 * 12.88 - 13163) < 2 and abs(1022 * 12.88 / 65536 - 0.201) < 0.001)
check("39 positions in M (Q >= 8, not DC)", len(D._PAYLOAD_CANDIDATES) == 39 and D.MAX_K == 38)
check("matched payload: n = 19,680 bits gives K = ceil(n/1022) - 1 = 19", math.ceil(19680 / 1022) - 1 == 19)
check("matched payload: n = 39,304 bits gives K = 38", math.ceil(39304 / 1022) - 1 == 38)

say("\n== 9. Header: 16 + 24 + 6 + 4 + 14 = 64 bits, CRC over 50 bits padded to 56")
hb = D._pack_header(31280, 30)
pl, k6, ver, ok = D._unpack_header(hb)
check("header round trip (n = 31,280, K = 30)", len(hb) == 64 and pl == 31280 and k6 == 30 and ok)
bad = hb.copy(); bad[20] ^= 1
check("one flipped length bit is caught by the CRC", not D._unpack_header(bad)[3])

say("\n== 10. Robust mode: n_s = 1022 x 39 = 39,858 slots; frame L = 8m + 32; copies = n_s / L")
check("39,858 slots", 1022 * 39 == 39858)
for bpp, quoted in ((0.005, 110.7), (0.02, 29.7), (0.6, 1.0)):
    m = round(bpp * 65536 / 8)
    cp = 39858 / (8 * m + 32)
    check(f"net {bpp} bpp: m = {m} bytes, {cp:.1f} copies, quoted {quoted}", abs(cp - quoted) < 0.2)

say("\n== 11. Reliability weight: log odds = ((1-|f|)^2 - f^2) Q^2 / (2 sigma^2) = (1 - 2|f|) Q^2/(2 sigma^2) = a Q^2/(2 sigma^2)")
for f in (0.0, 0.1, 0.33, 0.49):
    check(f"identity at |f| = {f}", abs(((1 - f) ** 2 - f ** 2) - (1 - 2 * f)) < 1e-12)
Qr, sig = 8.0, 2.5
for f in (0.05, 0.25, 0.45):
    lo = math.log(math.exp(-(f * Qr) ** 2 / (2 * sig ** 2)) / math.exp(-((1 - f) * Qr) ** 2 / (2 * sig ** 2)))
    check(f"Gaussian log odds at |f| = {f} equal a Q^2/(2 sigma^2)", abs(lo - (1 - 2 * f) * Qr ** 2 / (2 * sig ** 2)) < 1e-9)

say("\n== 12. Payload units")
check("0.30 bpp of a 256x256 image = 19,661 bits (paper: 19,680 achieved)", abs(0.3 * 65536 - 19661) < 1)
check("67-character alphabet: log2(67) = 6.07 bits per byte, as quoted", abs(math.log2(67) - 6.07) < 0.005)

say(f"\nSUMMARY: {len(fails)} failure(s)" + ("" if not fails else ": " + "; ".join(fails)))
open(os.path.join(OUT, "derivations_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
