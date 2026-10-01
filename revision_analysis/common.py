"""Shared helpers for the revision analyses (cover statistics, SPAM features, S-UNIWARD simulator)."""
import numpy as np
import cv2
from scipy.fft import dctn
from scipy.signal import convolve2d

RES = r"F:\Results"
BASE_DIR = RES + r"\baseline"
FT_DIR = RES + r"\finetuned"
STEGO_FT_DIR = RES + r"\stego_finetuned"   # stego images produced by the authors' comparison.py

Q_TABLE = np.array([
    [3, 2, 2, 3, 4, 6, 8, 10],
    [2, 2, 3, 4, 5, 9, 10, 9],
    [3, 3, 4, 5, 6, 9, 11, 9],
    [3, 4, 5, 6, 8, 14, 13, 10],
    [4, 5, 7, 9, 11, 17, 16, 12],
    [5, 7, 9, 10, 13, 17, 18, 15],
    [10, 13, 12, 14, 16, 19, 19, 17],
    [14, 17, 18, 18, 19, 18, 19, 17]], dtype=np.float64)
STABLE = (Q_TABLE >= 8)
STABLE[0, 0] = False
assert STABLE.sum() == 39


def read_y(path):
    """Luminance exactly as the embedder reads it (OpenCV 8-bit BGR->YCrCb)."""
    bgr = cv2.imread(path)
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0]


def block_dct(y):
    h, w = y.shape
    b = (y.astype(np.float64) - 128.0).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return dctn(b, axes=(-2, -1), norm="ortho")          # (nr, nc, 8, 8)


def cover_stats(y):
    """Eligible-count statistics and adaptive K exactly as DCT_Adaptive._compute_K."""
    c = block_dct(y)
    q = np.rint(c / Q_TABLE)
    elig = (np.abs(q) >= 2) & STABLE                      # (nr, nc, 8, 8)
    counts = elig.sum(axis=(-2, -1)).astype(float)
    mask = np.ones(counts.shape, bool)
    mask[0, 0] = mask[0, 1] = False                       # header blocks
    mean_cnt = counts[mask].mean()
    mean_int = int(mean_cnt)
    K = 24 if mean_int <= 0 else max(1, min(38, mean_int + mean_int // 2))
    n_payload = int(mask.sum())
    return {"mean_eligible": mean_cnt, "K": K, "fallback": mean_int <= 0,
            "capacity_bits": n_payload * (K + 1),
            "elig_map": elig[mask].mean(axis=0),          # per-position eligibility rate
            "energy_map": (c[mask] ** 2).mean(axis=0)}


# ----------------------------------------------------------------------------
# SPAM-686 features (Pevny, Bas, Fridrich, IEEE TIFS 2010), second order, T = 3
# ----------------------------------------------------------------------------
def _spam_dir(x, dy, dx, T=3):
    """Second-order transition probabilities of truncated differences along (dy, dx)."""
    h, w = x.shape
    # views for positions p, p+d, p+2d, p+3d
    y0 = max(0, -3 * dy); y1 = h - max(0, 3 * dy)
    x0 = max(0, -3 * dx); x1 = w - max(0, 3 * dx)
    def v(k):
        return x[y0 + k * dy:y1 + k * dy, x0 + k * dx:x1 + k * dx]
    d1 = np.clip(v(0) - v(1), -T, T)
    d2 = np.clip(v(1) - v(2), -T, T)
    d3 = np.clip(v(2) - v(3), -T, T)
    n = 2 * T + 1
    idx = ((d1 + T) * n + (d2 + T)) * n + (d3 + T)
    joint = np.bincount(idx.ravel(), minlength=n ** 3).astype(np.float64).reshape(n, n, n)
    denom = joint.sum(axis=2, keepdims=True)
    return np.divide(joint, denom, out=np.zeros_like(joint), where=denom > 0).ravel()   # P(d3 | d1, d2)


def spam686(y):
    x = y.astype(np.int32)
    hv = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dg = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    f1 = np.mean([_spam_dir(x, a, b) for a, b in hv], axis=0)
    f2 = np.mean([_spam_dir(x, a, b) for a, b in dg], axis=0)
    return np.concatenate([f1, f2])                      # 686-D


# ----------------------------------------------------------------------------
# S-UNIWARD (Holub, Fridrich, Denemark, EURASIP JIS 2014): costs + payload-limited
# ternary embedding simulator, following the authors' reference MATLAB code.
# ----------------------------------------------------------------------------
_HPDF = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053,
                  -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539,
                  -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494,
                  -0.0001174768])
_LPDF = ((-1.0) ** np.arange(16)) * _HPDF[::-1]
assert abs(_LPDF.sum() - np.sqrt(2)) < 1e-6 and abs(_HPDF.sum()) < 1e-6 and abs((_HPDF ** 2).sum() - 1) < 1e-6
_F = [np.outer(_LPDF, _HPDF), np.outer(_HPDF, _LPDF), np.outer(_HPDF, _HPDF)]


def suniward_costs(cover, sgm=1.0):
    wet = 1e10
    x = cover.astype(np.float64)
    pad = 16
    xp = np.pad(x, pad, mode="symmetric")
    xi_sum = np.zeros_like(x)
    for F in _F:
        R = convolve2d(xp, F, mode="same")
        xi = convolve2d(1.0 / (np.abs(R) + sgm), np.rot90(np.abs(F), 2), mode="same")
        # Even-sized filters: the reference MATLAB code circshifts by (+1,+1); SciPy's
        # 'same' centring differs from MATLAB's by one sample per convolution, so the
        # equivalent shift here is (-1,-1). Verified against a brute-force evaluation of
        # the UNIWARD distortion (max relative error 1e-14).
        xi = np.roll(xi, (-1, -1), axis=(0, 1))
        xi_sum += xi[pad:-pad, pad:-pad]
    rho = xi_sum.copy()
    rho[rho > wet] = wet
    rho[np.isnan(rho)] = wet
    rho_p1 = rho.copy(); rho_m1 = rho.copy()
    rho_p1[x == 255] = wet
    rho_m1[x == 0] = wet
    return rho_p1, rho_m1


def _ternary_entropy(pp, pm):
    p0 = 1 - pp - pm
    P = np.stack([p0, pp, pm])
    P = P[P > 0]
    return float(-(P * np.log2(P)).sum())


def embed_simulator(cover, rho_p1, rho_m1, m_bits, rng):
    """Simulate optimal ternary embedding of m_bits (payload-limited sender)."""
    def probs(lam):
        ep = np.exp(-lam * rho_p1); em = np.exp(-lam * rho_m1)
        z = 1 + ep + em
        return ep / z, em / z
    l3 = 1e3; m3 = float(m_bits + 1); it = 0
    while m3 > m_bits:
        l3 *= 2
        m3 = _ternary_entropy(*probs(l3)); it += 1
        if it > 10:
            break
    l1, m1 = 0.0, float(cover.size) * np.log2(3)
    lam = l3
    for _ in range(40):
        lam = l1 + (l3 - l1) / 2
        m2 = _ternary_entropy(*probs(lam))
        if m2 < m_bits:
            l3, m3 = lam, m2
        else:
            l1, m1 = lam, m2
        if abs(m2 - m_bits) / cover.size < 1e-6:
            break
    pp, pm = probs(lam)
    r = rng.random(cover.shape)
    stego = cover.astype(np.int32).copy()
    stego[r < pp] += 1
    stego[(r >= pp) & (r < pp + pm)] -= 1
    return np.clip(stego, 0, 255).astype(np.uint8)


def suniward(y, m_bits, seed):
    rp, rm = suniward_costs(y)
    return embed_simulator(y, rp, rm, m_bits, np.random.default_rng(seed))


def lsb_matching(y, m_bits, seed):
    """Non-adaptive +-1 embedding (LSB matching) at the same relative payload, simulated
    as random +-1 changes with the ternary-optimal change rate for m_bits."""
    ones = np.ones(y.shape)
    rp = ones.copy(); rm = ones.copy()
    rp[y == 255] = 1e10; rm[y == 0] = 1e10
    return embed_simulator(y, rp, rm, m_bits, np.random.default_rng(seed))
