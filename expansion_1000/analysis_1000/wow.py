"""WOW (Holub & Fridrich, WIFS 2012) costs and payload-limited simulation, following the reference
MATLAB code: three directional residuals R_k = X * F_k with the Daubechies 8-tap filter bank used by
S-UNIWARD, embedding suitability xi_k = |R_k| * rot180(|F_k|), costs aggregated with p = -1,
rho = (sum_k xi_k^p)^(-1/p). Alignment follows common.suniward_costs (shift (-1,-1)); verified below
against a brute-force evaluation of xi_k from its definition."""
import numpy as np
from scipy.signal import convolve2d
import common as C

def wow_costs(cover, p=-1.0):
    wet = 1e10
    x = cover.astype(np.float64); pad = 16
    xp = np.pad(x, pad, mode="symmetric"); acc = np.zeros_like(x)
    for F in C._F:
        R = convolve2d(xp, F, mode="same")
        xi = convolve2d(np.abs(R), np.rot90(np.abs(F), 2), mode="same")
        xi = np.roll(xi, (-1, -1), axis=(0, 1))[pad:-pad, pad:-pad]
        acc += xi ** p
    rho = acc ** (-1.0 / p)
    rho[rho > wet] = wet; rho[np.isnan(rho)] = wet
    rp = rho.copy(); rm = rho.copy(); rp[x == 255] = wet; rm[x == 0] = wet
    return rp, rm

def wow(y, m_bits, seed):
    rp, rm = wow_costs(y)
    return C.embed_simulator(y, rp, rm, m_bits, np.random.default_rng(seed))

def _bruteforce_xi(x, F, i, j, pad=16):
    """xi_k(i,j) from the definition: sum over residual samples of |R| weighted by the magnitude of the
    change a unit change of pixel (i,j) causes in that residual sample."""
    xp = np.pad(x.astype(np.float64), pad, mode="symmetric")
    R0 = convolve2d(xp, F, mode="same")
    xq = xp.copy(); xq[i + pad, j + pad] += 1
    D = convolve2d(xq, F, mode="same") - R0
    return float((np.abs(R0) * np.abs(D)).sum())

if __name__ == "__main__":
    rng = np.random.default_rng(0); x = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    worst = 0.0
    for F in C._F:
        pad = 16; xp = np.pad(x.astype(np.float64), pad, mode="symmetric")
        R = convolve2d(xp, F, mode="same")
        xi = np.roll(convolve2d(np.abs(R), np.rot90(np.abs(F), 2), mode="same"), (-1, -1), axis=(0, 1))[pad:-pad, pad:-pad]
        for (i, j) in [(20, 20), (31, 40), (25, 33), (40, 22)]:
            b = _bruteforce_xi(x, F, i, j); worst = max(worst, abs(xi[i, j] - b) / b)
    print("max relative error of fast xi against brute force: %.2e" % worst)
