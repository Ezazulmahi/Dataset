r"""J-MiPOD (R. Cogranne, Q. Giboulot, P. Bas, "Steganography by minimizing statistical detectability: the cases
of JPEG and color images", ACM IH&MMSec 2020), simulated embedding for one grayscale JPEG channel.

conseal 2025.11 has J-UNIWARD but no J-MiPOD. This reimplementation reuses conseal's MiPOD building blocks
(Innsbruck; mipod._defs.wiener2, mipod._costmap.estimate_variance, mipod._defs.ternary_probs), so only the
JPEG step is new:
  1. pixel residual of the decompressed cover: x - wiener2(x, 2x2); its local variance by the 2-D DCT polynomial
     model with a 3x3 neighbourhood and degree 3 (as in the J-MiPOD paper), floored at 0.01;
  2. variance of each DCT coefficient of each 8x8 block: sum over the block's pixels of the pixel variance times
     the squared DCT basis function, divided by the squared quantisation step; floored at 1e-5;
  3. Fisher information 1 / variance^2; change rates beta from the ternary payload constraint
     alpha * nzAC bits (same Lagrangian search as MiPOD); +1 and -1 each with probability beta.
Coefficients with |c| >= 1023 are wet. Describe it as a reimplementation in the paper.
"""
import numpy as np
from conseal.mipod import _defs as _md
from conseal.mipod._costmap import estimate_variance

def _dct_mat():
    D = np.zeros((8, 8))
    for k in range(8):
        for n in range(8):
            D[k, n] = (np.sqrt(1 / 8) if k == 0 else np.sqrt(2 / 8)) * np.cos(np.pi * (2 * n + 1) * k / 16)
    return D

def fisher_information(x0, qt):
    """x0: decompressed cover (H, W) float; qt: (8, 8) quantisation table. Returns (H/8, W/8, 8, 8)."""
    x0 = x0.astype(float)
    res = x0 - _md.wiener2(x0, kernel_size=(2, 2))
    var = np.clip(estimate_variance(res, block_size=3, degree=3), 0.01, None)
    H, W = var.shape; D2 = _dct_mat() ** 2
    vb = var.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)                 # [bi, bj, m, n]
    vdct = np.einsum("km,ln,abmn->abkl", D2, D2, vb) / qt.astype(float) ** 2
    return 1.0 / np.clip(vdct, 1e-5, None) ** 2

def probability(x0, y0, qt, alpha):
    fi = fisher_information(x0, qt)
    nzac = int(np.count_nonzero(y0) - np.count_nonzero(y0[:, :, 0, 0]))
    payload = alpha * nzac * np.log(2)                                           # nats
    beta = _md.ternary_probs(fi, payload, max_num_iterations=60, excess_values_num_iter=20)
    beta = beta.reshape(fi.shape, order="F")
    beta[np.abs(y0) >= 1023] = 0
    return beta, nzac

def simulate_single_channel(x0, y0, qt, alpha, seed=None):
    """Returns the stego DCT coefficients (same shape as y0)."""
    beta, _ = probability(x0, y0, qt, alpha)
    rng = np.random.default_rng(seed); u = rng.random(y0.shape)
    delta = np.where(u < beta, 1, np.where(u < 2 * beta, -1, 0))
    return (y0 + delta).astype(y0.dtype)
