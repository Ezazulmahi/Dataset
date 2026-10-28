"""
DCT_Adaptive_v5.py  -  DEDS embedder v5, single self-contained file.

    embed(cover_bgr, message, mode="normal")   v4 method (default): margin-projected
                                               parity write, adaptive K, highest PSNR,
                                               exact recovery on a lossless channel
    embed(cover_bgr, message, mode="robust")   keyed dispersion + framed repetition
                                               filling every slot + soft voting;
                                               survives requantisation / noise / JPEG
                                               95-90 at reduced payload
    embed(cover_bgr, message, mode="secure")   syndrome-trellis coding with a texture/
                                               saturation-weighted cost (experimental)
    extract(stego_bgr)                         reads the mode from the header (v3 and
                                               v4 images included)

The derivations are in v5_derivation.md.  Sections below: (1) v4 normal mode,
(2) secure mode, (3) robust mode, (4) public API.
"""
import hashlib
# ════ (1) normal mode (v4) ════
'''
DCT_Adaptive_v4.py  —  DSTG v4: Self-Contained Adaptive DCT Steganography (Y channel)
=====================================================================================
v4 change (only the WRITE rule; extraction is identical to v3, so v3 stego
images still extract):
  v3 snapped every payload coefficient to the centre of its quantisation bin
  (q*Q), even when its parity already matched.  That adds error up to Q/2 to
  every slot.  v4 moves a coefficient only as far as needed: it is projected
  onto the part of the bin of the wanted parity that lies at least
  PAYLOAD_MARGIN*Q from the bin edges.  PAYLOAD_MARGIN = 0.5 reproduces v3
  exactly; smaller values give higher PSNR and less margin against
  requantisation.  The header keeps the v3 rule (HEADER_MARGIN = 0.5) because
  one wrong header bit loses the whole message.
embed()   : cover BGR + message  →  stego BGR
extract() : stego BGR            →  message  (no key, no sidecar)

Design
------
  Channel     : Luminance (Y) channel of YCrCb.  Bits are embedded in the DCT
                of Y.  Because OpenCV's integer BGR↔YCrCb conversion is lossy
                (±1 on a few hundred pixels), the stego image is written by
                _exact_y_repair: it starts from the high-PSNR naive YCrCb→BGR
                inverse and repairs ONLY the 8×8 blocks whose embedded bits would
                otherwise read wrong after the BGR→YCrCb round-trip.  This keeps
                colour distortion minimal while guaranteeing that extract()'s
                BGR→YCrCb read returns every embedded bit exactly — 100% accurate
                extraction, no key and no sidecar, at any message size.  The
                grayscale identity (v,v,v) → Y=v (4899+9617+1868 = 16384 = 2**14)
                guarantees every target Y is reachable for the repair.

  DCT         : 8×8 blocks, level-shift pixel−128, 2-D ortho DCT.
  Quantisation: q = round(coeff / Q[u,v])  (JPEG luminance Q_TABLE).
  Eligibility : Q[u,v] ≥ 8  (stable positions only; 39 positions per block).
  Freq order  : Eligible positions sorted by zigzag index (mid-freq first).
  Adaptive K  : mean (int) of eligible mid-freq counts + mean//2;
                fallback FALLBACK_K.  Max K = 38 (all stable positions).
  LSB embed   : quantised LSB (q & 1), min-distortion direction (flip_dir=-1).
  Placement   : distortion-ordered slots — lowest-Q coefficients across all
                blocks are filled first, maximising PSNR at a given payload.
                The order depends only on K and image size (both known to the
                decoder), so no side information is needed.
  Verify-fix  : after IDCT→uint8→re-DCT, every embedded bit is verified and
                corrected in the Y pixel domain.  The colour round-trip is
                handled exactly by _exact_y_repair, so this loop sees only
                IDCT rounding and converges in ≤ 12 iterations at any size.
  Header      : 64 bits across 2 blocks (0,0)(0,1) = 32 bits each.
                Layout: magic(16) | payload_bits(24) | K(6) |
                        version(4) | CRC14(14).

  PSNR notes  : Typical PSNR for the reference 256×256 image (Y channel,
                selective repair):
                  530 chars  → ~44 dB     2189 chars → ~36 dB
                  820 chars  → ~42 dB     3520 chars → ~31 dB
                PSNR is above 40 dB up to roughly 0.10 BPP; at higher payloads
                it falls as more coefficients (and repair pixels) must change.
'''

import os, zlib
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global constants ─────────────────────────────────────────────────────────
N                     = 8
VERSION               = 4                  # v4: margin-projected parity write
PAYLOAD_MARGIN        = 0.15               # fraction of Q kept clear of the bin edges (0.5 = v3); chosen in the pilot
HEADER_MARGIN         = 0.5                # header stays on bin centres, as in v3
PREFER_ELIGIBLE       = False              # v3 rule (prefer |q'| >= 2 when flipping) off: +5.5 to 6.8 dB in the pilot
FALLBACK_K            = 24
ELIGIBILITY_THRESHOLD = 2
MAGIC_SHORT           = 0x4447            # 'DG'
HEADER_BLOCKS         = [(0, 0), (0, 1)]
HEADER_BLOCK_BITS     = 32
HEADER_TOTAL_BITS     = 64
MAX_FIX_ITERATIONS    = 60                 # v4: passes are cheap (vectorised DCT)

# JPEG luminance quantisation table (quality ≈85)
_Q_TABLE = np.array([
    [ 3,  2,  2,  3,  4,  6,  8, 10],
    [ 2,  2,  3,  4,  5,  9, 10,  9],
    [ 3,  3,  4,  5,  6,  9, 11,  9],
    [ 3,  4,  5,  6,  8, 14, 13, 10],
    [ 4,  5,  7,  9, 11, 17, 16, 12],
    [ 5,  7,  9, 10, 13, 17, 18, 15],
    [10, 13, 12, 14, 16, 19, 19, 17],
    [14, 17, 18, 18, 19, 18, 19, 17],
], dtype=np.float32)

# ── Zigzag scan for 8×8 ──────────────────────────────────────────────────────
def _build_zigzag():
    order = []
    for s in range(2 * N - 1):
        if s % 2 == 0:
            r = min(s, N - 1); c = s - r
            while r >= 0 and c < N:
                order.append((r, c)); r -= 1; c += 1
        else:
            c = min(s, N - 1); r = s - c
            while c >= 0 and r < N:
                order.append((r, c)); r += 1; c -= 1
    return order

_ZIGZAG = _build_zigzag()
_ZZ_IDX = {uv: i for i, uv in enumerate(_ZIGZAG)}

# ── Stable positions: non-DC, Q≥8 (39 positions) ────────────────────────────
_STABLE = [
    (u, v)
    for u in range(N) for v in range(N)
    if not (u == 0 and v == 0) and _Q_TABLE[u, v] >= 8
]

_PAYLOAD_CANDIDATES = sorted(_STABLE, key=lambda uv: _ZZ_IDX[uv])
MAX_K = len(_PAYLOAD_CANDIDATES) - 1

ZZ_START = _ZZ_IDX[_PAYLOAD_CANDIDATES[0]]
ZZ_END   = _ZZ_IDX[_PAYLOAD_CANDIDATES[-1]]

_HEADER_POSITIONS = sorted(
    _STABLE,
    key=lambda uv: (-float(_Q_TABLE[uv[0]][uv[1]]), uv[0], uv[1])
)[:HEADER_BLOCK_BITS]

_HEADER_BLOCK_SET = set(HEADER_BLOCKS)

# ── DCT helpers ──────────────────────────────────────────────────────────────
def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def _to_blocks(channel: np.ndarray):
    h, w   = channel.shape
    h8, w8 = h - h % N, w - w % N
    nr, nc = h8 // N, w8 // N
    ch     = channel[:h8, :w8].astype(np.float64) - 128.0
    tiles  = ch.reshape(nr, N, nc, N).transpose(0, 2, 1, 3)        # (nr, nc, N, N)
    blocks = dct(dct(tiles, axis=2, norm='ortho'), axis=3, norm='ortho')
    return blocks, h8, w8

def _idct_tiles(blocks: np.ndarray) -> np.ndarray:
    """Unrounded, unclipped pixel values (level shift added) of every block."""
    return idct(idct(blocks, axis=2, norm='ortho'), axis=3, norm='ortho') + 128.0

def _from_blocks(blocks: np.ndarray, orig: np.ndarray, h8: int, w8: int) -> np.ndarray:
    out    = orig.copy()
    nr, nc = blocks.shape[:2]
    patch  = np.clip(np.round(_idct_tiles(blocks)), 0, 255).astype(np.uint8)
    out[:h8, :w8] = patch.transpose(0, 2, 1, 3).reshape(nr * N, nc * N)
    return out

# ── Quantised LSB helpers ────────────────────────────────────────────────────
def _qi(coeff: float, u: int, v: int) -> int:
    return int(np.round(float(coeff) / float(_Q_TABLE[u, v])))

def _lsb_read(coeff: float, u: int, v: int) -> int:
    return _qi(coeff, u, v) & 1

def _slot_arrays(slots):
    a = np.asarray(slots, dtype=np.int64).reshape(-1, 4)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], _Q_TABLE[a[:, 2], a[:, 3]].astype(np.float64)

def _parities(blocks, arrs):
    I, J, U, V, Qs = arrs
    return (np.round(blocks[I, J, U, V] / Qs).astype(np.int64) & 1)

def _lsb_write(coeff: float, u: int, v: int, bit: int, flip_dir: int = -1) -> float:
    Q     = float(_Q_TABLE[u, v])
    q     = _qi(coeff, u, v)
    if (q & 1) == bit:
        return float(q) * Q
    q_up, q_dn = q + 1, q - 1
    up_ok = abs(q_up) >= ELIGIBILITY_THRESHOLD
    dn_ok = abs(q_dn) >= ELIGIBILITY_THRESHOLD
    if up_ok and dn_ok:
        if flip_dir == 1:   return float(q_up) * Q
        if flip_dir == 0:   return float(q_dn) * Q
        return (float(q_up) if abs(q_up * Q - coeff) <= abs(q_dn * Q - coeff)
                else float(q_dn)) * Q
    if up_ok:  return float(q_up) * Q
    if dn_ok:  return float(q_dn) * Q
    return float(q_up if q >= 0 else q_dn) * Q

def _parity_write(coeff: float, u: int, v: int, bit: int,
                  margin: float = None) -> float:
    """
    Minimal-change parity write.  The coefficient is moved to the nearest point
    whose quantised value q' has parity `bit` and that lies at least margin*Q
    inside the bin of q'.  If the current bin already has the right parity the
    coefficient stays in it (and is only pulled in from the edge if needed).
    Bin choice when a flip is needed follows _lsb_write: prefer |q'| >= 2,
    then the nearer of q+1 and q-1.  margin = 0.5 gives exactly _lsb_write.
    """
    if margin is None:
        margin = PAYLOAD_MARGIN
    Q = float(_Q_TABLE[u, v])
    m = min(max(float(margin), 1e-3), 0.5) * Q
    c = float(coeff)
    q = _qi(c, u, v)

    def proj(qc):
        return min(max(c, (qc - 0.5) * Q + m), (qc + 0.5) * Q - m)

    if (q & 1) == bit:
        return proj(q)
    q_up, q_dn = q + 1, q - 1
    up_ok = abs(q_up) >= ELIGIBILITY_THRESHOLD or not PREFER_ELIGIBLE
    dn_ok = abs(q_dn) >= ELIGIBILITY_THRESHOLD or not PREFER_ELIGIBLE
    if up_ok and dn_ok:
        a, b = proj(q_up), proj(q_dn)
        return a if abs(a - c) <= abs(b - c) else b
    if up_ok:  return proj(q_up)
    if dn_ok:  return proj(q_dn)
    return proj(q_up if q >= 0 else q_dn)

# ── PRNG for LSB-matching direction ──────────────────────────────────────────
def _make_prng(h: int, w: int):
    seed = (int(h) * 0x9e3779b9 ^ int(w) * 0x6c62272e) & 0xFFFFFFFF
    seed = seed or 0xDEADBEEF
    state = [seed]
    def _next() -> int:
        s = state[0]
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17)
        s ^= (s <<  5) & 0xFFFFFFFF
        state[0] = s
        return (s >> 16) & 1
    return _next

# ── Position selection ───────────────────────────────────────────────────────
def _payload_positions(K: int) -> list:
    return _PAYLOAD_CANDIDATES[:K + 1]

def _build_slot_order(K: int, nr: int, nc: int) -> list:
    """
    Deterministic distortion-ordered list of payload slots (i, j, u, v).

    PSNR optimisation (v2.1): instead of filling each block with all K+1
    coefficients in raster order, we fill the LOWEST-distortion slots across
    ALL blocks first.  The distortion of changing coefficient (u,v) scales
    with its quantisation step Q[u,v], so slots are sorted by Q ascending,
    then by zigzag rank, then by block raster index.

    This ordering depends ONLY on K and the image dimensions (nr, nc) — both
    of which the decoder recovers from the header — so embed() and extract()
    reconstruct the identical order with no side information.

    Returns a list of (i, j, u, v) tuples in fill order.
    """
    positions = _payload_positions(K)
    slots = []
    for pos_rank, (u, v) in enumerate(positions):
        q_val = float(_Q_TABLE[u, v])
        for i in range(nr):
            for j in range(nc):
                if (i, j) in _HEADER_BLOCK_SET:
                    continue
                slots.append((q_val, pos_rank, i, j, u, v))
    slots.sort(key=lambda s: (s[0], s[1], s[2], s[3]))
    return [(i, j, u, v) for (_, _, i, j, u, v) in slots]

def _compute_K(blocks: np.ndarray) -> int:
    nr, nc = blocks.shape[:2]
    counts = [
        sum(1 for u, v in _PAYLOAD_CANDIDATES
            if abs(_qi(blocks[i, j, u, v], u, v)) >= ELIGIBILITY_THRESHOLD)
        for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    ]
    if not counts:
        return FALLBACK_K
    mean_val = int(np.mean(counts))
    if mean_val <= 0:
        return FALLBACK_K
    extra = mean_val // 2
    K     = max(1, min(MAX_K, mean_val + extra))
    print(f"  K-stats  : mean={mean_val}  extra={extra}  K={K}  (freqs 1-{K+1})")
    return K

# ── Header ────────────────────────────────────────────────────────────────────
def _int_to_bits(value: int, n_bits: int) -> list:
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

def _bits_to_int(bits: list) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (int(b) & 1)
    return v

def _pack_header(n_message_bits: int, K: int) -> list:
    magic   = MAGIC_SHORT
    payload = n_message_bits & 0xFFFFFF
    k6      = K & 0x3F
    ver4    = VERSION & 0xF
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc14   = zlib.crc32(body) & 0x3FFF
    return (
        _int_to_bits(magic,   16)
        + _int_to_bits(payload, 24)
        + _int_to_bits(k6,       6)
        + _int_to_bits(ver4,     4)
        + _int_to_bits(crc14,   14)
    )

def _unpack_header(bits: list) -> tuple:
    magic   = _bits_to_int(bits[0:16])
    payload = _bits_to_int(bits[16:40])
    k6      = _bits_to_int(bits[40:46])
    ver4    = _bits_to_int(bits[46:50])
    crc14r  = _bits_to_int(bits[50:64])
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc_ok  = (zlib.crc32(body) & 0x3FFF) == crc14r
    magic_ok = magic == MAGIC_SHORT
    return payload, k6, ver4, (crc_ok and magic_ok)

def _write_header(blocks: np.ndarray, n_message_bits: int, K: int) -> None:
    bits = _pack_header(n_message_bits, K)
    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            blocks[bi, bj, u, v] = _lsb_write(
                blocks[bi, bj, u, v], u, v, chunk[k], flip_dir=-1
            )

def _read_header(blocks: np.ndarray) -> tuple:
    bits = []
    for bi, bj in HEADER_BLOCKS:
        for u, v in _HEADER_POSITIONS:
            bits.append(_lsb_read(blocks[bi, bj, u, v], u, v))
    return _unpack_header(bits)

# ── Verify-fix pass (Y channel: only IDCT roundtrip, no colour loss) ─────────
def _verify_and_fix(slot_order: list,
                    current_y: np.ndarray,
                    h8: int, w8: int,
                    header_bits: list,
                    payload_bits: list,
                    K: int,
                    state: dict = None) -> tuple:
    """
    Verify and fix embedded bits after IDCT→uint8→re-DCT roundtrip.

    This loop runs entirely in the luminance (Y) pixel domain, so the only
    source of bit-flip is IDCT→uint8 rounding.  It converges reliably in
    ≤ 12 iterations for any message size.  The lossy BGR↔YCrCb colour
    round-trip is handled separately (and exactly) by _exact_y_repair at
    the end of embed(), so it never enters this loop.

    Distortion is minimised by always re-writing flipped bits with
    flip_dir=-1 (choose the q±1 candidate nearest the current coefficient).

    `slot_order` is the distortion-ordered list of (i,j,u,v) payload slots
    from _build_slot_order; payload_bits[k] lives in slot_order[k].

    Returns (new_y_uint8, n_remaining_errors_after_roundtrip).
    """
    # Re-DCT the current uint8 Y channel — exactly what extract() will see.
    blk, _, _ = _to_blocks(current_y)

    # ── Fix header (min-distortion direction) ────────────────────────────────
    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            expected = chunk[k]
            if _lsb_read(blk[bi, bj, u, v], u, v) != expected:
                blk[bi, bj, u, v] = _lsb_write(
                    blk[bi, bj, u, v], u, v, expected, flip_dir=-1
                )

    # ── Fix payload (slot order, min-distortion direction) ───────────────────
    # v4: a slot that reads wrong again gets a larger margin, doubling with each
    # failure up to the bin centre (the v3 rule).  This is what makes small
    # margins converge on covers with many pixels clipped at 0/255.  Slots that
    # never fail keep PAYLOAD_MARGIN.  (A variant that also offset the write by
    # the clipping loss of the previous pass oscillated on saturated covers and
    # was dropped; see results/compensation_test.txt.)
    if state is None:
        state = {}
    fails    = state.setdefault('fails', {})
    n_payload_bits = len(payload_bits)
    arrs = state.get('arrs')
    if arrs is None:
        arrs = state['arrs'] = _slot_arrays(slot_order[:n_payload_bits])
    want = np.asarray(payload_bits, dtype=np.int64)
    for k in np.nonzero(_parities(blk, arrs) != want)[0]:
        i, j, u, v = slot_order[k]
        f = fails.get(k, 0) + 1
        fails[k] = f
        read = float(blk[i, j, u, v])
        new  = _parity_write(read, u, v, int(want[k]),
                             margin=min(0.5, PAYLOAD_MARGIN * 2.0 ** (f - 1)))
        blk[i, j, u, v] = new

    # Reconstruct Y channel from fixed DCT blocks.
    new_y = _from_blocks(blk, current_y, h8, w8)

    # ── Count remaining errors AFTER the IDCT roundtrip ─────────────────────
    # This is the definitive check: does extract() get the right bits?
    blk_v, _, _ = _to_blocks(new_y)
    n_remaining = 0

    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            if _lsb_read(blk_v[bi, bj, u, v], u, v) != chunk[k]:
                n_remaining += 1

    n_remaining += int(np.sum(_parities(blk_v, arrs) != want))

    return new_y, n_remaining

# ── Public API ───────────────────────────────────────────────────────────────
# ── Exact-Y BGR reconstruction (selective, distortion-minimal) ───────────────
# OpenCV's 8-bit BGR→YCrCb luma:  Y = (4899*R + 9617*G + 1868*B + 8192) >> 14.
# Because 4899 + 9617 + 1868 = 16384 = 2**14, a grayscale pixel (v, v, v) maps to
# exactly Y = v, so every target Y is reachable.
#
# The naive YCrCb→BGR inverse keeps colour very close to the cover (high PSNR)
# but lets Y drift by ±1 on a few hundred pixels.  The vast majority of those
# drifts are harmless — they do not change any embedded DCT LSB.  Repairing every
# embedded pixel (the brute-force approach) needlessly destroys ~10 dB of PSNR.
#
# _exact_y_repair therefore repairs ONLY the pixels inside 8×8 blocks where a
# header or payload bit is actually read wrong after the BGR round-trip, leaving
# the high-PSNR naive conversion intact everywhere else.  It loops until every
# embedded bit reads correctly through BGR→YCrCb (guaranteeing 100% extraction).
def _set_pixel_y_minl2(B, G, R, pr, pc, target_y, max_radius):
    """
    Move BGR pixel (pr,pc) to the colour NEAREST (min L2) to its current value that
    yields OpenCV luma == target_y, searching a window of the given radius.

    Vectorised version of the original triple loop (kept below as
    _set_pixel_y_minl2_loop): for each radius the whole (dR, dB) square is
    evaluated at once, in the same order (dR outer, dB inner, floor(G) before
    ceil(G)), and np.argmin returns the first minimum, so the chosen colour,
    including tie-breaks, is identical to the loop.

    Returns True if a solution was found within `max_radius`, else False.
    """
    b, g, r = int(B[pr, pc]), int(G[pr, pc]), int(R[pr, pc])
    for rad in range(1, max_radius + 1):
        d = np.arange(-rad, rad + 1)
        rr = (r + d)[:, None].repeat(len(d), 1).ravel()            # dR outer
        bb = (b + d)[None, :].repeat(len(d), 0).ravel()            # dB inner
        ok_rb = (rr >= 0) & (rr <= 255) & (bb >= 0) & (bb <= 255)
        gi = (target_y * 16384 + 8192 - 4899 * rr - 1868 * bb) / 9617.0
        gg = np.stack([np.floor(gi), np.ceil(gi)], axis=1).astype(np.int64)   # floor then ceil
        rr2, bb2 = rr[:, None], bb[:, None]
        valid = ok_rb[:, None] & (gg >= 0) & (gg <= 255) &                 (((4899 * rr2 + 9617 * gg + 1868 * bb2 + 8192) >> 14) == target_y)
        if valid.any():
            cost = (rr2 - r) ** 2 + (bb2 - b) ** 2 + (gg - g) ** 2
            cost = np.where(valid, cost, np.iinfo(np.int64).max).ravel()
            k = int(np.argmin(cost))
            B[pr, pc], G[pr, pc], R[pr, pc] = int(bb2.ravel()[k // 2]), int(gg.ravel()[k]), int(rr2.ravel()[k // 2])
            return True
    return False


def _set_pixel_y_minl2_loop(B, G, R, pr, pc, target_y, max_radius):
    """
    Move BGR pixel (pr,pc) to the colour NEAREST (min L2) to its current value that
    yields OpenCV luma == target_y, searching a window of the given radius.

    Crucially, this does NOT use the grayscale fallback (B=G=R=target_y) that the
    old repair relied on: that fallback forced the blue channel — which has the
    smallest luma weight (1868/16384) and so should barely move — all the way to
    the luma value, costing up to ~200 levels of blue error per pixel and ~3 dB of
    colour PSNR overall.  Solving for the minimum joint (dR,dG,dB) keeps blue near
    its natural value and recovers that PSNR.

    Returns True if a solution was found within `max_radius`, else False (caller
    decides whether to widen the radius or, only as a last resort, force luma).
    """
    b, g, r = int(B[pr, pc]), int(G[pr, pc]), int(R[pr, pc])
    best = None
    best_cost = 1 << 30
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            rr = r + dr
            if rr < 0 or rr > 255:
                continue
            for db in range(-rad, rad + 1):
                bb = b + db
                if bb < 0 or bb > 255:
                    continue
                # solve the G that lands on target_y for this (rr, bb)
                gi = (target_y * 16384 + 8192 - 4899 * rr - 1868 * bb) / 9617.0
                for gg in (int(np.floor(gi)), int(np.ceil(gi))):
                    if 0 <= gg <= 255 and \
                       ((4899 * rr + 9617 * gg + 1868 * bb + 8192) >> 14) == target_y:
                        cost = dr * dr + db * db + (gg - g) * (gg - g)
                        if cost < best_cost:
                            best_cost = cost
                            best = (bb, gg, rr)
        if best is not None:
            break  # found at the smallest possible radius — minimal distortion
    if best is not None:
        B[pr, pc], G[pr, pc], R[pr, pc] = best
        return True
    return False


def _set_pixel_y_minl2_batch(B, G, R, prs, pcs, targets, max_radius, chunk=8192):
    """_set_pixel_y_minl2 for many independent pixels at once, with exactly the same
    result per pixel.  The loop version stops at the first radius whose square holds
    a solution; every smaller square held none, so only the new outer ring of that
    square can contain solutions.  The ring is evaluated in the loop's order (dR
    outer, dB inner, floor(G) before ceil(G)) and np.argmin keeps the first minimum.
    Returns a boolean array: solution found within max_radius."""
    ok = np.zeros(len(prs), dtype=bool)
    b0 = B[prs, pcs].astype(np.int64); g0 = G[prs, pcs].astype(np.int64); r0 = R[prs, pcs].astype(np.int64)
    todo = np.arange(len(prs))
    for rad in range(1, max_radius + 1):
        if len(todo) == 0:
            break
        d = np.arange(-rad, rad + 1, dtype=np.int64)
        DR, DB = np.meshgrid(d, d, indexing="ij")                  # row-major = dR outer, dB inner
        ring = (np.abs(DR) == rad) | (np.abs(DB) == rad) | (rad == 1)   # radius 1: whole square incl. centre
        dr, db = DR[ring], DB[ring]                                 # ring cells in loop order
        still = []
        for c0 in range(0, len(todo), chunk):
            idx = todo[c0:c0 + chunk]
            r, b, g, t = r0[idx, None], b0[idx, None], g0[idx, None], targets[idx, None]
            rr = r + dr[None, :]; bb = b + db[None, :]
            ok_rb = (rr >= 0) & (rr <= 255) & (bb >= 0) & (bb <= 255)
            gi = (t * 16384 + 8192 - 4899 * rr - 1868 * bb) / 9617.0
            gg = np.stack([np.floor(gi), np.ceil(gi)], axis=-1).astype(np.int64)
            rr3, bb3, t3 = rr[..., None], bb[..., None], t[..., None]
            valid = ok_rb[..., None] & (gg >= 0) & (gg <= 255) &                     (((4899 * rr3 + 9617 * gg + 1868 * bb3 + 8192) >> 14) == t3)
            found = valid.reshape(len(idx), -1).any(axis=1)
            rows = np.nonzero(found)[0]
            if len(rows):
                cost = (rr3 - r[..., None]) ** 2 + (bb3 - b[..., None]) ** 2 + (gg - g[..., None]) ** 2
                cost = np.where(valid, cost, np.iinfo(np.int64).max).reshape(len(idx), -1)[rows]
                kk = np.argmin(cost, axis=1)
                cell = kk // 2
                sel = idx[rows]
                B[prs[sel], pcs[sel]] = b0[sel] + db[cell]
                R[prs[sel], pcs[sel]] = r0[sel] + dr[cell]
                G[prs[sel], pcs[sel]] = gg.reshape(len(idx), -1)[rows, kk]
                ok[sel] = True
            still.append(idx[~found])
        todo = np.concatenate(still) if still else todo[:0]
    return ok


def _set_pixel_y(B, G, R, pr, pc, target_y):
    """Guaranteed luma hit (last resort).  Solve G alone, clamped; if that is
    impossible, fall back to grayscale.  Used only for the final stubborn pixels."""
    b, g, r = int(B[pr, pc]), int(G[pr, pc]), int(R[pr, pc])
    gi = (target_y * 16384 + 8192 - 4899 * r - 1868 * b) / 9617.0
    for gg in range(max(0, int(gi) - 2), min(255, int(gi) + 2) + 1):
        if ((4899 * r + 9617 * gg + 1868 * b + 8192) >> 14) == target_y:
            G[pr, pc] = gg
            return
    R[pr, pc] = G[pr, pc] = B[pr, pc] = target_y


def _exact_y_repair(cover_bgr: np.ndarray, target_y: np.ndarray, cr: np.ndarray,
                    cb: np.ndarray, header_bits: list, payload_bits: list,
                    slot_order: list) -> np.ndarray:
    """
    Build a BGR image whose Y channel (read via BGR→YCrCb) yields the embedded
    bits exactly, while keeping colour distortion minimal.

    Starts from the naive YCrCb→BGR inverse (high PSNR) and repairs only the
    blocks whose bits are wrong after the colour round-trip.  Each repaired pixel
    is moved by the MINIMUM L2 colour change that achieves its target luma
    (_set_pixel_y_minl2), which keeps the blue channel near its natural value and
    avoids the ~3 dB colour-PSNR loss of the old grayscale fallback.

    The L2 search radius widens with iteration so that the loop is guaranteed to
    converge to 100 % correct extraction even for the few stubborn pixels; only in
    the final iterations is the guaranteed (but costlier) _set_pixel_y used.
    """
    bgr = cv2.cvtColor(cv2.merge([target_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    n = len(payload_bits)
    arrs = _slot_arrays(slot_order[:n])
    want = np.asarray(payload_bits, dtype=np.int64)
    max_iter = max(MAX_FIX_ITERATIONS, 40)

    for rep in range(max_iter):
        y_read = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0]
        blk, _, _ = _to_blocks(y_read)

        wrong_blocks = set()
        for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
            chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
            for k, (u, v) in enumerate(_HEADER_POSITIONS):
                if _lsb_read(blk[bi, bj, u, v], u, v) != chunk[k]:
                    wrong_blocks.add((bi, bj))
        bad = np.nonzero(_parities(blk, arrs) != want)[0]
        wrong_blocks.update(zip(arrs[0][bad].tolist(), arrs[1][bad].tolist()))

        if not wrong_blocks:
            break

        # Widen the min-L2 search as iterations progress; force luma only at the end.
        max_radius = 4 if rep < 20 else (10 if rep < 30 else 40)
        force_last = rep >= max_iter - 3

        bgr_i = bgr.astype(np.int32)
        B, G, R = bgr_i[:, :, 0], bgr_i[:, :, 1], bgr_i[:, :, 2]
        # every pixel of a wrong block whose luma differs from the target; pixels
        # are independent, so they are repaired together (_set_pixel_y_minl2_batch
        # gives exactly the result of calling _set_pixel_y_minl2 on each in turn)
        sel = np.zeros(y_read.shape, dtype=bool)
        for (bi, bj) in wrong_blocks:
            sel[bi * N:bi * N + N, bj * N:bj * N + N] = True
        sel &= (y_read.astype(np.int64) != target_y.astype(np.int64))
        prs, pcs = np.nonzero(sel)
        ok = _set_pixel_y_minl2_batch(B, G, R, prs, pcs, target_y[prs, pcs].astype(np.int64), max_radius)
        if force_last:
            for pr, pc in zip(prs[~ok], pcs[~ok]):
                _set_pixel_y(B, G, R, pr, pc, int(target_y[pr, pc]))
        bgr = np.stack([B, G, R], axis=2).clip(0, 255).astype(np.uint8)

    return bgr


def get_capacity(cover_bgr: np.ndarray) -> dict:
    Y            = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _to_blocks(Y)
    K            = _compute_K(blocks)
    nr, nc       = blocks.shape[:2]
    n_payload    = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    total = n_payload * (K + 1)
    h, w  = cover_bgr.shape[:2]
    return {
        'payload_bits' : total,
        'K'            : K,
        'bpp'          : total / (h * w),
        'approx_chars' : total // 8,
        'zz_start'     : ZZ_START,
        'zz_end'       : ZZ_END,
    }


def _embed_normal(cover_bgr: np.ndarray, message: str) -> np.ndarray:
    """
    Embed UTF-8 message into cover image using adaptive DCT steganography.

    Embeds into the luminance (Y) channel.  The verify-fix loop runs purely in
    the Y domain (IDCT→uint8 round-trip only, which converges for any message
    size), then the stego image is written as BGR via _exact_y_repair so that
    extract()'s BGR→YCrCb read returns the embedded Y bit-exactly.  This keeps
    extraction 100% accurate despite OpenCV's otherwise-lossy colour round-trip.
    """
    h, w = cover_bgr.shape[:2]

    raw  = message.encode('utf-8')
    bits = [(b >> (7 - k)) & 1 for b in raw for k in range(8)]
    n    = len(bits)

    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))

    blocks, h8, w8 = _to_blocks(Y)
    K              = _compute_K(blocks)
    nr, nc         = blocks.shape[:2]

    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)

    print(f"  Image    : {h}x{w}  |  blocks {nr}x{nc}")
    print(f"  K        : {K}  (freqs 1-{K+1})  |  ZZ range {ZZ_START}-{ZZ_END}  "
          f"|  capacity {capacity} bits  ({capacity/(h*w):.4f} BPP  ~{capacity//8} chars)")
    print(f"  Message  : {len(message)} chars  {n} bits  ({n/(h*w):.4f} BPP)")

    if n > capacity:
        raise ValueError(
            f"Message too large: {n} bits needed, {capacity} bits available.  "
            f"Max message length: ~{capacity // 8} chars."
        )

    # ── Step 1: write header ─────────────────────────────────────────────────
    header_bits = _pack_header(n, K)
    _write_header(blocks, n, K)

    # ── Step 2: embed payload bits (distortion-ordered slots, min-distortion) ─
    # Fill the lowest-distortion slots across all blocks first, and always pick
    # the q±1 candidate closest to the original coefficient (flip_dir=-1).
    slot_order = _build_slot_order(K, nr, nc)
    for k in range(n):
        i, j, u, v = slot_order[k]
        blocks[i, j, u, v] = _parity_write(blocks[i, j, u, v], u, v, bits[k])

    print(f"  Embedded : {n} / {n} bits  OK")

    # ── Step 3: verify-fix loop (Y-domain IDCT round-trip only) ──────────────
    stego_y = _from_blocks(blocks, Y, h8, w8)

    total_corrections = 0
    fix_state = {}
    for fix_iter in range(MAX_FIX_ITERATIONS):
        stego_y, n_remaining = _verify_and_fix(
            slot_order, stego_y, h8, w8, header_bits, bits, K, fix_state
        )
        if n_remaining == 0:
            break
        total_corrections += n_remaining
        print(f"  Fix pass {fix_iter + 1}: {n_remaining} bit(s) wrong after IDCT roundtrip")
    else:
        raise RuntimeError(
            f"Verify-fix did not converge after {MAX_FIX_ITERATIONS} iterations.  "
            f"Please report this image and message."
        )

    if total_corrections > 0:
        print(f"  Verify   : converged in {fix_iter + 1} pass(es) — extraction guaranteed")
    else:
        print(f"  Verify   : all bits stable — no corrections needed")

    # ── Step 4: write BGR whose Y channel reproduces the embedded bits ───────
    # Selective repair: start from the high-PSNR naive YCrCb→BGR inverse and fix
    # only the blocks whose bits would otherwise read wrong after BGR→YCrCb.
    return _exact_y_repair(cover_bgr, stego_y, Cr, Cb, header_bits, bits, slot_order)


def _extract_normal(stego_bgr: np.ndarray) -> str:
    """Extract message from stego image.  No key or sidecar file needed."""
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, h8, w8 = _to_blocks(Y)

    n_bits, K, version, header_ok = _read_header(blocks)

    if not header_ok:
        raise ValueError(
            "Header CRC or magic check failed.  "
            "This image does not contain a DSTG-embedded message."
        )

    nr, nc    = blocks.shape[:2]
    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)

    if not (0 < n_bits <= capacity):
        raise ValueError(
            f"Header payload_bits={n_bits} outside valid range [1, {capacity}].  "
            "Corrupt header or wrong image."
        )

    print(f"  Header   : {n_bits} bits  K={K}  v{version}  CRC OK")

    # Reconstruct the identical distortion-ordered slot list from K, nr, nc
    # (all recovered/known), then read payload bits in that order.
    slot_order = _build_slot_order(K, nr, nc)
    bits = _parities(blocks, _slot_arrays(slot_order[:n_bits])).tolist()

    byte_arr = bytearray(
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, len(bits) - 7, 8)
    )
    try:
        return byte_arr.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"UTF-8 decode failed after extracting {len(bits)} bits.  "
            "Image may have been re-compressed after embedding."
        ) from exc


def quality_metrics(cover_bgr: np.ndarray, stego_bgr: np.ndarray,
                    n_payload_bits: int) -> dict:
    """
    Quality metrics for the stego image.

    Because this method embeds in the LUMINANCE (Y) channel, the primary,
    fair metric is the Y-channel PSNR/SSIM (luma-vs-luma).  This is the
    standard reported metric in luminance-domain steganography, since the
    carrier is Y and any chrominance error is purely an artefact of OpenCV's
    integer BGR↔YCrCb conversion, not of the hiding itself.

    The 3-channel colour PSNR/SSIM is also returned for completeness, but note
    it is intrinsically ~2-4 dB lower than single-channel-carrier methods (e.g.
    green-channel embedding) for the SAME perceptual quality: a luminance change
    must move all three of R, G, B, so every colour channel carries a share of
    the error, whereas a green-only method leaves B and R pristine and the
    3-channel average is inflated by two error-free channels.

    Returns both, with the luma values as the primary 'psnr'/'ssim' keys.
    """
    h = min(cover_bgr.shape[0], stego_bgr.shape[0])
    w = min(cover_bgr.shape[1], stego_bgr.shape[1])
    c, s = cover_bgr[:h, :w], stego_bgr[:h, :w]

    # Colour (3-channel BGR) metrics
    psnr_color = peak_signal_noise_ratio(c, s, data_range=255)
    ssim_color = structural_similarity(c, s, data_range=255, channel_axis=2)

    # Luma (Y-channel) metrics — the fair, primary metric for Y embedding
    yc = cv2.split(cv2.cvtColor(c, cv2.COLOR_BGR2YCrCb))[0]
    ys = cv2.split(cv2.cvtColor(s, cv2.COLOR_BGR2YCrCb))[0]
    psnr_luma = peak_signal_noise_ratio(yc, ys, data_range=255)
    ssim_luma = structural_similarity(yc, ys, data_range=255)

    return {
        'psnr'      : psnr_luma,    # primary = luma PSNR (fair for Y embedding)
        'ssim'      : ssim_luma,    # primary = luma SSIM
        'psnr_luma' : psnr_luma,
        'ssim_luma' : ssim_luma,
        'psnr_color': psnr_color,
        'ssim_color': ssim_color,
        'bpp'       : n_payload_bits / (h * w),
    }


# ════ (2) secure mode / (3) robust mode ════
VERSION_SECURE = 5
STC_H          = 7              # constraint height (2**7 = 128 trellis states)
DEFAULT_KEY    = b"DEDS-v5-public-key"
COST_MODE      = "dual"         # "dual" = texture/saturation weighted, "dist" = d_k^2 only
TAU            = 4.0
GAMMA          = 4.0
MARGIN         = 0.25           # margin used for the first write of a changed slot
PULL_ALL       = True           # pull every slot (changed or not) at least MARGIN*Q inside its bin


# ── keyed helpers ────────────────────────────────────────────────────────────
def _rng(key: bytes, tag: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(key + b"|" + tag.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def _all_slots(nr: int, nc: int, key: bytes) -> list:
    """Every (i, j, u, v) with (u, v) in M and (i, j) not a header block, in keyed order."""
    slots = [(i, j, u, v) for i in range(nr) for j in range(nc)
             if (i, j) not in _HEADER_BLOCK_SET for (u, v) in _PAYLOAD_CANDIDATES]
    perm = _rng(key, f"perm-{nr}x{nc}").permutation(len(slots))
    return [slots[p] for p in perm]


def _submatrix(h: int, wmax: int, key: bytes) -> np.ndarray:
    """wmax columns of h bits (as ints); top and bottom bit set, as usual for STC."""
    r = _rng(key, f"H-{h}-{wmax}").integers(0, 2 ** h, size=wmax)
    return (r | 1 | (1 << (h - 1))).astype(np.int64)


def _widths(n: int, m: int) -> np.ndarray:
    """Columns per message bit: floor(n/m) or one more, spread evenly; sum = n."""
    base, extra = divmod(n, m)
    w = np.full(m, base, dtype=np.int64)
    if extra:
        w[(np.arange(extra) * m) // extra] += 1
    return w


# ── syndrome-trellis code ────────────────────────────────────────────────────
def stc_embed(x: np.ndarray, rho: np.ndarray, msg: np.ndarray, key: bytes, h: int = STC_H) -> np.ndarray:
    """Return y minimising sum rho[y != x] subject to H y = msg."""
    n, m = len(x), len(msg)
    if m > n:
        raise ValueError("message longer than the number of slots")
    w = _widths(n, m)
    Hs = _submatrix(h, int(w.max()), key)
    S = 1 << h
    states = np.arange(S)
    INF = np.inf
    cost = np.full(S, INF); cost[0] = 0.0
    path = np.zeros((n, S), dtype=bool)
    c = 0
    for j in range(m):
        mask = (1 << min(h, m - j)) - 1
        for t in range(int(w[j])):
            col = int(Hs[t]) & mask
            k0 = 0.0 if x[c] == 0 else rho[c]
            k1 = rho[c] if x[c] == 0 else 0.0
            a = cost + k0                      # y_c = 0: state unchanged
            b = cost[states ^ col] + k1        # y_c = 1: came from state ^ col
            take1 = b < a
            path[c] = take1
            cost = np.where(take1, b, a)
            c += 1
        # the lowest state bit is syndrome bit j; it must equal msg[j]
        cost = _shift(cost, int(msg[j]), S)
    # backtrack from the cheapest final state
    s = int(np.argmin(cost))
    if not np.isfinite(cost[s]):
        raise RuntimeError("STC: no solution")
    y = np.zeros(n, dtype=np.int64)
    c = n
    for j in range(m - 1, -1, -1):
        s = ((s << 1) | int(msg[j])) & (S - 1)
        mask = (1 << min(h, m - j)) - 1
        for t in range(int(w[j]) - 1, -1, -1):
            c -= 1
            if path[c, s]:
                y[c] = 1
                s ^= int(Hs[t]) & mask
    return y


def _shift(cost: np.ndarray, bit: int, S: int) -> np.ndarray:
    """After a block: keep states whose lowest bit == bit, then drop that bit."""
    new = np.full(S, np.inf)
    half = S >> 1
    new[:half] = cost[(np.arange(half) << 1) | bit]
    return new


def stc_extract(y: np.ndarray, m: int, key: bytes, h: int = STC_H) -> np.ndarray:
    n = len(y)
    w = _widths(n, m)
    Hs = _submatrix(h, int(w.max()), key)
    syn = np.zeros(m + h, dtype=np.int64)
    c = 0
    for j in range(m):
        mask = (1 << min(h, m - j)) - 1
        for t in range(int(w[j])):
            if y[c]:
                col = int(Hs[t]) & mask
                for r in range(h):
                    if col >> r & 1:
                        syn[j + r] ^= 1
            c += 1
    return syn[:m]


# ── costs ────────────────────────────────────────────────────────────────────
def _flip_distance(coeff: np.ndarray, Q: np.ndarray, mu: float) -> np.ndarray:
    """|move| needed to reach the nearest opposite-parity bin, mu*Q inside it."""
    q = np.round(coeff / Q)
    up = (q + 0.5) * Q + mu * Q - coeff       # into bin q+1
    dn = coeff - ((q - 0.5) * Q - mu * Q)      # into bin q-1
    return np.minimum(up, dn)


def _block_texture(Y: np.ndarray, nr: int, nc: int):
    y = Y[:nr * N, :nc * N].astype(np.float64)
    lap = cv2.Laplacian(y, cv2.CV_64F, ksize=1)
    t = (lap ** 2).reshape(nr, N, nc, N).mean(axis=(1, 3))
    sat = ((Y[:nr * N, :nc * N] == 0) | (Y[:nr * N, :nc * N] == 255)).reshape(nr, N, nc, N).mean(axis=(1, 3))
    return t, sat


def _slot_costs(blocks: np.ndarray, Y: np.ndarray, arrs, mode: str = None) -> np.ndarray:
    mode = mode or COST_MODE
    I, J, U, V, Qs = arrs
    d = _flip_distance(blocks[I, J, U, V], Qs, MARGIN)
    rho = d ** 2
    if mode == "dual":
        nr, nc = blocks.shape[:2]
        tex, sat = _block_texture(Y, nr, nc)
        rho = rho * (1.0 + GAMMA * sat[I, J]) / (tex[I, J] + TAU)
    return np.maximum(rho, 1e-9)


# ── header ───────────────────────────────────────────────────────────────────
def _pack_header_v(n_bits: int, K: int, version: int) -> list:
    global VERSION
    old = VERSION; VERSION = version
    try:
        return _pack_header(n_bits, K)
    finally:
        VERSION = old


def _write_header_bits(blocks: np.ndarray, bits: list) -> None:
    for bidx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = bits[bidx * HEADER_BLOCK_BITS:(bidx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            blocks[bi, bj, u, v] = _lsb_write(blocks[bi, bj, u, v], u, v, chunk[k])


# ── public API ───────────────────────────────────────────────────────────────
def _embed_parities(cover_bgr, Y, Cr, Cb, blocks, h8, w8, header_bits, slots, target, verbose, margin=None):
    """v4 pipeline for an arbitrary slot list and target parity vector."""
    _write_header_bits(blocks, header_bits)
    arrs = _slot_arrays(slots)
    cur = _parities(blocks, arrs)
    # PULL_ALL: also pull unchanged slots in from their bin edges (as v4 does),
    # so that rounding after neighbouring changes does not flip them.
    todo = range(len(slots)) if PULL_ALL else np.nonzero(cur != target)[0]
    for k in todo:
        i, j, u, v = slots[k]
        blocks[i, j, u, v] = _parity_write(blocks[i, j, u, v], u, v, int(target[k]),
                                             margin=MARGIN if margin is None else margin)
    stego_y = _from_blocks(blocks, Y, h8, w8)
    state = {}
    tbits = target.tolist()
    for it in range(MAX_FIX_ITERATIONS):
        stego_y, rem = _verify_and_fix(slots, stego_y, h8, w8, header_bits, tbits, MAX_K, state)
        if rem == 0:
            break
        if verbose:
            print(f"  Fix pass {it + 1}: {rem} bit(s) wrong after IDCT roundtrip")
    else:
        raise RuntimeError(f"Verify-fix did not converge after {MAX_FIX_ITERATIONS} iterations.")
    if verbose:
        print(f"  Verify   : converged in {it + 1} pass(es)")
    return _exact_y_repair(cover_bgr, stego_y, Cr, Cb, header_bits, tbits, slots)


def secure_embed(cover_bgr: np.ndarray, message: str, key: bytes = DEFAULT_KEY, verbose: bool = True) -> np.ndarray:
    raw = message.encode("utf-8")
    msg = np.array([(b >> (7 - k)) & 1 for b in raw for k in range(8)], dtype=np.int64)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = _to_blocks(Y)
    nr, nc = blocks.shape[:2]
    slots = _all_slots(nr, nc, key)
    if len(msg) > len(slots):
        raise ValueError(f"Message too large: {len(msg)} bits, {len(slots)} slots.")
    arrs = _slot_arrays(slots)
    x = _parities(blocks, arrs)
    rho = _slot_costs(blocks, Y, arrs)
    y = stc_embed(x, rho, msg, key)
    if verbose:
        print(f"  STC      : m={len(msg)} n={len(slots)} rate={len(msg)/len(slots):.3f} "
              f"changes={int(np.sum(y != x))} ({np.mean(y != x):.3f} of slots)")
    header_bits = _pack_header_v(len(msg), MAX_K, VERSION_SECURE)
    return _embed_parities(cover_bgr, Y, Cr, Cb, blocks, h8, w8, header_bits, slots, y, verbose)


def extract(stego_bgr: np.ndarray, key: bytes = DEFAULT_KEY) -> str:
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _to_blocks(Y)
    n_bits, K, version, ok = _read_header(blocks)
    if not ok:
        raise ValueError("Header CRC or magic check failed.")
    if version == VERSION_ROBUST:
        return robust_extract(stego_bgr, key)
    if version != VERSION_SECURE:
        return _extract_normal(stego_bgr)            # v3/v4 image
    nr, nc = blocks.shape[:2]
    slots = _all_slots(nr, nc, key)
    y = _parities(blocks, _slot_arrays(slots))
    bits = stc_extract(y, n_bits, key)
    raw = bytearray(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, n_bits - 7, 8))
    return raw.decode("utf-8")




# ════════════════════════════════════════════════════════════════════════════
# Robust mode (version 6): keyed dispersion + framed repetition + soft voting
# ════════════════════════════════════════════════════════════════════════════
# Frame F = [len (16 bits, message bytes) | message | CRC-16/CCITT (16 bits)],
# L = |F|.  The n slots (keyed order, as in secure mode) hold r = floor(n / L)
# copies of F back to back; because the slot order is a keyed permutation, the
# r copies of each frame bit lie in r unrelated blocks and positions, so damage
# that is local in the image (a block, a band) is dispersed over the frame.
# Slots beyond r*L are left untouched.  Bits are written at the bin centre
# (margin 0.5, the most tolerant setting).
#
# Soft decoding (ours).  For a read coefficient c at step Q, let
#     f = c/Q - round(c/Q)   in [-1/2, 1/2],
#     a = 1 - 2|f|           in [0, 1]  (1 at the bin centre, 0 at a bin edge).
# a written coefficient sits at f = 0; a perturbation e moves it to f = e/Q, and
# its parity is wrong only once |e/Q| > 1/2, i.e. once a would have passed 0.
# So a is a monotone reliability: a read near the centre is very likely right,
# one near an edge is close to a coin toss.  Each copy votes  (2b - 1) * a  and
# frame bit i is decided by the sign of the sum over its r copies.  With a = 1
# for every read this reduces to the plain majority vote.
#
# Length recovery.  If the header survives, |m| comes from it.  If not, every
# candidate byte length is tried; a candidate is accepted when the decoded
# length field equals it and the CRC-16 matches (false accept ~ 2^-32).
VERSION_ROBUST = 6
ROBUST_MARGIN  = 0.5


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _bits_of(data: bytes) -> np.ndarray:
    return np.array([(b >> (7 - k)) & 1 for b in data for k in range(8)], dtype=np.int64)


def _frame(raw: bytes) -> np.ndarray:
    return _bits_of(len(raw).to_bytes(2, "big") + raw + _crc16(raw).to_bytes(2, "big"))


def robust_capacity_bytes(cover_bgr: np.ndarray, r: float, key: bytes = DEFAULT_KEY) -> int:
    """Largest message (bytes) that gets at least r copies on average."""
    Y = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))[0]
    nr, nc = _to_blocks(Y)[0].shape[:2]
    n = len(_all_slots(nr, nc, key))
    return max(0, (int(n / r) - 32) // 8)


def robust_embed(cover_bgr: np.ndarray, message: str, key: bytes = DEFAULT_KEY, verbose: bool = True) -> np.ndarray:
    raw = message.encode("utf-8")
    F = _frame(raw)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = _to_blocks(Y)
    nr, nc = blocks.shape[:2]
    slots = _all_slots(nr, nc, key)
    if len(F) > len(slots):
        raise ValueError(f"Message too large for robust mode: frame {len(F)} bits, {len(slots)} slots.")
    # every slot is used: frame bit i is written at slots i, i+L, i+2L, ... so the
    # first (n mod L) frame bits get one extra copy.  Redundancy therefore adapts to
    # the message size: short messages get many copies, long ones few.
    used = slots
    target = np.resize(F, len(slots))
    if verbose:
        print(f"  Robust   : {len(raw)} bytes, frame {len(F)} bits, "
              f"{len(slots) / len(F):.2f} copies on average, {8*len(raw)/65536:.4f} bpp net")
    header_bits = _pack_header_v(8 * len(raw), MAX_K, VERSION_ROBUST)
    return _embed_parities(cover_bgr, Y, Cr, Cb, blocks, h8, w8, header_bits, used, target,
                           verbose, margin=ROBUST_MARGIN)


def _soft_reads(stego_bgr: np.ndarray, key: bytes):
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _to_blocks(Y)
    nr, nc = blocks.shape[:2]
    slots = _all_slots(nr, nc, key)
    I, J, U, V, Qs = _slot_arrays(slots)
    z = blocks[I, J, U, V] / Qs
    q = np.round(z)
    b = (q.astype(np.int64) & 1)
    a = 1.0 - 2.0 * np.abs(z - q)
    return blocks, (2 * b - 1) * a


def _votes(soft: np.ndarray, L: int, upto: int = None) -> np.ndarray:
    """Sum of soft votes for frame bits 0..upto-1 (all copies, incl. the partial last one)."""
    upto = L if upto is None else upto
    r = len(soft) // L
    v = soft[:r * L].reshape(r, L)[:, :upto].sum(axis=0)
    rem = soft[r * L:r * L + upto]
    v[:len(rem)] += rem
    return v


def _decode_frame(soft: np.ndarray, n_bytes: int):
    L = 8 * n_bytes + 32
    if L > len(soft):
        return None
    # cheap test first: the 16-bit length field must decode to n_bytes
    head = np.packbits((_votes(soft, L, 16) > 0).astype(np.uint8))
    if int.from_bytes(head.tobytes(), "big") != n_bytes:
        return None
    by = np.packbits((_votes(soft, L) > 0).astype(np.uint8)).tobytes()
    raw, crc = by[2:-2], int.from_bytes(by[-2:], "big")
    return raw if _crc16(raw) == crc else None


def robust_bit_recovery(stego_bgr: np.ndarray, message: str, key: bytes = DEFAULT_KEY) -> float:
    """Share of message bits correct after soft voting, frame length known (for evaluation)."""
    raw = message.encode("utf-8")
    F = _frame(raw)
    _, soft = _soft_reads(stego_bgr, key)
    dec = (_votes(soft, len(F)) > 0).astype(np.int64)
    return float(np.mean(dec[16:-16] == F[16:-16]))


def robust_extract(stego_bgr: np.ndarray, key: bytes = DEFAULT_KEY):
    blocks, soft = _soft_reads(stego_bgr, key)
    n_bits, K, version, ok = _read_header(blocks)
    if ok and version == VERSION_ROBUST:
        raw = _decode_frame(soft, n_bits // 8)
        if raw is not None:
            return raw.decode("utf-8", errors="replace")
    for n_bytes in range(1, (len(soft) - 32) // 8 + 1):       # header lost: search the length
        raw = _decode_frame(soft, n_bytes)
        if raw is not None:
            return raw.decode("utf-8", errors="replace")
    return None




# ════ (4) public API ════
def embed(cover_bgr: np.ndarray, message: str, mode: str = "normal",
          key: bytes = DEFAULT_KEY, verbose: bool = True) -> np.ndarray:
    if mode == "normal":
        if verbose:
            return _embed_normal(cover_bgr, message)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            return _embed_normal(cover_bgr, message)
    if mode == "secure":
        return secure_embed(cover_bgr, message, key=key, verbose=verbose)
    if mode == "robust":
        return robust_embed(cover_bgr, message, key=key, verbose=verbose)
    raise ValueError("mode must be 'normal', 'robust' or 'secure'")
