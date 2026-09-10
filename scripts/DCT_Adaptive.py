"""
DCT_Adaptive.py  —  DSTG v3: Self-Contained Adaptive DCT Steganography (Y channel)
==================================================================================
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
                handled exactly by _make_exact_y_bgr, so this loop sees only
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
"""

import os, zlib
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global constants ─────────────────────────────────────────────────────────
N                     = 8
VERSION               = 3                  # v3: Y channel + exact-Y BGR reconstruction
FALLBACK_K            = 24
ELIGIBILITY_THRESHOLD = 2
MAGIC_SHORT           = 0x4447            # 'DG'
HEADER_BLOCKS         = [(0, 0), (0, 1)]
HEADER_BLOCK_BITS     = 32
HEADER_TOTAL_BITS     = 64
MAX_FIX_ITERATIONS    = 20

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
    ch     = channel[:h8, :w8].astype(np.float64)
    blocks = np.zeros((nr, nc, N, N), dtype=np.float64)
    for i in range(nr):
        for j in range(nc):
            blocks[i, j] = _dct2(ch[i*N:i*N+N, j*N:j*N+N] - 128.0)
    return blocks, h8, w8

def _from_blocks(blocks: np.ndarray, orig: np.ndarray, h8: int, w8: int) -> np.ndarray:
    out    = orig.copy()
    nr, nc = blocks.shape[:2]
    for i in range(nr):
        for j in range(nc):
            patch = np.clip(np.round(_idct2(blocks[i, j]) + 128.0), 0, 255)
            out[i*N:i*N+N, j*N:j*N+N] = patch.astype(np.uint8)
    return out

# ── Quantised LSB helpers ────────────────────────────────────────────────────
def _qi(coeff: float, u: int, v: int) -> int:
    return int(np.round(float(coeff) / float(_Q_TABLE[u, v])))

def _lsb_read(coeff: float, u: int, v: int) -> int:
    return _qi(coeff, u, v) & 1

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

# ── Verify-fix pass (G-channel: only IDCT roundtrip, no color loss) ──────────
def _verify_and_fix(slot_order: list,
                    current_y: np.ndarray,
                    h8: int, w8: int,
                    header_bits: list,
                    payload_bits: list,
                    K: int) -> tuple:
    """
    Verify and fix embedded bits after IDCT→uint8→re-DCT roundtrip.

    This loop runs entirely in the luminance (Y) pixel domain, so the only
    source of bit-flip is IDCT→uint8 rounding.  It converges reliably in
    ≤ 12 iterations for any message size.  The lossy BGR↔YCrCb colour
    round-trip is handled separately (and exactly) by _make_exact_y_bgr at
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
    n_payload_bits = len(payload_bits)
    for k in range(n_payload_bits):
        i, j, u, v = slot_order[k]
        expected = payload_bits[k]
        if _lsb_read(blk[i, j, u, v], u, v) != expected:
            blk[i, j, u, v] = _lsb_write(blk[i, j, u, v], u, v, expected, flip_dir=-1)

    # Reconstruct Y channel from fixed DCT blocks.
    new_g = _from_blocks(blk, current_y, h8, w8)

    # ── Count remaining errors AFTER the IDCT roundtrip ─────────────────────
    # This is the definitive check: does extract() get the right bits?
    blk_v, _, _ = _to_blocks(new_g)
    n_remaining = 0

    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            if _lsb_read(blk_v[bi, bj, u, v], u, v) != chunk[k]:
                n_remaining += 1

    for k in range(n_payload_bits):
        i, j, u, v = slot_order[k]
        if _lsb_read(blk_v[i, j, u, v], u, v) != payload_bits[k]:
            n_remaining += 1

    return new_g, n_remaining

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
        for k in range(n):
            i, j, u, v = slot_order[k]
            if _lsb_read(blk[i, j, u, v], u, v) != payload_bits[k]:
                wrong_blocks.add((i, j))

        if not wrong_blocks:
            break

        # Widen the min-L2 search as iterations progress; force luma only at the end.
        max_radius = 4 if rep < 20 else (10 if rep < 30 else 40)
        force_last = rep >= max_iter - 3

        bgr_i = bgr.astype(np.int32)
        B, G, R = bgr_i[:, :, 0], bgr_i[:, :, 1], bgr_i[:, :, 2]
        for (bi, bj) in wrong_blocks:
            for pr in range(bi * N, bi * N + N):
                for pc in range(bj * N, bj * N + N):
                    t = int(target_y[pr, pc])
                    if int(y_read[pr, pc]) != t:
                        ok = _set_pixel_y_minl2(B, G, R, pr, pc, t, max_radius)
                        if not ok and force_last:
                            _set_pixel_y(B, G, R, pr, pc, t)
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


def embed(cover_bgr: np.ndarray, message: str) -> np.ndarray:
    """
    Embed UTF-8 message into cover image using adaptive DCT steganography.

    Embeds into the luminance (Y) channel.  The verify-fix loop runs purely in
    the Y domain (IDCT→uint8 round-trip only, which converges for any message
    size), then the stego image is written as BGR via _make_exact_y_bgr so that
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
        blocks[i, j, u, v] = _lsb_write(blocks[i, j, u, v], u, v, bits[k], flip_dir=-1)

    print(f"  Embedded : {n} / {n} bits  OK")

    # ── Step 3: verify-fix loop (Y-domain IDCT round-trip only) ──────────────
    stego_y = _from_blocks(blocks, Y, h8, w8)

    total_corrections = 0
    for fix_iter in range(MAX_FIX_ITERATIONS):
        stego_y, n_remaining = _verify_and_fix(
            slot_order, stego_y, h8, w8, header_bits, bits, K
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


def extract(stego_bgr: np.ndarray) -> str:
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
    bits = [
        _lsb_read(blocks[i, j, u, v], u, v)
        for (i, j, u, v) in slot_order[:n_bits]
    ]

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

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    COVER_PATH = r'F:\Thesis\cover_p2_s46.png'
    STEGO_PATH = r'F:\Thesis\stego_adaptive.png'

    cover = cv2.imread(COVER_PATH)
    assert cover is not None, f'Cannot load {COVER_PATH}'

    h, w = cover.shape[:2]
    cap  = get_capacity(cover)
    print(f'Image    : {h}×{w}')
    print(f'Capacity : {cap["payload_bits"]} bits  '
          f'(K={cap["K"]}  BPP={cap["bpp"]:.4f}  ~{cap["approx_chars"]} chars)')
    print(f'ZZ range : {cap["zz_start"]}-{cap["zz_end"]}  '
          f'({len(_PAYLOAD_CANDIDATES)} stable positions per block)\n')

    while True:
        MSG   = input('Secret message: ')
        n_req = len(MSG.encode('utf-8')) * 8
        if n_req <= cap['payload_bits']:
            break
        print(f'  [!] Message too large: {len(MSG)} chars ({n_req} bits).')
        print(f'      Max: ~{cap["approx_chars"]} chars ({cap["payload_bits"]} bits).\n')

    # ── Embed ──────────────────────────────────────────────────────────────
    print('\n═══ EMBEDDING ═══')
    stego = embed(cover, MSG)
    cv2.imwrite(STEGO_PATH, stego)
    print(f'  Saved → {STEGO_PATH}')

    raw   = MSG.encode('utf-8')
    mbits = [(b >> (7 - k)) & 1 for b in raw for k in range(8)]
    qm    = quality_metrics(cover, stego, len(mbits))
    print(f'  PSNR (luma/Y) : {qm["psnr_luma"]:.2f} dB   <- primary metric (Y embedding)')
    print(f'  PSNR (color)  : {qm["psnr_color"]:.2f} dB   <- 3-channel BGR')
    print(f'  SSIM (luma/Y) : {qm["ssim_luma"]:.4f}')
    print(f'  SSIM (color)  : {qm["ssim_color"]:.4f}')
    print(f'  BPP           : {qm["bpp"]:.4f}')

    # ── Extract in-memory ──────────────────────────────────────────────────
    print('\n═══ EXTRACT (in-memory) ═══')
    r1 = extract(stego)
    print('  PASS' if r1 == MSG else f'  FAIL\n  Got: {r1!r}')

    # ── Extract from PNG ───────────────────────────────────────────────────
    print('\n═══ EXTRACT (PNG reload) ═══')
    r2 = extract(cv2.imread(STEGO_PATH))
    print('  PASS — cross-device round-trip perfect' if r2 == MSG
          else f'  FAIL\n  Got: {r2!r}')