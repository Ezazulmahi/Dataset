r"""Robust mode for DCT_Adaptive -- an outer repetition code around the UNCHANGED embedder.

What is the same as DCT_Adaptive.py (imported, not modified):
  luminance channel, 8x8 block DCT, the same quantisation table, the same 39 positions,
  the same adaptive K, the same slot order, the same parity rule (_lsb_write), the same
  64-bit header, the same verify-fix loop and the same exact-luminance colour repair.

What is added (message level only):
  * the message is wrapped in a frame  [16-bit byte count | message | 16-bit CRC]
  * the frame is written r times in a row into the payload slots (repetition code, rate 1/r)
  * extraction first tries the normal header; if the header is damaged it searches over K and
    the frame length, takes a majority vote over the r copies and accepts the candidate whose
    CRC is correct. No key or side information is used.

robust_embed(cover_bgr, message_bytes, r) -> stego_bgr
robust_extract(stego_bgr, r)             -> message_bytes or None
"""
import os, zlib, importlib.util
import numpy as np, cv2

_spec = importlib.util.spec_from_file_location("DCT_Adaptive", os.path.join(os.path.dirname(os.path.abspath(__file__)), "DCT_Adaptive.py"))
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)
_Q = D._Q_TABLE.astype(np.float64)
_SLOT_CACHE = {}


def _bits(data: bytes):
    return np.unpackbits(np.frombuffer(data, np.uint8))

def frame_bits(message: bytes) -> np.ndarray:
    head = len(message).to_bytes(2, "big")
    crc = (zlib.crc32(head + message) & 0xFFFF).to_bytes(2, "big")
    return _bits(head + message + crc)

def max_message_bytes(cover_bgr, r: int) -> int:
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cap = D.get_capacity(cover_bgr)["payload_bits"]
    return max(0, (cap // r - 32) // 8)


def robust_embed(cover_bgr: np.ndarray, message: bytes, r: int) -> np.ndarray:
    """Same steps as DCT_Adaptive.embed(), with the payload bits = frame repeated r times."""
    bits = [int(b) for b in np.tile(frame_bits(message), r)]
    n = len(bits)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = D._to_blocks(Y)
    K = D._compute_K(blocks)
    nr, nc = blocks.shape[:2]
    capacity = (nr * nc - len(D.HEADER_BLOCKS)) * (K + 1)
    if n > capacity:
        raise ValueError(f"coded payload {n} bits exceeds capacity {capacity}")
    header_bits = D._pack_header(n, K)
    D._write_header(blocks, n, K)
    slot_order = D._build_slot_order(K, nr, nc)
    for k in range(n):
        i, j, u, v = slot_order[k]
        blocks[i, j, u, v] = D._lsb_write(blocks[i, j, u, v], u, v, bits[k], flip_dir=-1)
    stego_y = D._from_blocks(blocks, Y, h8, w8)
    for _ in range(D.MAX_FIX_ITERATIONS):
        stego_y, remaining = D._verify_and_fix(slot_order, stego_y, h8, w8, header_bits, bits, K)
        if remaining == 0:
            break
    else:
        raise RuntimeError("verify-fix did not converge")
    return D._exact_y_repair(cover_bgr, stego_y, Cr, Cb, header_bits, bits, slot_order)


def _parities(stego_bgr):
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = D._to_blocks(Y)
    return (np.rint(blocks / _Q).astype(np.int64) & 1).astype(np.uint8), blocks

def _slots(K, nr, nc):
    key = (K, nr, nc)
    if key not in _SLOT_CACHE:
        a = np.array(D._build_slot_order(K, nr, nc), dtype=np.intp)
        _SLOT_CACHE[key] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    return _SLOT_CACHE[key]

def _try_decode(stream: np.ndarray, r: int, n_hint=None):
    """Majority-decode a frame repeated r times at the start of `stream`; search the frame length."""
    cap = len(stream)
    cands = [n_hint // r] if n_hint else range(40, cap // r + 1, 8)
    for F in cands:
        if F < 40 or F * r > cap or (F - 32) % 8:
            continue
        B = (F - 32) // 8
        votes = stream[:F * r].reshape(r, F).sum(axis=0)
        fr = (votes * 2 > r).astype(np.uint8)
        if int.from_bytes(np.packbits(fr[:16]).tobytes(), "big") != B:
            continue
        data = np.packbits(fr).tobytes()
        if (zlib.crc32(data[:2 + B]) & 0xFFFF) == int.from_bytes(data[2 + B:4 + B], "big"):
            return data[2:2 + B]
    return None

def robust_extract(stego_bgr: np.ndarray, r: int):
    P, blocks = _parities(stego_bgr)
    nr, nc = P.shape[:2]
    n_bits, K, _, ok = D._read_header(blocks)
    if ok and 1 <= K <= D.MAX_K:                      # header intact: direct read
        i, j, u, v = _slots(K, nr, nc)
        out = _try_decode(P[i, j, u, v], r, n_hint=n_bits)
        if out is not None:
            return out
    for K in range(1, D.MAX_K + 1):                    # header damaged: search over K
        i, j, u, v = _slots(K, nr, nc)
        out = _try_decode(P[i, j, u, v], r)
        if out is not None:
            return out
    return None
