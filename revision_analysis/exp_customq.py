"""Repair and re-run the three 'recompression with the embedder's own table' conditions of
Robustness.py that failed with a shape error (the inverse-DCT blocks were written back without
being re-assembled into an image). Only that bug is fixed; the protocol is unchanged:
payload = half of each fine-tuned cover's adaptive capacity, messages seed-reproducible."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, random, string, contextlib, importlib.util, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
RES = r"F:\Results"
ALPHABET = string.ascii_letters + string.digits + " .,!?"
_D = _R = None

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def _init():
    global _D, _R
    _D = _load("DCT_Adaptive", RES + r"\DCT_Adaptive.py"); _R = _load("Robustness", RES + r"\Robustness.py")

def custom_q_fixed(module, img_bgr, quality):
    scale = _R._ijg_scale(quality)
    Qs = np.clip(np.floor((module._Q_TABLE * scale + 50.0) / 100.0), 1, 255)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = _R._blockify(Y)
    q = np.round(_R._dct2_batch(blocks - 128.0) / Qs) * Qs
    recon = np.clip(np.round(_R._idct2_batch(q) + 128.0), 0, 255)
    Y_out = Y.copy(); Y_out[:h8, :w8] = _R._deblockify(recon, h8, w8).astype(np.uint8)      # the fix
    return cv2.cvtColor(cv2.merge([Y_out, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

def work(i):
    name = f"{i:03d}.png"; cover = cv2.imread(os.path.join(RES, "finetuned", name))
    with contextlib.redirect_stdout(io.StringIO()):
        cap = _D.get_capacity(cover)
    n_bytes = max(8, int(cap["payload_bits"] * 0.5) // 8)
    msg = "".join(random.Random(777000 + i).choice(ALPHABET) for _ in range(n_bytes))
    bits = [(b >> (7 - k)) & 1 for b in msg.encode() for k in range(8)]
    Y = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _D._to_blocks(Y)
    with contextlib.redirect_stdout(io.StringIO()):
        K = _D._compute_K(blocks); stego = _D.embed(cover, msg)
    slot = _D._build_slot_order(K, *blocks.shape[:2])
    rows = []
    for label, fn in (("none", lambda s: s.copy()), ("customQ_QF90", lambda s: custom_q_fixed(_D, s, 90)),
                      ("customQ_QF70", lambda s: custom_q_fixed(_D, s, 70)), ("customQ_QF50", lambda s: custom_q_fixed(_D, s, 50))):
        d = fn(stego)
        hdr = _R.header_survives(_D, d)[0]
        ber = _R.bit_error_rate(bits, _R.read_payload_bits_at(_D, d, slot, len(bits)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = (_D.extract(d) == msg)
        except Exception:
            ok = False
        rows.append({"image": name, "condition": label, "K": K, "n_bits": len(bits), "header_intact": bool(hdr), "bit_accuracy": 1 - ber, "exact": bool(ok)})
    return rows

if __name__ == "__main__":
    rows = []
    with cf.ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        for k, r in enumerate(ex.map(work, range(400))):
            rows.extend(r)
            if (k + 1) % 50 == 0: print(k + 1, flush=True)
    d = pd.DataFrame(rows); d.to_csv("customq_rerun.csv", index=False)
    print(d.groupby("condition").agg(n=("exact", "size"), bit_accuracy=("bit_accuracy", "mean"), header_intact=("header_intact", "sum"), exact=("exact", "sum")).round(4))
