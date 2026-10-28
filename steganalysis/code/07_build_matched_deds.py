r"""Step 7 - DEDS-v4 stego for the matched comparison: same covers and same payload as step 06.

Embedder: a frozen copy of F:\scirep results\10_fixes_2026-10-16\embedder_v4\DCT_Adaptive_v4.py taken on the
first run (embedder_v4_snapshot.py; its SHA-256 and PAYLOAD_MARGIN go to deds_v4_build.json), so every image
uses the same version even if the original file is edited later. Pass --margin to set PAYLOAD_MARGIN.
Payload: 0.4 bpp of uniformly random BYTES (3,276 bytes = 26,208 bits for 256x256), K forced from
the payload as in the paper's protocol. The module's embed() is unchanged; random bytes are passed through a
str subclass whose encode() returns them. Every stego image is read back bit for bit (header CRC + payload).

Output: matched_0.4\deds_v4\<set>\<name>.png (Y plane, grayscale, for the detectors),
        matched_0.4\deds_v4_bgr\<set>\<name>.png (the colour stego image), matched_0.4\deds_v4_embed.csv
Order: universal covers first (training data), then paper_ft, paper_bs, unseen_ft, unseen_bs.
Resumable: images listed as exact in the CSV are skipped.
    python 07_build_matched_deds.py [--workers 10] [--margin 0.15]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import io, sys, csv, json, time, shutil, hashlib, argparse, contextlib, importlib.util, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec6 = importlib.util.spec_from_file_location("b6", HERE / "06_build_matched.py"); B = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(B)

SRC_V4 = Path(r"F:\scirep results\10_fixes_2026-10-16\embedder_v4\DCT_Adaptive_v4.py")
SNAP = HERE / "embedder_v4_snapshot.py"; INFO = B.OUT / "deds_v4_build.json"; LOG = B.OUT / "deds_v4_embed.csv"
N_BYTES = int(B.PAYLOAD * 256 * 256) // 8
FIELDS = ["set", "image", "K", "n_bits", "embed_s", "exact", "psnr_y", "error"]
_D = None

class RawMessage(str):
    """Passes arbitrary bytes through embed(), which only calls message.encode('utf-8')."""
    def __new__(cls, data):
        o = super().__new__(cls, "x" * len(data)); o.data = data; return o
    def encode(self, *a, **k):
        return self.data

def _init(margin):
    global _D
    spec = importlib.util.spec_from_file_location("DCT_Adaptive_v4_snapshot", SNAP)
    _D = importlib.util.module_from_spec(spec); spec.loader.exec_module(_D)
    if margin is not None:
        _D.PAYLOAD_MARGIN = margin
    # Speed-up: the pixel-repair search of embedder_fast\\DCT_Adaptive_v4_fast.py (byte-identical output, proven by
    # embedder_fast\\check_identical.py; ~30x faster). Everything else is the frozen v4 snapshot.
    spec_f = importlib.util.spec_from_file_location("v4_fast", HERE / "embedder_fast" / "DCT_Adaptive_v4_fast.py")
    F = importlib.util.module_from_spec(spec_f); spec_f.loader.exec_module(F)
    _D._set_pixel_y_minl2 = F._set_pixel_y_minl2

def read_bits(stego_bgr):
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _D._to_blocks(Y)
    n_bits, K, _, ok = _D._read_header(blocks)
    if not ok:
        return None
    nr, nc = blocks.shape[:2]
    return np.array(_D._parities(blocks, _D._slot_arrays(_D._build_slot_order(K, nr, nc)[:n_bits])), dtype=np.uint8)

def work(task):
    set_name, src = task
    name = Path(src).name; cover = cv2.imread(src)
    data = np.random.default_rng(B.seed_of("deds_v4", set_name, name)).integers(0, 256, N_BYTES, dtype=np.uint8).tobytes()
    n_bits = 8 * N_BYTES; K = max(1, min(_D.MAX_K, int(np.ceil(n_bits / 1022)) - 1))
    row = {"set": set_name, "image": name, "K": K, "n_bits": n_bits, "error": ""}
    orig = _D._compute_K; _D._compute_K = lambda blocks, K=K: K
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stego = _D.embed(cover, RawMessage(data))
    except Exception as e:
        row.update(embed_s=round(time.perf_counter() - t0, 1), exact=False, error=repr(e)[:200]); return row
    finally:
        _D._compute_K = orig
    row["embed_s"] = round(time.perf_counter() - t0, 1)
    B.save(stego, B.OUT / "deds_v4_bgr" / set_name / name)
    back = cv2.imread(str(B.OUT / "deds_v4_bgr" / set_name / name))
    bits = read_bits(back)
    want = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    row["exact"] = bool(bits is not None and len(bits) == len(want) and (bits == want).all())
    y_c = B.read_y(src); y_s = cv2.cvtColor(back, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mse = ((y_s.astype(float) - y_c) ** 2).mean(); row["psnr_y"] = round(10 * np.log10(255 ** 2 / mse), 3) if mse else float("inf")
    B.save(y_s, B.OUT / "deds_v4" / set_name / name)
    if not (B.OUT / "covers_y" / set_name / name).exists():
        B.save(y_c, B.OUT / "covers_y" / set_name / name)
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--margin", type=float, default=None)
    a = ap.parse_args(); B.OUT.mkdir(parents=True, exist_ok=True)
    if not SNAP.exists():
        shutil.copyfile(SRC_V4, SNAP)
    sha = hashlib.sha256(SNAP.read_bytes()).hexdigest()
    _init(a.margin); margin = _D.PAYLOAD_MARGIN
    if INFO.exists():
        info = json.load(open(INFO))
        if info["sha256"] != sha or info["payload_margin"] != margin:
            sys.exit(f"Snapshot or margin differs from the earlier run ({info}); delete {INFO.name} and the deds_v4 folders to restart.")
    else:
        json.dump({"source": str(SRC_V4), "sha256": sha, "payload_margin": margin, "header_margin": _D.HEADER_MARGIN,
                   "payload_bits": 8 * N_BYTES, "created": time.strftime("%Y-%m-%d %H:%M")}, open(INFO, "w"), indent=1)
    done = set()
    if LOG.exists():
        done = {(r["set"], r["image"]) for r in csv.DictReader(open(LOG, newline="")) if r["exact"] == "True"}
    tasks = [(s, str(f)) for s, files in B.cover_sets().items() for f in files if (s, f.name) not in done]
    print(f"DEDS v4 (PAYLOAD_MARGIN {margin}, sha {sha[:12]}): {len(done)} done, {len(tasks)} to do, {a.workers} workers", flush=True)
    new = not LOG.exists(); t0 = time.time()
    with open(LOG, "a", newline="") as f, cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.margin,)) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futs = [ex.submit(work, t) for t in tasks]
        for k, fu in enumerate(cf.as_completed(futs)):
            r = fu.result(); w.writerow(r); f.flush()
            if (k + 1) % 20 == 0 or not r["exact"]:
                el = time.time() - t0
                print(f"{len(done) + k + 1}/{len(done) + len(tasks)}  {r['set']}/{r['image']} exact {r['exact']} psnr_y {r.get('psnr_y')} "
                      f"{r['error']}  elapsed {el / 3600:.1f} h, eta {el / (k + 1) * (len(tasks) - k - 1) / 3600:.1f} h", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
