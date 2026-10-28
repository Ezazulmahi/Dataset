r"""Step 12 - JPEG-domain comparison (J-UNIWARD and J-MiPOD on the same covers, same detector).

Covers: the same 8-bit Y covers as the spatial comparison (matched_0.4\covers_y), compressed with libjpeg 6b
(jpeglib 1.0.2) at quality 75, grayscale:  4,000 universal (train / val / test) + the paper's 1,000 fine-tuned covers.
Stego at 0.4 bits per non-zero AC coefficient (bpnzac), random messages (embedding simulators, seeded per image):
    juniward   conseal 2025.11 J-UNIWARD (Holub, Fridrich, Denemark, EURASIP JIS 2014)
    jmipod     jmipod.py, J-MiPOD (Cogranne, Giboulot, Bas, IH&MMSec 2020), reimplementation on conseal's MiPOD parts
Detector features: DCTR (Holub & Fridrich, IEEE TIFS 2015; sealwatch 2025.9), 8,000-D, of every cover and stego;
classification in 13_jpeg_classify.py (FLD ensemble).
Output: matched_jpeg75\{covers,juniward,jmipod}\<set>\<name>.jpg, features_dctr\<set>\chunk_NNN.npz
Resumable: finished chunks are skipped.   python 12_build_jpeg.py [--workers 3]
DEDS is a spatial-domain embedder and has no JPEG version; it is not part of this table.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, argparse, hashlib, warnings, time, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

QF, ALPHA, CHUNK = 75, 0.4, 100
OUT = C.ROOT / f"matched_jpeg{QF}"; FEAT = C.ROOT / "features_dctr"
METHODS = ["juniward", "jmipod"]

def sets():
    return {"universal": [r[0] for r in C.matched_split("suniward")],
            "paper_ft": [f"{i:03d}.png" for i in range(1000)]}

def seed_of(*p):
    return int(hashlib.md5("|".join(p).encode()).hexdigest()[:8], 16)

def one(set_name, name):
    import jpeglib, conseal, sealwatch as sw, jmipod
    warnings.filterwarnings("ignore")
    stem = Path(name).stem; paths = {k: OUT / k / set_name / f"{stem}.jpg" for k in ["covers", *METHODS]}
    if not all(p.exists() for p in paths.values()):
        x = cv2.imread(str(C.MATCHED / "covers_y" / set_name / name), cv2.IMREAD_GRAYSCALE)
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        tmp = paths["covers"].with_name("tmp_" + paths["covers"].name)
        jpeglib.from_spatial(x[:, :, None]).write_spatial(str(tmp), qt=QF); os.replace(tmp, paths["covers"])
        im = jpeglib.read_dct(str(paths["covers"])); y0 = im.Y.copy(); qt = im.qt[im.quant_tbl_no[0]]
        x0 = jpeglib.read_spatial(str(paths["covers"])).spatial[:, :, 0].astype(float)
        for m in METHODS:
            sd = seed_of(m, set_name, name)
            y1 = (conseal.juniward.simulate_single_channel(x0, y0, qt, ALPHA, seed=sd) if m == "juniward"
                  else jmipod.simulate_single_channel(x0, y0, qt, ALPHA, seed=sd))
            im.Y = y1.astype(y0.dtype); tmp = paths[m].with_name("tmp_" + paths[m].name)
            im.write_dct(str(tmp)); os.replace(tmp, paths[m])
    f = {}
    for k, p in paths.items():                         # DCTR returns an OrderedDict of sub-feature arrays
        f[k] = np.concatenate([np.asarray(v, np.float32).ravel() for v in sw.dctr.extract_from_file(str(p), qf=QF).values()])
    return f

def work(task):
    set_name, k, names, out = task
    rows = [one(set_name, n) for n in names]
    data = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    tmp = str(out) + ".tmp.npz"; np.savez(tmp, **data); os.replace(tmp, out)
    return out

def load(set_name):
    """dict covers / juniward / jmipod -> features in the order of sets()[set_name]."""
    n = len(sets()[set_name]); parts = [np.load(FEAT / set_name / f"chunk_{k:03d}.npz") for k in range((n + CHUNK - 1) // CHUNK)]
    return {key: np.concatenate([p[key] for p in parts]) for key in ["covers", *METHODS]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    tasks = []
    for s, names in sets().items():
        (FEAT / s).mkdir(parents=True, exist_ok=True)
        for k in range(0, len(names), CHUNK):
            out = FEAT / s / f"chunk_{k // CHUNK:03d}.npz"
            if not out.exists():
                tasks.append((s, k, names[k:k + CHUNK], out))
    print(f"JPEG QF {QF}, {ALPHA} bpnzac, {METHODS}: {len(tasks)} chunks to do, {a.workers} workers", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(work, t) for t in tasks])):
            out = fut.result(); el = time.time() - t0
            print(f"{time.strftime('%H:%M')} {out.parent.name}/{out.name}  {i + 1}/{len(tasks)}  ~{el / (i + 1) * (len(tasks) - i - 1) / 3600:.1f} h left", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
