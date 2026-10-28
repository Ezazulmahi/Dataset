r"""Step 10 - SRM features (rich model + linear classifier) for the matched comparison.

Spatial Rich Model (Fridrich & Kodovsky, IEEE TIFS 2012), 34,671 features, sealwatch 2025.9 (sealwatch.srm.extract,
University of Innsbruck), on the same 8-bit Y images as the CNN detectors:
    covers_universal          4,000 universal covers (train / val / test of the fixed split)
    <method>_universal        their 0.4 bpp stego, method in suniward wow hill mipod deds_v4
    covers_paper_ft           the paper's 1,000 fine-tuned covers
    v5_paper_ft               the 1,000 DEDS-v5 fine-tuned stego of F:\scirep submission 2.0 (09_prepare_v5_test.py)
    <method>_paper_ft         S-UNIWARD / WOW / HILL / MiPOD 0.4 bpp stego of the same 1,000 fine-tuned covers
Output: features_srm\<group>\chunk_NNN.npy (float32, 200 images each, in the order of <group>.txt).
Resumable: finished chunks are skipped.  python 10_srm_extract.py [--workers 5]
maxSRM is not computed: it needs per-pixel change probabilities, which the DEDS embedder does not define.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, argparse, time, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = C.ROOT / "features_srm"; CHUNK = 200
METHODS = ["suniward", "wow", "hill", "mipod", "deds_v4"]

def groups():
    names = [r[0] for r in C.matched_split("suniward")]
    g = {"covers_universal": [C.MATCHED / "covers_y" / "universal" / n for n in names]}
    for m in METHODS:
        g[f"{m}_universal"] = [C.MATCHED / m / "universal" / n for n in names]
    g["covers_paper_ft"] = [C.MATCHED / "covers_y" / "paper_ft" / f"{i:03d}.png" for i in range(1000)]
    g["v5_paper_ft"] = [C.V5_TEST / "paper_ft" / f"{i:03d}.png" for i in range(1000)]
    for m in METHODS[:-1]:   # each classical method's own stego of the same 1,000 fine-tuned covers (Table 3, Fig. 4c)
        g[f"{m}_paper_ft"] = [C.MATCHED / m / "paper_ft" / f"{i:03d}.png" for i in range(1000)]
    return g

def srm(path):
    import sealwatch as sw
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)
    return np.concatenate([np.asarray(a, np.float32).ravel() for a in sw.srm.extract(x).values()])

def work(task):
    out, paths = task
    f = np.stack([srm(p) for p in paths])
    tmp = str(out) + ".tmp.npy"; np.save(tmp, f); os.replace(tmp, out)
    return out

def load(group):
    """Features of a group, rows in the order of features_srm/<group>.txt."""
    d = OUT / group; n = len(open(OUT / f"{group}.txt").read().split())
    return np.concatenate([np.load(d / f"chunk_{k:03d}.npy") for k in range((n + CHUNK - 1) // CHUNK)])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); a = ap.parse_args()
    tasks = []
    for g, paths in groups().items():
        (OUT / g).mkdir(parents=True, exist_ok=True)
        open(OUT / f"{g}.txt", "w").write("\n".join(str(p) for p in paths))
        for k in range(0, len(paths), CHUNK):
            out = OUT / g / f"chunk_{k // CHUNK:03d}.npy"
            if not out.exists():
                tasks.append((out, paths[k:k + CHUNK]))
    total = sum(len(p) for _, p in tasks)
    print(f"SRM: {len(tasks)} chunks ({total} images) to do, {a.workers} workers", flush=True)
    t0 = time.time(); done = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in cf.as_completed([ex.submit(work, t) for t in tasks]):
            out = fut.result(); done += CHUNK
            el = time.time() - t0
            print(f"{time.strftime('%H:%M')} {out.parent.name}/{out.name}  {min(done, total)}/{total}  "
                  f"~{el / min(done, total) * (total - min(done, total)) / 3600:.1f} h left", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
