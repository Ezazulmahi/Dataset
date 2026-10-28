r"""Step 6 - same covers, same payload, every method: the data for the matched-detector comparison.

All methods work on the luminance plane Y (OpenCV BGR->YCrCb, uint8), which is the plane DEDS embeds in and
the plane its extractor reads. Covers and stego images are saved as 8-bit grayscale PNG.

Payload: PAYLOAD bits per pixel of random bits for every method (fixed payloads, not message-dependent).
Classical methods (S-UNIWARD, WOW, HILL, MiPOD) use the conseal simulators at relative payload PAYLOAD.
DEDS-v4 stego is added by 07_build_matched_deds.py with the same number of bits per image.

Cover sets: universal (5,000 covers of F:\Steganalyzer_UniversalDCT, seeds 100000+, used for TRAINING),
            paper_ft / paper_bs (the paper's 1,000 + 1,000 covers, TEST only),
            unseen_ft / unseen_bs (200 + 200 unseen-prompt covers, TEST only).
Output: matched_<PAYLOAD>\covers_y\<set>\<name>.png and matched_<PAYLOAD>\<method>\<set>\<name>.png
Resumable: existing files are skipped.   python 06_build_matched.py [--methods suniward wow hill mipod] [--workers 4]
Curriculum data (SRNet does not learn 0.4 bpp from scratch on 3,195 pairs): a higher payload for the training
covers only, e.g.  python 06_build_matched.py --payload 1.0 --sets universal  ->  matched_1.0\<method>\universal\
(covers are not duplicated: they stay in matched_0.4\covers_y\).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, argparse, warnings, hashlib, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PAYLOAD = 0.4                                   # default; --payload changes it (stego output folder only)
OUT = C.ROOT / f"matched_{PAYLOAD}"
COVERS_Y = C.MATCHED / "covers_y"

def cover_sets():
    c = C.R / "09_expansion_1000" / "corpus1000"; u = C.R / "08_unseen_prompts"
    return {
        "universal": [C.UNIV / "covers" / n for n in C.matched_universal_names()],   # 4,000 covers, bs/ft interleaved
        "paper_ft": [c / "finetuned" / f"{i:03d}.png" for i in range(1000)],
        "paper_bs": [c / "baseline" / f"{i:03d}.png" for i in range(1000)],
        "unseen_ft": [u / "finetuned" / f"{i:03d}.png" for i in range(200)],
        "unseen_bs": [u / "baseline" / f"{i:03d}.png" for i in range(200)],
    }

def read_y(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]

def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name); cv2.imwrite(str(tmp), img); os.replace(tmp, path)

def seed_of(*parts):
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)

def work(task):
    import conseal
    warnings.filterwarnings("ignore")
    set_name, src, methods, payload = task
    out = C.ROOT / f"matched_{payload}"
    name = Path(src).name; y = read_y(src)
    cpath = COVERS_Y / set_name / name
    if not cpath.exists():
        save(y, cpath)
    for m in methods:
        p = out / m / set_name / name
        if not p.exists():
            sd = seed_of(m, set_name, name) if payload == PAYLOAD else seed_of(m, set_name, name, str(payload))
            s = getattr(conseal, m).simulate_single_channel(y, payload, seed=sd)
            save(s, p)
    return set_name

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="+", default=["suniward", "wow", "hill", "mipod"])
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--payload", type=float, default=PAYLOAD)
    ap.add_argument("--sets", nargs="+", default=None, help="cover sets to build (default all)"); a = ap.parse_args()
    out = C.ROOT / f"matched_{a.payload}"
    tasks = []
    for s, files in cover_sets().items():
        if a.sets and s not in a.sets:
            continue
        for f in files:
            need = [m for m in a.methods if not (out / m / s / f.name).exists()]
            if need or not (COVERS_Y / s / f.name).exists():
                tasks.append((s, str(f), need, a.payload))
    print(f"{len(tasks)} covers to process, methods {a.methods}, payload {a.payload} bpp, {a.workers} workers -> {out}", flush=True)
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, _ in enumerate(ex.map(work, tasks, chunksize=8)):
            if (k + 1) % 500 == 0 or k + 1 == len(tasks):
                print(f"{k + 1} / {len(tasks)}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
