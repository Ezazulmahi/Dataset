r"""Step 0c - curriculum data for WSERNet on the fixed universal-DCT set (see 00_build_universal_fixed.py).

WSERNet stayed at chance on universal_fixed (val AUC 0.50 after 7 epochs): embed_dct_random changes the LSB of
only 5-15 % of the eligible quantised coefficients (|c| > 1), the first ones in block order, and only half of
those flips change a value. Here the same embedder runs at rate 1.0 (every eligible coefficient carries a bit)
on the same 5,000 covers, so the network first learns the embedding trace and is then fine-tuned on the real
rates (02_train.py --init). The covers are those of universal_fixed\covers (same requantisation, no message).
Output: universal_r1.0\stego\<name>.png. Training only; nothing here is a test image.
    python 00c_build_universal_curriculum.py [--rate 1.0] [--workers 3]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, argparse, importlib.util, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parent; SRC = Path(r"F:\Steganalyzer_UniversalDCT")
_G = None

def _init():
    global _G
    spec = importlib.util.spec_from_file_location("gen_dataset", r"F:\DEDS_project\gen_dataset.py")
    _G = importlib.util.module_from_spec(spec); spec.loader.exec_module(_G)

def work(task):
    idx, name, rate, out = task
    cover = cv2.imread(str(SRC / "covers" / name))
    np.random.seed(600000 + idx)                     # own seed, not the one of the real-rate stego
    stego = _G.embed_dct_random(cover, rate)
    p = Path(out) / name; tmp = p.with_name("tmp_" + name); cv2.imwrite(str(tmp), stego); os.replace(tmp, p)
    return name

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rate", type=float, default=1.0); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / f"universal_r{a.rate}" / "stego"; out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (SRC / "covers").glob("*.png"))
    tasks = [(i, n, a.rate, str(out)) for i, n in enumerate(names) if not (out / n).exists()]
    print(f"rate {a.rate}: {len(names) - len(tasks)} done, {len(tasks)} to do -> {out}", flush=True)
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for k, _ in enumerate(ex.map(work, tasks, chunksize=20)):
            if (k + 1) % 500 == 0 or k + 1 == len(tasks):
                print(k + 1, "of", len(tasks), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
