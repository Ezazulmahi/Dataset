r"""Step 9 - test images from the submission: the DEDS-v5 stego of F:\scirep submission 2.0\4_evaluation.

The matched detectors are trained as before (universal covers, 0.4 bpp, step 06/07). Their DEDS test set is the
v5 stego the manuscript reports (normal mode, the paper's K and payload of 0.30-0.60 bpp, Table 2 protocol),
on the same covers as matched_0.4\covers_y\paper_* / unseen_*.
This writes the luminance Y of each v5 stego (OpenCV BGR->YCrCb, as in step 06) as 8-bit grayscale PNG to
matched_0.4\deds_v5_sub2\<set>\, so the detectors see exactly the format they were trained on.
The submission folder is only read. Resumable: existing files are skipped.
    python 09_prepare_v5_test.py
"""
import os, sys, json, time
from pathlib import Path
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SUB = Path(r"F:\scirep submission 2.0\4_evaluation")
SETS = {"paper_ft": ("stego_finetuned", 1000), "paper_bs": ("stego_baseline", 1000),
        "unseen_ft": ("stego_unseen_finetuned", 200), "unseen_bs": ("stego_unseen_baseline", 200)}

def main():
    out = C.V5_TEST; n_new = 0
    for name, (src, n) in SETS.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            p = out / name / f"{i:03d}.png"
            if p.exists():
                continue
            im = cv2.imread(str(SUB / src / f"{i:03d}.png"))
            if im is None:
                raise FileNotFoundError(SUB / src / f"{i:03d}.png")
            cov = C.MATCHED / "covers_y" / name / f"{i:03d}.png"
            if not cov.exists():
                raise FileNotFoundError(f"{cov} (build the covers with 06_build_matched.py)")
            tmp = p.with_name("tmp_" + p.name); cv2.imwrite(str(tmp), cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0]); os.replace(tmp, p)
            n_new += 1
        print(f"{name}: {n} images from {SUB / src}", flush=True)
    json.dump({"source": str(SUB), "sets": {k: v[0] for k, v in SETS.items()}, "plane": "Y (OpenCV BGR2YCrCb)",
               "created": time.strftime("%Y-%m-%d %H:%M")}, open(out / "source.json", "w"), indent=1)
    print(f"DONE ({n_new} written) -> {out}", flush=True)

if __name__ == "__main__":
    main()
