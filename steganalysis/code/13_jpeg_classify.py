r"""Step 13 - DCTR + FLD ensemble for J-UNIWARD and J-MiPOD (JPEG QF 75, 0.4 bpnzac), matched per method.

Same protocol as the spatial SRM table (11_srm_classify.py): ensemble trained on the 3,195 training pairs of the
matched split (JPEG covers / that method's stego), threshold from the 399 validation pairs, tested on
    paper_ft            the paper's 1,000 fine-tuned covers (JPEG QF 75) vs their J-UNIWARD / J-MiPOD stego
    universal_test406   the 406 held-out universal pairs
Output: results\dctr_fld\<method>_<set>_scores.csv, results\dctr_fld\summary.csv
    python 13_jpeg_classify.py
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
J = __import__("12_build_jpeg"); S = __import__("11_srm_classify")
OUT = C.RESULTS / "dctr_fld"

def main():
    from sealwatch.ensemble_classifier import FldEnsembleTrainer
    OUT.mkdir(parents=True, exist_ok=True)
    split = np.array([r[3] for r in C.matched_split("suniward")])     # same order as 12_build_jpeg.sets()["universal"]
    U, P = J.load("universal"), J.load("paper_ft")
    old = pd.read_csv(OUT / "summary.csv") if (OUT / "summary.csv").exists() else pd.DataFrame(columns=["method", "set"])
    rows = old.to_dict("records")                      # methods already classified are kept, not retrained
    tr, va, te = split == "train", split == "val", split == "test"
    for m in J.METHODS:
        if len(old[old.method == m]) >= 2:
            print(f"[{m}] already done - skipped", flush=True); continue
        rows = [r for r in rows if r["method"] != m]
        t0 = time.time()
        ens, _ = FldEnsembleTrainer(U["covers"][tr], U[m][tr], seed=2026, verbose=0).train()
        sv = S.soft(ens, np.r_[U["covers"][va], U[m][va]]); yv = np.r_[np.zeros(va.sum()), np.ones(va.sum())]
        _, thr = S.pe_thr(yv, sv)
        print(f"[{m}] trained in {(time.time() - t0) / 60:.1f} min: L={ens.num_base_learners}, d_sub={ens.d_sub}, "
              f"val AUC {roc_auc_score(yv, sv):.4f}", flush=True)
        for name, cov, stg in (("paper_ft", P["covers"], P[m]), ("universal_test406", U["covers"][te], U[m][te])):
            sc, ss = S.soft(ens, cov), S.soft(ens, stg); y = np.r_[np.zeros(len(sc)), np.ones(len(ss))]; s = np.r_[sc, ss]
            pd.DataFrame({"label": y, "score": s}).to_csv(OUT / f"{m}_{name}_scores.csv", index=False)
            pe, _ = S.pe_thr(y, s); lo, hi = S.boot_ci(sc, ss)
            r = {"method": m, "set": name, "pairs": len(sc), "auc": roc_auc_score(y, s), "auc_ci_low": lo, "auc_ci_high": hi,
                 "pe_min": pe, "acc_at_val_thr": float(((s > thr) == y).mean()), "stego_flagged": int((ss > thr).sum()),
                 "covers_flagged": int((sc > thr).sum()), "L": ens.num_base_learners, "d_sub": ens.d_sub}
            rows.append(r)
            print(f"   {name:18s} AUC {r['auc']:.4f} [{lo:.4f}, {hi:.4f}]  P_E {pe:.4f}  acc {r['acc_at_val_thr']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("DONE ->", OUT / "summary.csv", flush=True)

if __name__ == "__main__":
    main()
