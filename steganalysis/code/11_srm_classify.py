r"""Step 11 - SRM + FLD ensemble (rich model + linear classifier), matched per method.

For each method (suniward wow hill mipod deds_v4): the Kodovsky-Fridrich-Holub ensemble of Fisher linear
discriminants (IEEE TIFS 2012; sealwatch FldEnsembleTrainer, automatic L and d_sub) is trained on the SRM
features (10_srm_extract.py) of the 3,195 training pairs of the matched split (covers / that method's 0.4 bpp
stego). The 399 validation pairs give the decision threshold. Tested on:
    paper_ft            the paper's 1,000 fine-tuned covers vs that method's stego of them; for deds_v4 these are
                        the 1,000 DEDS-v5 fine-tuned stego of F:\scirep submission 2.0 (Table 3, Fig. 4c)
    v5_paper_ft         every detector on the DEDS-v5 stego as well (cross-method check)
    universal_test406   in-domain check: the 406 held-out universal covers vs that method's stego
Missing features are extracted first (10_srm_extract.py, resumable).
Score: mean projection of the base learners (soft vote) for AUC and P_E; accuracy at the validation threshold.
Output: results\srm_fld\<method>_<set>_scores.csv, results\srm_fld\summary.csv
    python 11_srm_classify.py [--methods ...]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
SRM = __import__("10_srm_extract")
OUT = C.RESULTS / "srm_fld"

def soft(ens, X):
    return np.mean([bl.predict(X) for bl in ens.base_learners], axis=0)

def pe_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s); pe = 0.5 * (fpr + 1 - tpr); k = int(np.argmin(pe))
    return float(pe[k]), float(thr[k])

def boot_ci(cov, stg, n=2000):
    rng = np.random.default_rng(0); m = len(cov); b = []
    for _ in range(n):
        k = rng.integers(0, m, m); b.append(roc_auc_score(np.r_[np.zeros(m), np.ones(m)], np.r_[cov[k], stg[k]]))
    return np.percentile(b, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="+", default=SRM.METHODS); a = ap.parse_args()
    from sealwatch.ensemble_classifier import FldEnsembleTrainer
    import subprocess
    if subprocess.call([sys.executable, str(C.ROOT / "10_srm_extract.py"), "--workers", "5"]) != 0:
        sys.exit("SRM extraction stopped - run again")
    OUT.mkdir(parents=True, exist_ok=True)
    split = np.array([r[3] for r in C.matched_split("suniward")])     # same row order as features_srm/*.txt
    Xc = SRM.load("covers_universal")
    Xpc, Xv5 = SRM.load("covers_paper_ft"), SRM.load("v5_paper_ft")
    old = pd.read_csv(OUT / "summary.csv") if (OUT / "summary.csv").exists() else pd.DataFrame(columns=["method", "set"])
    rows = old.to_dict("records")                      # methods already classified are kept, not retrained
    for m in a.methods:
        if len(old[old.method == m]) >= 3:
            print(f"[{m}] already done - skipped", flush=True); continue
        rows = [r for r in rows if r["method"] != m]
        Xs = SRM.load(f"{m}_universal"); t0 = time.time()
        tr, va, te = split == "train", split == "val", split == "test"
        ens, _ = FldEnsembleTrainer(Xc[tr], Xs[tr], seed=2026, verbose=0).train()
        sv = soft(ens, np.r_[Xc[va], Xs[va]]); yv = np.r_[np.zeros(va.sum()), np.ones(va.sum())]
        _, thr = pe_thr(yv, sv)
        print(f"[{m}] trained in {(time.time() - t0) / 60:.1f} min: L={ens.num_base_learners}, d_sub={ens.d_sub}, "
              f"val AUC {roc_auc_score(yv, sv):.4f}", flush=True)
        own = Xv5 if m == "deds_v4" else SRM.load(f"{m}_paper_ft")
        for name, cov, stg in (("paper_ft", Xpc, own), ("v5_paper_ft", Xpc, Xv5), ("universal_test406", Xc[te], Xs[te])):
            sc, ss = soft(ens, cov), soft(ens, stg); y = np.r_[np.zeros(len(sc)), np.ones(len(ss))]; s = np.r_[sc, ss]
            pd.DataFrame({"label": y, "score": s}).to_csv(OUT / f"{m}_{name}_scores.csv", index=False)
            pe, _ = pe_thr(y, s); lo, hi = boot_ci(sc, ss)
            r = {"method": m, "set": name, "pairs": len(sc), "auc": roc_auc_score(y, s), "auc_ci_low": lo, "auc_ci_high": hi,
                 "pe_min": pe, "acc_at_val_thr": float(((s > thr) == y).mean()), "stego_flagged": int((ss > thr).sum()),
                 "covers_flagged": int((sc > thr).sum()), "L": ens.num_base_learners, "d_sub": ens.d_sub}
            rows.append(r)
            print(f"   {name:18s} AUC {r['auc']:.4f} [{lo:.4f}, {hi:.4f}]  P_E {pe:.4f}  acc {r['acc_at_val_thr']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("DONE ->", OUT / "summary.csv", flush=True)

if __name__ == "__main__":
    main()
