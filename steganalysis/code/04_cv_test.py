r"""Step 4 (paper data) - score every paper stego image with the fold model that never saw it.

    python 04_cv_test.py --model srnet        # add --tta for the 8-view average, --force to redo

For fold k, runs/<model>_paper_f<k>/best.pt scores the test indices of that fold (both generators),
so the 1,000 fine-tuned and 1,000 baseline pairs each get an out-of-fold score. The 200-pair
unseen-prompt sets, never used in any fold, are scored by all five fold models and the margins
averaged (ensemble), with the mean of the five validation thresholds.

Output in results/<model>_paper_cv[_tta]/:
    fold<k>_scores.csv     per image, per fold model (cached: a finished fold is not rescored)
    <set>_scores.csv       final per-image scores used for the summary
    summary.csv            pooled ROC-AUC with 95% bootstrap CI, P_E, accuracy, stego / covers flagged,
                           and the mean and SD of the five per-fold AUCs
and results/summary_all.csv collects every run.
"""
import os, sys, argparse, importlib.util
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
spec = importlib.util.spec_from_file_location("t3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_test.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", default=None)
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    base = a.run or f"{a.model}_paper"; tag = f"{base}_cv" + ("_tta" if a.tta else "")
    out = C.RESULTS / tag; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    unseen = {k: v for k, v in C.test_sets().items() if k.startswith("unseen")}
    per_fold = []
    for k in range(C.N_FOLDS):
        run = C.RUNS / f"{base}_f{k}"; f = out / f"fold{k}_scores.csv"
        if not (run / "done.txt").exists():
            sys.exit(f"{run.name} has not finished training yet - run 02_train.py first")
        if f.exists() and not a.force:
            per_fold.append(pd.read_csv(f)); continue
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model = C.build_model(ck["cfg"]["model"], pretrained=False); model.load_state_dict(ck["model"])
        model = model.to(dev).to(memory_format=torch.channels_last)
        sp = pd.read_csv(run / "split.csv"); te = sp[sp.split == "test"]
        groups = {"paper1000_finetuned": te[te.id.str.startswith("ft_")], "paper1000_baseline": te[te.id.str.startswith("bs_")]}
        rows = []
        for name, g in groups.items():
            pairs = list(zip(g.cover, g.stego)); ids = list(g.id)
            rows.append((name, ids, pairs))
        for name, pairs in unseen.items():
            pairs = [p for p in pairs if p[1] is not None]
            rows.append((name, [f"u_{i:03d}" for i in range(len(pairs))], pairs))
        dfs = []
        for name, ids, pairs in rows:
            cov, stg = C.load_pairs(pairs)
            m = T.scores(model, np.concatenate([cov, stg]), dev, a.tta)
            dfs.append(pd.DataFrame({"set": name, "fold": k, "pair": ids * 2, "image": [str(c) for c, _ in pairs] + [str(s) for _, s in pairs],
                                     "label": [0] * len(pairs) + [1] * len(pairs), "margin": m, "threshold": ck["threshold"]}))
        d = pd.concat(dfs); d.to_csv(f, index=False); per_fold.append(d)
        print(f"fold {k} scored (best epoch {ck['epoch']}, val P_E {ck['val_pe']:.4f})", flush=True)
    allf = pd.concat(per_fold)
    summ = []
    for name in ["paper1000_finetuned", "paper1000_baseline", *unseen]:
        d = allf[allf.set == name]
        if name.startswith("unseen"):        # ensemble of the five fold models
            d = d.groupby(["pair", "image", "label"], as_index=False).agg(margin=("margin", "mean"), threshold=("threshold", "mean"))
            fold_aucs = [roc_auc_score(g.label, g.margin) for _, g in allf[allf.set == name].groupby("fold")]
            how = "ensemble of 5 fold models"
        else:
            fold_aucs = [roc_auc_score(g.label, g.margin) for _, g in d.groupby("fold")]
            how = "out-of-fold"
        d = d.sort_values(["label", "pair"]).reset_index(drop=True)
        d["flagged"] = (d.margin > d.threshold).astype(int); d["p_stego"] = 1 / (1 + np.exp(-np.clip(d.margin, -50, 50)))
        d.to_csv(out / f"{name}_scores.csv", index=False)
        thr = d.threshold.values
        r = T.summarise(d, 0.0)                      # threshold-free numbers; flagged counts below use each fold's threshold
        n = len(d) // 2
        r.update({"acc_at_val_thr": float((d.flagged == d.label).mean()), "stego_flagged": int(d.flagged[d.label == 1].sum()),
                  "covers_flagged": int(d.flagged[d.label == 0].sum()), "val_threshold": float(np.mean(thr)),
                  "fold_auc_mean": float(np.mean(fold_aucs)), "fold_auc_sd": float(np.std(fold_aucs, ddof=1)), "scored_by": how})
        r = {"run": tag, "set": name, **r}; summ.append(r)
        print(f"{name:22s} pairs {n:4d}  AUC {r['auc']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]  folds {r['fold_auc_mean']:.4f}+-{r['fold_auc_sd']:.4f}  "
              f"P_E {r['pe_min']:.4f}  acc {r['acc_at_val_thr']:.4f}  stego flagged {r['stego_flagged']}/{n}  covers flagged {r['covers_flagged']}/{n}", flush=True)
    sm = pd.DataFrame(summ); sm.to_csv(out / "summary.csv", index=False)
    af = C.RESULTS / "summary_all.csv"
    if af.exists():
        old = pd.read_csv(af); sm = pd.concat([old[old.run != tag], sm])
    sm.to_csv(af, index=False)

if __name__ == "__main__":
    main()
