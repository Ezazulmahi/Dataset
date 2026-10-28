r"""Step 3 - score the paper's stego images with a trained detector (runs/<run>/best.pt).

    python 03_test.py --run effnet            # add --tta to average the 8 rotations/flips, --force to redo
    python 03_test.py --run srnet --ckpt last # use the latest weights instead of the best epoch

Test sets (none used in training): paper1000_finetuned, paper1000_baseline, unseen200_finetuned,
unseen200_baseline. Resumable: a set whose scores CSV exists is skipped.

Output in results/<run>[_tta]/:
    <set>_scores.csv   one row per image (cover and stego): logit margin (stego minus cover logit), p(stego), label, decision at the validation threshold
    summary.csv        per set: ROC-AUC with 95% bootstrap CI (pairs resampled), P_E min, accuracy at the
                       validation threshold and at 0.5, stego / covers flagged, TPR at 5% FPR
and results/summary_all.csv collects every run.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

@torch.no_grad()
def scores(model, imgs, dev, tta, bs=32):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = C.to_tensor(imgs[i:i + bs], dev); acc = 0
        views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
        for k, f in views:
            v = torch.rot90(x, k, (2, 3)); v = v.flip(3) if f else v
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(v.contiguous(memory_format=torch.channels_last)).float(); acc = acc + (o[:, 1] - o[:, 0])
        out.append((acc / len(views)).cpu().numpy())
    return np.concatenate(out)

def summarise(df, thr, n_boot=2000):
    y, s = df.label.values, df.margin.values
    fpr, tpr, _ = roc_curve(y, s); pe = 0.5 * (fpr + 1 - tpr)
    n = len(df) // 2; cov, stg = s[:n], s[n:]; rng = np.random.default_rng(0); boots = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        boots.append(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[cov[k], stg[k]]))
    return {"pairs": n, "auc": roc_auc_score(y, s), "auc_ci_low": np.percentile(boots, 2.5), "auc_ci_high": np.percentile(boots, 97.5),
            "pe_min": pe.min(), "acc_at_val_thr": ((s > thr) == y).mean(), "acc_at_margin0": ((s > 0) == y).mean(),
            "stego_flagged": int((stg > thr).sum()), "covers_flagged": int((cov > thr).sum()),
            "tpr_at_fpr5": float(np.interp(0.05, fpr, tpr)), "val_threshold": thr}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    dev = torch.device("cuda"); run = C.RUNS / a.run
    ck = torch.load(run / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; thr = ck.get("threshold", ck.get("best", {}).get("thr", 0.5))
    model = C.build_model(cfg["model"], pretrained=False); model.load_state_dict(ck["model"])
    model = model.to(dev).to(memory_format=torch.channels_last)
    tag = a.run + ("_last" if a.ckpt == "last" else "") + ("_tta" if a.tta else "")
    out = C.RESULTS / tag; out.mkdir(parents=True, exist_ok=True)
    print(f"{a.run} {a.ckpt}.pt (epoch {ck.get('epoch')}), validation threshold {thr:.4f}", flush=True)
    rows = []
    for name, pairs in C.test_sets(cfg.get("data")).items():
        if a.sets and name not in a.sets:
            continue
        missing = [p for p in pairs if p[1] is None or not p[0].exists()]
        pairs = [p for p in pairs if p not in missing]
        f = out / f"{name}_scores.csv"
        if f.exists() and not a.force:
            df = pd.read_csv(f)
        else:
            cov, stg = C.load_pairs(pairs)
            s = scores(model, np.concatenate([cov, stg]), dev, a.tta)
            df = pd.DataFrame({"image": [str(c) for c, _ in pairs] + [str(t) for _, t in pairs],
                               "label": [0] * len(pairs) + [1] * len(pairs), "margin": s})
            df["p_stego"] = 1 / (1 + np.exp(-np.clip(s, -50, 50)))
            df["flagged"] = (df.margin > thr).astype(int); df.to_csv(f, index=False)
        r = {"run": tag, "set": name, "missing_pairs": len(missing), **summarise(df, thr)}; rows.append(r)
        print(f"{name:22s} pairs {r['pairs']:4d}  AUC {r['auc']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]  P_E {r['pe_min']:.4f}  "
              f"acc {r['acc_at_val_thr']:.4f}  stego flagged {r['stego_flagged']}/{r['pairs']}  covers flagged {r['covers_flagged']}/{r['pairs']}", flush=True)
    sm = pd.DataFrame(rows); sm.to_csv(out / "summary.csv", index=False)
    allf = C.RESULTS / "summary_all.csv"
    if allf.exists():
        old = pd.read_csv(allf); sm = pd.concat([old[~((old.run == tag) & old.set.isin(sm.set))], sm])
    sm.to_csv(allf, index=False)

if __name__ == "__main__":
    main()
