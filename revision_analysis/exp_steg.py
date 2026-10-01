"""Feature-based steganalysis (SPAM-686 + regularised linear discriminant) with
subject-grouped cross-validation, on the luminance plane of the 400 cover pairs."""
import os, sys, numpy as np, pandas as pd, common as C
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve

N = 400
names = [f"{i:03d}.png" for i in range(N)]
groups = np.arange(N) // 10                      # 40 subjects x 10 prompt variants
cmp_ = pd.read_excel(C.RES + r"\comparison_results.xlsx", sheet_name="per_image_results").set_index("image")
nbits = np.array([int(cmp_.loc[n, "n_bits"]) for n in names])
bpp = nbits / 65536.0
MY_FT = "stego_ft"; MY_BASE = "stego_base"       # written by exp_embed.py (optional)

def feats(tag, fn):
    p = f"feat_{tag}.npy"
    if os.path.exists(p):
        return np.load(p)
    X = np.stack([C.spam686(fn(i)) for i in range(N)]); np.save(p, X); return X

Yft = [C.read_y(C.FT_DIR + "\\" + n) for n in names]
Ybs = [C.read_y(C.BASE_DIR + "\\" + n) for n in names]
F = {}
F["ft_cover"] = feats("ft_cover", lambda i: Yft[i])
F["bs_cover"] = feats("bs_cover", lambda i: Ybs[i])
F["ft_deds"] = feats("ft_deds", lambda i: C.read_y(C.STEGO_FT_DIR + "\\" + names[i]))
F["ft_suni"] = feats("ft_suni", lambda i: C.suniward(Yft[i], nbits[i], 1000 + i))
F["bs_suni"] = feats("bs_suni", lambda i: C.suniward(Ybs[i], nbits[i], 1000 + i))
F["ft_lsbm"] = feats("ft_lsbm", lambda i: C.lsb_matching(Yft[i], nbits[i], 2000 + i))
F["bs_lsbm"] = feats("bs_lsbm", lambda i: C.lsb_matching(Ybs[i], nbits[i], 2000 + i))
if os.path.isdir(MY_BASE) and len(os.listdir(MY_BASE)) == N:
    F["bs_deds"] = feats("bs_deds", lambda i: C.read_y(MY_BASE + "\\" + names[i]))
if os.path.isdir(MY_FT) and len(os.listdir(MY_FT)) == N:
    F["ft_deds_rerun"] = feats("ft_deds_rerun", lambda i: C.read_y(MY_FT + "\\" + names[i]))

def clf():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

def pe_at(th, sc, y):
    pred = sc >= th
    return 0.5 * ((pred & (y == 0)).sum() / (y == 0).sum() + ((~pred) & (y == 1)).sum() / (y == 1).sum())

def best_th(sc, y):
    fpr, tpr, th = roc_curve(y, sc)
    return th[np.argmin(fpr + 1 - tpr)]

def run(cover, stego_train, stego_test, reps=10, folds=5, seed=0):
    """Train cover-vs-stego_train on training subjects, test cover-vs-stego_test on held-out subjects."""
    rng = np.random.default_rng(seed)
    pes, aucs, bs, oof_c, oof_s = [], [], [], None, None
    y = np.r_[np.zeros(N), np.ones(N)]
    for r in range(reps):
        perm = rng.permutation(40); fold_of = np.empty(40, int); fold_of[perm] = np.arange(40) % folds
        sc_c = np.zeros(N); sc_s = np.zeros(N); pe_f = []
        for k in range(folds):
            te = fold_of[groups] == k; tr = ~te
            Xtr = np.r_[cover[tr], stego_train[tr]]; ytr = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
            m = clf().fit(Xtr, ytr)
            th = best_th(m.decision_function(Xtr), ytr)      # threshold fixed on training data only
            sc_c[te] = m.decision_function(cover[te]); sc_s[te] = m.decision_function(stego_test[te])
            yte = np.r_[np.zeros(te.sum()), np.ones(te.sum())]
            pe_f.append(pe_at(th, np.r_[sc_c[te], sc_s[te]], yte))
        pes.append(np.mean(pe_f)); aucs.append(roc_auc_score(y, np.r_[sc_c, sc_s]))
        for _ in range(200):                                  # bootstrap over image pairs, pooled over repetitions
            ix = rng.integers(0, N, N); bs.append(roc_auc_score(y, np.r_[sc_c[ix], sc_s[ix]]))
        if r == 0: oof_c, oof_s = sc_c.copy(), sc_s.copy()
    return {"PE": np.mean(pes), "PE_sd": np.std(pes), "AUC": np.mean(aucs), "AUC_lo": np.percentile(bs, 2.5),
            "AUC_hi": np.percentile(bs, 97.5), "oof_c": oof_c, "oof_s": oof_s}

if __name__ == "__main__":
    exps = [
        ("DEDS embedder, fine-tuned covers (matched detector)", "ft_cover", "ft_deds", "ft_deds"),
        ("S-UNIWARD, fine-tuned covers (matched detector)", "ft_cover", "ft_suni", "ft_suni"),
        ("LSB matching, fine-tuned covers (matched detector)", "ft_cover", "ft_lsbm", "ft_lsbm"),
        ("S-UNIWARD, baseline covers (matched detector)", "bs_cover", "bs_suni", "bs_suni"),
        ("LSB matching, baseline covers (matched detector)", "bs_cover", "bs_lsbm", "bs_lsbm"),
        ("DEDS embedder, fine-tuned covers (detector trained on S-UNIWARD)", "ft_cover", "ft_suni", "ft_deds"),
        ("DEDS embedder, fine-tuned covers (detector trained on LSB matching)", "ft_cover", "ft_lsbm", "ft_deds"),
    ]
    if "bs_deds" in F:
        exps.insert(1, ("DEDS embedder, baseline covers (matched detector)", "bs_cover", "bs_deds", "bs_deds"))
    if "ft_deds_rerun" in F:
        exps.insert(1, ("DEDS embedder, fine-tuned covers, re-embedded (matched detector)", "ft_cover", "ft_deds_rerun", "ft_deds_rerun"))
    rows = []; scores = {}
    edges = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6001]
    for name, c, st, se in exps:
        r = run(F[c], F[st], F[se])
        row = {"experiment": name, "PE": r["PE"], "PE_sd": r["PE_sd"], "AUC": r["AUC"], "AUC_lo": r["AUC_lo"], "AUC_hi": r["AUC_hi"]}
        for a, b in zip(edges[:-1], edges[1:]):
            s = (bpp >= a) & (bpp < b)
            row[f"AUC_{a:.2f}"] = roc_auc_score(np.r_[np.zeros(s.sum()), np.ones(s.sum())], np.r_[r["oof_c"][s], r["oof_s"][s]])
        rows.append(row); scores[name] = (r["oof_c"], r["oof_s"])
        print(f"{name:75s} PE={r['PE']:.3f}±{r['PE_sd']:.3f}  AUC={r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv("steganalysis_spam.csv", index=False)
    np.savez("steganalysis_spam_scores.npz", **{f"e{i}_{k}": v for i, (n, cs) in enumerate(scores.items()) for k, v in zip("cs", cs)},
             names=np.array(list(scores.keys())), bpp=bpp)
    print(pd.DataFrame(rows).drop(columns=["experiment"]).round(3).to_string())

