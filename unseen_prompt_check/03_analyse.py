r"""Unseen-prompt check, step 3: recovery, distortion and detection on the 200 new pairs.
Detectors: the universal-DCT CNN (no retraining); SPAM-686 + LDA trained on the 400 original
fine-tuned cover / DEDS stego pairs and applied to the new pairs; SPAM-686 + LDA trained and tested on
the new pairs (subject-grouped 5-fold cross-validation, 10 repetitions)."""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); AN = r"F:\Publication\scirep_submission\analysis"
os.chdir(AN); sys.path.insert(0, AN)
import numpy as np, pandas as pd, cv2, torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
import common as C, exp_steg as E
N = 200; names = [f"{i:03d}.png" for i in range(N)]; groups = np.arange(N) // 10
L = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
def pe(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.min(fpr + 1 - tpr) / 2)

cs = pd.read_csv(os.path.join(HERE, "cover_stats_unseen.csv")); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
log("== cover statistics, 200 unseen-prompt pairs (20 subjects not in the prompt bank, not texture close-ups)")
log("eligible coefficients per block: baseline %.2f +/- %.2f, fine-tuned %.2f +/- %.2f; fine-tuned higher in %d of 200; Wilcoxon p = %.2g"
    % (b.mean_eligible.mean(), b.mean_eligible.std(), f.mean_eligible.mean(), f.mean_eligible.std(), (f.mean_eligible > b.mean_eligible).sum(), wilcoxon(b.mean_eligible, f.mean_eligible).pvalue))
log("baseline covers on the fallback K: %d; cover-derived capacity: baseline without fallback %.3f bpp (n = %d), fine-tuned %.3f bpp; fine-tuned covers with capacity >= 0.30 bpp: %d"
    % (b.fallback.sum(), b[~b.fallback].capacity_bpp.mean(), (~b.fallback).sum(), f.capacity_bpp.mean(), (f.capacity_bpp >= 0.30).sum()))

e = pd.read_csv(os.path.join(HERE, "embed_unseen.csv")); eb = e[e.set == "base"].set_index("image").loc[names]; ef = e[e.set == "ft"].set_index("image").loc[names]
log("\n== embedding at 0.30 to 0.60 bpp (forced K, same message on both covers)")
log("errors: %d; exact recovery fine-tuned %d of 200, baseline %d of 200" % ((e.error.fillna("") != "").sum(), ef.exact.sum(), eb.exact.sum()))
log("forced K above the cover-derived K: fine-tuned %d, baseline %d of 200" % ((ef.K.values > f.loc[names].K.values).sum(), (eb.K.values > b.loc[names].K.values).sum()))
bins = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6001]; rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = ((ef.bpp >= lo) & (ef.bpp < hi)).values
    rows.append({"bpp": "%.2f to %.2f" % (lo, min(hi, 0.60)), "n": int(m.sum()), "psnr_base": eb.psnr_luma[m].mean(), "psnr_ft": ef.psnr_luma[m].mean(),
                 "d_psnr": (ef.psnr_luma[m].values - eb.psnr_luma[m].values).mean(), "p_psnr": wilcoxon(ef.psnr_luma[m].values, eb.psnr_luma[m].values).pvalue,
                 "ssim_base": eb.ssim_luma[m].mean(), "ssim_ft": ef.ssim_luma[m].mean(), "d_ssim": (ef.ssim_luma[m].values - eb.ssim_luma[m].values).mean()})
q = pd.DataFrame(rows); q.to_csv(os.path.join(HERE, "quality_bins_unseen.csv"), index=False); log(q.round(4).to_string(index=False))
log("overall luminance SSIM: fine-tuned %.3f, baseline %.3f; fine-tuned higher in %d of 200" % (ef.ssim_luma.mean(), eb.ssim_luma.mean(), (ef.ssim_luma.values > eb.ssim_luma.values).sum()))

spec = importlib.util.spec_from_file_location("Steganalyzer", r"F:\Thesis\Steganalyzer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = torch.device("cuda"); ck = torch.load(r"F:\Steganalyzer_UniversalDCT\model\universal_dct_srnet.pth", map_location=dev, weights_only=False)
model = S.SRNet().to(dev); model.load_state_dict(ck["state"]); model.eval(); TH = ck["val_threshold"]
@torch.no_grad()
def p(path):
    return float(torch.softmax(model(S.preprocess(cv2.imread(path)).to(dev)).float(), 1)[0, 1].cpu())
y = np.r_[np.zeros(N), np.ones(N)]; det = pd.DataFrame({"image": names})
log("\n== universal-DCT CNN (trained threshold %.4f), no retraining" % TH)
for tag, cov, stg in (("ft", "finetuned", "stego_finetuned"), ("base", "baseline", "stego_baseline")):
    pc = np.array([p(os.path.join(HERE, cov, n)) for n in names]); ps = np.array([p(os.path.join(HERE, stg, n)) for n in names])
    det[tag + "_p_cover"] = pc; det[tag + "_p_stego"] = ps
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        ix = rng.integers(0, N, N); bs.append(roc_auc_score(y, np.r_[pc[ix], ps[ix]]))
    log("%-4s AUC %.3f (95%% CI %.3f to %.3f), P_E %.3f | stego flagged at trained threshold %d, at 0.5 %d of 200 | covers flagged %d, %d"
        % (tag, roc_auc_score(y, np.r_[pc, ps]), *np.percentile(bs, [2.5, 97.5]), pe(y, np.r_[pc, ps]), (ps >= TH).sum(), (ps >= .5).sum(), (pc >= TH).sum(), (pc >= .5).sum()))
det.to_csv(os.path.join(HERE, "cnn_scores_unseen.csv"), index=False)

def feats(folder):
    pth = os.path.join(HERE, "feat_" + folder + ".npy")
    if os.path.exists(pth): return np.load(pth)
    X = np.stack([C.spam686(C.read_y(os.path.join(HERE, folder, n))) for n in names]); np.save(pth, X); return X
X = {k: feats(k) for k in ("finetuned", "stego_finetuned", "baseline", "stego_baseline")}
log("\n== SPAM-686 + LDA")
m = E.clf().fit(np.r_[E.F["ft_cover"], E.F["ft_deds"]], np.r_[np.zeros(400), np.ones(400)])
s = np.r_[m.decision_function(X["finetuned"]), m.decision_function(X["stego_finetuned"])]
log("trained on the 400 original fine-tuned pairs, tested on the 200 new fine-tuned pairs: AUC %.3f, P_E %.3f, error at the trained decision boundary %.3f"
    % (roc_auc_score(y, s), pe(y, s), ((s >= 0) != y).mean()))
for tag, c, st in (("fine-tuned", "finetuned", "stego_finetuned"), ("baseline", "baseline", "stego_baseline")):
    rng = np.random.default_rng(0); aucs = []; pes = []
    for rep in range(10):
        perm = rng.permutation(20); fold = np.empty(20, int); fold[perm] = np.arange(20) % 5; sc = np.zeros(2 * N)
        for k in range(5):
            te = fold[groups] == k; tr = ~te
            mm = E.clf().fit(np.r_[X[c][tr], X[st][tr]], np.r_[np.zeros(tr.sum()), np.ones(tr.sum())])
            sc[np.r_[te, np.zeros(N, bool)]] = mm.decision_function(X[c][te]); sc[np.r_[np.zeros(N, bool), te]] = mm.decision_function(X[st][te])
        aucs.append(roc_auc_score(y, sc)); pes.append(pe(y, sc))
    log("matched, %s new pairs, subject-grouped CV: AUC %.3f, P_E %.3f" % (tag, np.mean(aucs), np.mean(pes)))
open(os.path.join(HERE, "unseen_report.txt"), "w", encoding="utf-8").write("\n".join(L))
