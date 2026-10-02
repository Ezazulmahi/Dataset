r"""T11, step 2 - detection of DEDS stego images carrying random bits versus ASCII text, same 50 covers,
same K and payload length. (a) SPAM-686 + LDA trained on the 400 ASCII pairs with the test subjects
held out (subject-grouped 5-fold, 10 repetitions); (b) the universal-DCT CNN, no retraining."""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
AN = r"F:\Publication\scirep_submission\analysis"
os.chdir(AN); sys.path.insert(0, AN)
import numpy as np, pandas as pd, cv2, torch
from sklearn.metrics import roc_auc_score, roc_curve
import common as C, exp_steg as E

IDX = np.arange(0, 400, 8); N = 400; groups = np.arange(N) // 10
RND = os.path.join(HERE, "t11_stego_random")
Xc = E.F["ft_cover"]; Xa = E.F["ft_deds"]
Xr = np.stack([C.spam686(C.read_y(os.path.join(RND, f"{i:03d}.png"))) for i in IDX])
sel = np.zeros(N, bool); sel[IDX] = True; pos = {i: k for k, i in enumerate(IDX)}

def pe(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.min(fpr + 1 - tpr) / 2)

rng = np.random.default_rng(0); res = []
for rep in range(10):
    perm = rng.permutation(40); fold_of = np.empty(40, int); fold_of[perm] = np.arange(40) % 5
    sc = np.zeros(50); sa = np.zeros(50); sr = np.zeros(50)
    for k in range(5):
        te = fold_of[groups] == k; tr = ~te
        m = E.clf().fit(np.r_[Xc[tr], Xa[tr]], np.r_[np.zeros(tr.sum()), np.ones(tr.sum())])
        ii = np.where(te & sel)[0]; kk = [pos[i] for i in ii]
        sc[kk] = m.decision_function(Xc[ii]); sa[kk] = m.decision_function(Xa[ii]); sr[kk] = m.decision_function(Xr[kk])
    y = np.r_[np.zeros(50), np.ones(50)]
    res.append({"rep": rep, "auc_ascii": roc_auc_score(y, np.r_[sc, sa]), "auc_random": roc_auc_score(y, np.r_[sc, sr]),
                "pe_ascii": pe(y, np.r_[sc, sa]), "pe_random": pe(y, np.r_[sc, sr]),
                "mean_score_ascii": sa.mean(), "mean_score_random": sr.mean(), "auc_random_vs_ascii": roc_auc_score(y, np.r_[sa, sr])})
spam = pd.DataFrame(res)

spec = importlib.util.spec_from_file_location("Steganalyzer", r"F:\Thesis\Steganalyzer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = torch.device("cuda")
ck = torch.load(r"F:\Steganalyzer_UniversalDCT\model\universal_dct_srnet.pth", map_location=dev, weights_only=False)
model = S.SRNet().to(dev); model.load_state_dict(ck["state"]); model.eval(); TH = ck["val_threshold"]
@torch.no_grad()
def p(path):
    return float(torch.softmax(model(S.preprocess(cv2.imread(path)).to(dev)).float(), 1)[0, 1].cpu())
def psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))
rows = []
for i in IDX:
    n = f"{i:03d}.png"; c = os.path.join(C.FT_DIR, n); a = os.path.join(C.STEGO_FT_DIR, n); r = os.path.join(RND, n)
    yc = C.read_y(c)
    rows.append({"image": n, "p_cover": p(c), "p_ascii": p(a), "p_random": p(r),
                 "psnr_y_ascii": psnr(yc, C.read_y(a)), "psnr_y_random": psnr(yc, C.read_y(r))})
d = pd.DataFrame(rows); y = np.r_[np.zeros(50), np.ones(50)]
from scipy.stats import wilcoxon
txt = ["T11 - random-bit versus ASCII payloads, 50 fine-tuned covers (every eighth image), same K and length",
       "", "SPAM-686 + LDA trained on ASCII DEDS pairs, test subjects held out (mean over 10 repetitions)",
       spam.drop(columns="rep").mean().round(4).to_string(),
       "", "Universal-DCT CNN (threshold from training %.4f)" % TH,
       "AUC cover vs ASCII stego  %.4f" % roc_auc_score(y, np.r_[d.p_cover, d.p_ascii]),
       "AUC cover vs random stego %.4f" % roc_auc_score(y, np.r_[d.p_cover, d.p_random]),
       "flagged at trained threshold: cover %d, ASCII %d, random %d of 50" % ((d.p_cover >= TH).sum(), (d.p_ascii >= TH).sum(), (d.p_random >= TH).sum()),
       "flagged at 0.5:               cover %d, ASCII %d, random %d of 50" % ((d.p_cover >= .5).sum(), (d.p_ascii >= .5).sum(), (d.p_random >= .5).sum()),
       "mean P(stego): cover %.4f, ASCII %.4f, random %.4f" % (d.p_cover.mean(), d.p_ascii.mean(), d.p_random.mean()),
       "paired Wilcoxon P(stego) ASCII vs random: p = %.3g" % wilcoxon(d.p_ascii, d.p_random).pvalue,
       "", "Luminance PSNR: ASCII %.3f dB, random %.3f dB, mean paired difference %.3f dB, Wilcoxon p = %.3g"
       % (d.psnr_y_ascii.mean(), d.psnr_y_random.mean(), (d.psnr_y_random - d.psnr_y_ascii).mean(), wilcoxon(d.psnr_y_ascii, d.psnr_y_random).pvalue)]
spam.to_csv(os.path.join(HERE, "t11_spam.csv"), index=False); d.to_csv(os.path.join(HERE, "t11_cnn.csv"), index=False)
open(os.path.join(HERE, "t11_report.txt"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))
