r"""Expansion, step 6: the universal-DCT convolutional detector (F:\Steganalyzer_UniversalDCT, not
retrained) on all 1,000 pairs. Same statistics and sheet layout as 04_evaluate.py of that folder.
Its training covers used the 400 prompts of prompts.py with seeds from 100000, so the 60 new subjects
were never seen in training."""
import importlib.util, os
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from sklearn.metrics import roc_auc_score, roc_curve
HERE = Path(__file__).resolve().parent; OUT = HERE / "results_1000"; OUT.mkdir(exist_ok=True)
C1000 = HERE / "corpus1000"; N = 1000
spec = importlib.util.spec_from_file_location("Steganalyzer", r"F:\Thesis\Steganalyzer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = torch.device("cuda")
ck = torch.load(r"F:\Steganalyzer_UniversalDCT\model\universal_dct_srnet.pth", map_location=dev, weights_only=False)
model = S.SRNet().to(dev); model.load_state_dict(ck["state"]); model.eval(); VAL_TH = ck["val_threshold"]
@torch.no_grad()
def p_stego(path):
    return float(torch.softmax(model(S.preprocess(cv2.imread(str(path))).to(dev)).float(), 1)[0, 1].cpu())
cmp_ = pd.read_excel(C1000 / "comparison_results.xlsx", sheet_name="per_image_results").set_index("image")
LABELS = ["0.30-0.35", "0.35-0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60"]

def run_set(cover_dir, stego_dir, label):
    rows = []
    for i in range(N):
        n = f"{i:03d}.png"; pc = p_stego(Path(cover_dir) / n); ps = p_stego(Path(stego_dir) / n)
        rows.append({"image": n, "corpus_part": cmp_.loc[n, "corpus_part"], "achieved_bpp": float(cmp_.loc[n, "achieved_bpp"]), "cover_p_stego": pc, "stego_p_stego": ps,
                     "cover_pred": "STEGO" if pc >= 0.5 else "COVER", "stego_pred": "STEGO" if ps >= 0.5 else "COVER", "delta_stego_minus_cover": ps - pc})
    d = pd.DataFrame(rows)
    d["bpp_bin"] = pd.cut(d.achieved_bpp, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6001], include_lowest=True, labels=LABELS)
    def summ(sub, name):
        y = np.r_[np.zeros(len(sub)), np.ones(len(sub))]; sc = np.r_[sub.cover_p_stego, sub.stego_p_stego]
        fpr, tpr, th = roc_curve(y, sc); k = int(np.argmin(fpr + 1 - tpr)); acc = lambda t: float(((sc >= t) == y).mean())
        return {"set": label, "bpp_bin": name, "pairs": len(sub), "accuracy_at_0.5": acc(0.5), "accuracy_at_val_threshold": acc(VAL_TH), "val_threshold": VAL_TH,
                "accuracy_at_best_test_threshold": acc(th[k]), "best_test_threshold": float(th[k]), "roc_auc": float(roc_auc_score(y, sc)), "P_E_min": float((fpr[k] + 1 - tpr[k]) / 2),
                "fpr_at_0.5": float((sub.cover_p_stego >= 0.5).mean()), "tpr_at_0.5": float((sub.stego_p_stego >= 0.5).mean()),
                "fpr_at_val_threshold": float((sub.cover_p_stego >= VAL_TH).mean()), "tpr_at_val_threshold": float((sub.stego_p_stego >= VAL_TH).mean()),
                "stego_flagged_at_val_threshold": int((sub.stego_p_stego >= VAL_TH).sum()), "stego_flagged_at_0.5": int((sub.stego_p_stego >= 0.5).sum()),
                "covers_flagged_at_val_threshold": int((sub.cover_p_stego >= VAL_TH).sum()), "covers_flagged_at_0.5": int((sub.cover_p_stego >= 0.5).sum()),
                "mean_p_stego_given_cover": float(sub.cover_p_stego.mean()), "mean_p_stego_given_stego": float(sub.stego_p_stego.mean()),
                "stego_score_above_cover_frac": float((sub.stego_p_stego > sub.cover_p_stego).mean())}
    s = [summ(sub, b) for b, sub in d.groupby("bpp_bin", observed=True)] + [summ(sub, "part: " + b) for b, sub in d.groupby("corpus_part")] + [summ(d, "all")]
    rng = np.random.default_rng(0); y = np.r_[np.zeros(N), np.ones(N)]; bs = []; bc = []; subj = np.arange(N) // 10
    for _ in range(2000):
        ix = rng.integers(0, N, N); bs.append(roc_auc_score(y, np.r_[d.cover_p_stego.values[ix], d.stego_p_stego.values[ix]]))
        g = rng.integers(0, N // 10, N // 10); ix = np.concatenate([np.where(subj == k)[0] for k in g]); yy = np.r_[np.zeros(len(ix)), np.ones(len(ix))]
        bc.append(roc_auc_score(yy, np.r_[d.cover_p_stego.values[ix], d.stego_p_stego.values[ix]]))
    s[-1]["roc_auc_ci_low"], s[-1]["roc_auc_ci_high"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
    s[-1]["roc_auc_ci_low_subject_bootstrap"], s[-1]["roc_auc_ci_high_subject_bootstrap"] = [float(v) for v in np.percentile(bc, [2.5, 97.5])]
    d.insert(0, "set", label); return d, pd.DataFrame(s)

d1, s1 = run_set(C1000 / "finetuned", C1000 / "stego_finetuned", "DEDS on fine-tuned covers")
d2, s2 = run_set(C1000 / "baseline", HERE / "analysis1000" / "stego_base", "DEDS on baseline covers")
per = pd.concat([d1, d2], ignore_index=True); summary = pd.concat([s1, s2], ignore_index=True)
with pd.ExcelWriter(OUT / "universal_dct_steganalysis_results.xlsx", engine="openpyxl") as w:
    per.to_excel(w, sheet_name="per_image_detection", index=False); summary.to_excel(w, sheet_name="batch_summary", index=False)
    for sh in ("universal_dct_test", "training_log"):
        pd.read_excel(r"F:\Steganalyzer_UniversalDCT\results\universal_dct_steganalysis_results.xlsx", sheet_name=sh).to_excel(w, sheet_name=sh, index=False)
pd.set_option("display.width", 260)
cols = ["set", "bpp_bin", "pairs", "accuracy_at_val_threshold", "roc_auc", "P_E_min", "stego_flagged_at_val_threshold", "stego_flagged_at_0.5", "covers_flagged_at_0.5", "stego_score_above_cover_frac"]
txt = summary[cols].round(4).to_string(index=False) + "\n\n" + summary[summary.bpp_bin == "all"][["set", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "roc_auc_ci_low_subject_bootstrap", "roc_auc_ci_high_subject_bootstrap"]].round(4).to_string(index=False)
open(OUT / "universal_dct_steganalysis_report.txt", "w", encoding="utf-8").write(txt); print(txt)
