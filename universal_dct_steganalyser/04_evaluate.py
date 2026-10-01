r"""Step 4 - evaluate the universal-DCT steganalyser on the 400 DEDS cover/stego pairs.

Covers: F:\Results\finetuned   Stego: F:\Results\stego_finetuned (the authors' DEDS stego images,
payload 0.30-0.60 bpp taken from comparison_results.xlsx).
The detector never saw the DEDS embedder or any of these 400 covers during training.

Reported per payload bin and over all pairs:
  accuracy at 0.5, accuracy at the threshold fixed on the universal-DCT validation set,
  accuracy at the best threshold on these test scores (optimistic for the detector),
  ROC-AUC, minimum average error P_E, FPR / TPR, mean P(stego) for covers and stego images.
Output: results/universal_dct_steganalysis_results.xlsx (+ a text report).
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("Steganalyzer", r"F:\Thesis\Steganalyzer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = torch.device("cuda")
ck = torch.load(ROOT / "model" / "universal_dct_srnet.pth", map_location=dev, weights_only=False)
model = S.SRNet().to(dev); model.load_state_dict(ck["state"]); model.eval()
VAL_TH = ck["val_threshold"]

@torch.no_grad()
def p_stego(path):
    x = S.preprocess(cv2.imread(str(path))).to(dev)
    return float(torch.softmax(model(x).float(), 1)[0, 1].cpu())

cmp_ = pd.read_excel(r"F:\Results\comparison_results.xlsx", sheet_name="per_image_results").set_index("image")

def run_set(cover_dir, stego_dir, label):
    rows = []
    for i in range(400):
        n = f"{i:03d}.png"
        pc = p_stego(Path(cover_dir) / n); ps = p_stego(Path(stego_dir) / n)
        rows.append({"image": n, "achieved_bpp": float(cmp_.loc[n, "achieved_bpp"]), "cover_p_stego": pc, "stego_p_stego": ps,
                     "cover_pred": "STEGO" if pc >= 0.5 else "COVER", "stego_pred": "STEGO" if ps >= 0.5 else "COVER",
                     "delta_stego_minus_cover": ps - pc})
    d = pd.DataFrame(rows)
    d["bpp_bin"] = pd.cut(d.achieved_bpp, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6001], include_lowest=True,
                          labels=["0.30-0.35", "0.35-0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60"])
    def summ(sub, name):
        y = np.r_[np.zeros(len(sub)), np.ones(len(sub))]; sc = np.r_[sub.cover_p_stego, sub.stego_p_stego]
        fpr, tpr, th = roc_curve(y, sc); k = int(np.argmin(fpr + 1 - tpr))
        acc = lambda t: float(((sc >= t) == y).mean())
        return {"set": label, "bpp_bin": name, "pairs": len(sub), "accuracy_at_0.5": acc(0.5), "accuracy_at_val_threshold": acc(VAL_TH),
                "val_threshold": VAL_TH, "accuracy_at_best_test_threshold": acc(th[k]), "best_test_threshold": float(th[k]),
                "roc_auc": float(roc_auc_score(y, sc)), "P_E_min": float((fpr[k] + 1 - tpr[k]) / 2),
                "fpr_at_0.5": float((sub.cover_p_stego >= 0.5).mean()), "tpr_at_0.5": float((sub.stego_p_stego >= 0.5).mean()),
                "fpr_at_val_threshold": float((sub.cover_p_stego >= VAL_TH).mean()), "tpr_at_val_threshold": float((sub.stego_p_stego >= VAL_TH).mean()),
                "mean_p_stego_given_cover": float(sub.cover_p_stego.mean()), "mean_p_stego_given_stego": float(sub.stego_p_stego.mean()),
                "stego_score_above_cover_frac": float((sub.stego_p_stego > sub.cover_p_stego).mean())}
    s = [summ(sub, b) for b, sub in d.groupby("bpp_bin", observed=True)] + [summ(d, "all")]
    rng = np.random.default_rng(0); y = np.r_[np.zeros(400), np.ones(400)]; bs = []
    for _ in range(2000):
        ix = rng.integers(0, 400, 400); bs.append(roc_auc_score(y, np.r_[d.cover_p_stego.values[ix], d.stego_p_stego.values[ix]]))
    s[-1]["roc_auc_ci_low"], s[-1]["roc_auc_ci_high"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
    d.insert(0, "set", label)
    return d, pd.DataFrame(s)

d1, s1 = run_set(r"F:\Results\finetuned", r"F:\Results\stego_finetuned", "DEDS on fine-tuned covers")
frames_d, frames_s = [d1], [s1]
base_stego = Path(r"F:\Publication\scirep_submission\analysis\stego_base")
if base_stego.exists() and len(list(base_stego.glob("*.png"))) == 400:
    d2, s2 = run_set(r"F:\Results\baseline", base_stego, "DEDS on baseline covers (re-embedding run)")
    frames_d.append(d2); frames_s.append(s2)
ft_stego2 = Path("F:/Publication/scirep_submission/analysis/stego_ft")
if ft_stego2.exists() and len(list(ft_stego2.glob("*.png"))) == 400:
    d3, s3 = run_set("F:/Results/finetuned", ft_stego2, "DEDS on fine-tuned covers (re-embedding run)")
    frames_d.append(d3); frames_s.append(s3)
per_image = pd.concat(frames_d, ignore_index=True); summary = pd.concat(frames_s, ignore_index=True)
tlog = pd.read_csv(ROOT / "model" / "training_log.csv")
tst = pd.read_csv(ROOT / "model" / "test_scores_universal_dct.csv")
yt = np.r_[np.zeros(len(tst)), np.ones(len(tst))]; st = np.r_[tst.p_stego_cover, tst.p_stego_stego]
gen = pd.DataFrame([{"set": "universal DCT, held-out test pairs", "pairs": len(tst), "accuracy_at_0.5": float(((st >= 0.5) == yt).mean()),
                     "accuracy_at_val_threshold": float(((st >= VAL_TH) == yt).mean()), "roc_auc": float(roc_auc_score(yt, st)),
                     "checkpoint_epoch": ck["epoch"], "val_acc": ck["val_acc"], "val_threshold": VAL_TH}])
with pd.ExcelWriter(ROOT / "results" / "universal_dct_steganalysis_results.xlsx", engine="openpyxl") as w:
    per_image.to_excel(w, sheet_name="per_image_detection", index=False)
    summary.to_excel(w, sheet_name="batch_summary", index=False)
    gen.to_excel(w, sheet_name="universal_dct_test", index=False)
    tlog.to_excel(w, sheet_name="training_log", index=False)
pd.set_option("display.width", 250)
cols = ["set", "bpp_bin", "pairs", "accuracy_at_0.5", "accuracy_at_val_threshold", "accuracy_at_best_test_threshold", "roc_auc", "P_E_min", "fpr_at_0.5", "tpr_at_0.5", "mean_p_stego_given_cover", "mean_p_stego_given_stego"]
txt = "Universal-DCT steganalyser evaluated on DEDS stego images\n\n" + gen.round(4).to_string(index=False) + "\n\n" + summary[cols].round(4).to_string(index=False) + \
      "\n\nAUC 95%% CI (all, fine-tuned): %.4f-%.4f\n" % (s1.iloc[-1].roc_auc_ci_low, s1.iloc[-1].roc_auc_ci_high)
open(ROOT / "results" / "universal_dct_steganalysis_report.txt", "w", encoding="utf-8").write(txt); print(txt)
