r"""Step 3 - train the steganalyser from scratch on the universal-DCT pairs.

Network: the SRNet-style architecture of F:\Thesis\Steganalyzer.py (fixed bank of 30 high-pass
filters, residual blocks, squeeze-and-excitation in the last two stages), randomly initialised.
No weights from the earlier detectors are loaded, and no image produced by the DEDS embedder
is used.

Split: 4,000 training / 500 validation / 500 test pairs (cover and its stego always together).
Saved to model/universal_dct_srnet.pth: weights of the epoch with the best validation accuracy,
the validation-optimal decision threshold, and the training log.
"""
import importlib.util, time, json
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
(ROOT / "model").mkdir(exist_ok=True)
SEED = 2026; EPOCHS = 25; PAIRS_PER_BATCH = 12; LR = 2e-4; WD = 1e-4; PATIENCE = 6
torch.manual_seed(SEED); np.random.seed(SEED)

spec = importlib.util.spec_from_file_location("Steganalyzer", r"F:\Thesis\Steganalyzer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = torch.device("cuda")

names = sorted(p.name for p in (ROOT / "covers").glob("*.png"))
assert len(names) == 5000, len(names)
def load(folder):
    arr = np.empty((len(names), 256, 256, 3), np.uint8)
    for i, n in enumerate(names):
        arr[i] = cv2.cvtColor(cv2.imread(str(ROOT / folder / n)), cv2.COLOR_BGR2RGB)
    return arr
print("loading images ...", flush=True)
C = load("covers"); T = load("stego")

perm = np.random.RandomState(SEED).permutation(len(names))
tr, va, te = perm[:4000], perm[4000:4500], perm[4500:]
pd.DataFrame({"image": names, "split": np.where(np.isin(np.arange(len(names)), tr), "train", np.where(np.isin(np.arange(len(names)), va), "val", "test"))}).to_csv(ROOT / "split.csv", index=False)

def to_tensor(u8):                      # same preprocessing as Steganalyzer.preprocess: RGB, [-1, 1]
    return (torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5

def augment(x):                         # random flips / 90-degree rotations, applied per image
    out = []
    for im in x:
        k = np.random.randint(4)
        im = np.rot90(im, k, axes=(0, 1))
        if np.random.rand() < 0.5: im = im[:, ::-1]
        out.append(np.ascontiguousarray(im))
    return np.stack(out)

@torch.no_grad()
def scores(model, idx):
    model.eval(); pc, ps = [], []
    for b in range(0, len(idx), 50):
        ii = idx[b:b + 50]
        with torch.autocast("cuda", dtype=torch.float16):
            pc.append(torch.softmax(model(to_tensor(C[ii]).to(dev)).float(), 1)[:, 1].cpu())
            ps.append(torch.softmax(model(to_tensor(T[ii]).to(dev)).float(), 1)[:, 1].cpu())
    return torch.cat(pc).numpy(), torch.cat(ps).numpy()

def metrics(pc, ps, th=0.5):
    y = np.r_[np.zeros(len(pc)), np.ones(len(ps))]; sc = np.r_[pc, ps]
    return {"acc": float(((sc >= th) == y).mean()), "auc": float(roc_auc_score(y, sc)), "fpr": float((pc >= th).mean()), "tpr": float((ps >= th).mean())}

model = S.SRNet().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
steps = EPOCHS * (len(tr) // PAIRS_PER_BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda"); crit = nn.CrossEntropyLoss()
log = []; best = -1; bad = 0; t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); order = np.random.permutation(tr); tl = 0; n = 0
    for b in range(0, len(order) - PAIRS_PER_BATCH + 1, PAIRS_PER_BATCH):
        ii = order[b:b + PAIRS_PER_BATCH]
        x = np.concatenate([C[ii], T[ii]]); y = torch.cat([torch.zeros(len(ii)), torch.ones(len(ii))]).long().to(dev)
        x = to_tensor(augment(x)).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    pc, ps = scores(model, va); m = metrics(pc, ps)
    row = {"epoch": ep, "train_loss": tl / n, "val_acc": m["acc"], "val_auc": m["auc"], "val_fpr": m["fpr"], "val_tpr": m["tpr"], "minutes": (time.time() - t0) / 60}
    log.append(row); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    if m["acc"] > best + 1e-6:
        best = m["acc"]; bad = 0
        fpr, tpr, th = roc_curve(np.r_[np.zeros(len(pc)), np.ones(len(ps))], np.r_[pc, ps])
        val_th = float(th[np.argmin(fpr + 1 - tpr)]); val_th = min(max(val_th, 0.0), 1.0)
        torch.save({"state": model.state_dict(), "epoch": ep, "val_acc": m["acc"], "val_auc": m["auc"], "val_threshold": val_th,
                    "arch": "SRNet-style (F:/Thesis/Steganalyzer.py), trained from scratch",
                    "train_data": "5,000 generated covers + universal DCT embedding (gen_dataset.embed_dct_random, rate 0.05-0.15)"},
                   ROOT / "model" / "universal_dct_srnet.pth")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stop", flush=True); break
pd.DataFrame(log).to_csv(ROOT / "model" / "training_log.csv", index=False)
ck = torch.load(ROOT / "model" / "universal_dct_srnet.pth", map_location=dev, weights_only=False); model.load_state_dict(ck["state"])
pc, ps = scores(model, te)
m05 = metrics(pc, ps, 0.5); mv = metrics(pc, ps, ck["val_threshold"])
print("BEST epoch", ck["epoch"], "val_acc", round(ck["val_acc"], 4), "val threshold", round(ck["val_threshold"], 4))
print("TEST (500 held-out universal-DCT pairs): acc@0.5 %.4f  acc@val-threshold %.4f  AUC %.4f" % (m05["acc"], mv["acc"], m05["auc"]), flush=True)
pd.DataFrame({"image": [names[i] for i in te], "p_stego_cover": pc, "p_stego_stego": ps}).to_csv(ROOT / "model" / "test_scores_universal_dct.csv", index=False)
print("DONE", flush=True)
