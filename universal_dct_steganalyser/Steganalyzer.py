"""
Steganalyzer.py  —  SRNet v13 inference & batch evaluation
===========================================================
Architecture and preprocessing match the v13 training pipeline exactly:
  - SRM collapses 3-channel RGB mean before applying residual filters
  - Input tensor: BGR->RGB, [0,1]->[-1,1], center-crop to 256x256
  - Blk: Conv-BN-ReLU-Identity-Conv-BN  (Dropout2d disabled in eval)
  - SE squeeze-excitation on s3/s4 stages

Naming convention for paired folders:
  same filename  :  covers/img001.png  +  stego/img001.png
  prefix pair    :  covers/cover_001.png  +  stego/stego_001.png
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["COVER", "STEGO"]


# ─────────────────────────────────────────────────────────────────
# SRM FILTER BANK  (30 high-pass kernels, matches v13 training)
# ─────────────────────────────────────────────────────────────────
def _srm_kernels():
    F = []; z = lambda: np.zeros((5, 5))

    f = z(); f[2, 1:4] = [-1, 2, -1];       F.append(f / 2)
    f = z(); f[1:4, 2] = [-1, 2, -1];       F.append(f / 2)
    f = z(); f[2, 0:5] = [-1, 2, -6, 2, -1]; F.append(f / 4)
    f = z(); f[:, 2]   = [-1, 2, -6, 2, -1]; F.append(f / 4)
    f = z(); f[1,1] = f[3,3] = -1; f[2,2] = 2; F.append(f / 2)
    f = z(); f[1,3] = f[3,1] = -1; f[2,2] = 2; F.append(f / 2)
    f = z(); f[2, 1:5] = [-1, 3, -3, 1];    F.append(f / 2)
    f = z(); f[1:5, 2] = [-1, 3, -3, 1];    F.append(f / 2)
    F.append(np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,-4,2,-1],
                        [0,0,2,0,0],[0,0,-1,0,0]]) / 4)
    F.append(np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                        [0,-1,2,-1,0],[0,0,0,0,0]]) / 4)
    kb = np.array([5,5,5,-3,0,-3,-3,-3,-3], dtype=float).reshape(3,3) / 15
    for r in range(8):
        f = z(); f[1:4, 1:4] = np.rot90(kb, r % 4) * (1 if r < 4 else -1)
        F.append(f)
    for dy, dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
        f = z(); f[2,2] = 1; f[2+dy, 2+dx] = -1; F.append(f)
    while len(F) < 30:
        f = z(); f[2,2] = 1; f[2,1] = f[2,3] = f[1,2] = f[3,2] = -0.25
        F.append(f)

    return torch.tensor(np.stack(F[:30])[:, None], dtype=torch.float32)


class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", _srm_kernels())

    def forward(self, x):
        # x: (B, 3, H, W) in [-1, 1]
        x = x.mean(dim=1, keepdim=True)          # collapse RGB -> 1-channel
        return torch.tanh(F.conv2d(x * 255.0, self.w, padding=2) / 3.0)


# ─────────────────────────────────────────────────────────────────
# NETWORK BLOCKS  (exact v13 architecture)
# ─────────────────────────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // r, bias=False), nn.ReLU(True),
            nn.Linear(c // r, c, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class Blk(nn.Module):
    def __init__(self, ci, co, se=False, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co))
        self.skip = nn.Identity() if ci == co else nn.Conv2d(ci, co, 1, bias=False)
        self.se   = SE(co) if se else nn.Identity()

    def forward(self, x):
        return F.relu(self.se(self.net(x)) + self.skip(x), True)


def _dn(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(co), nn.ReLU(True))


class SRNet(nn.Module):
    def __init__(self, nc=2, drop=0.7):
        super().__init__()
        self.srm   = SRM()
        self.entry = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True))
        self.s1 = nn.Sequential(Blk(64, 64, drop=0.05),  Blk(64, 64))
        self.d1 = _dn(64, 64)
        self.s2 = nn.Sequential(Blk(64, 128, drop=0.05), Blk(128, 128))
        self.d2 = _dn(128, 128)
        self.s3 = nn.Sequential(Blk(128, 256, se=True, drop=0.1),
                                 Blk(256, 256, se=True))
        self.d3 = _dn(256, 256)
        self.s4 = nn.Sequential(Blk(256, 512, se=True, drop=0.1),
                                 Blk(512, 512, se=True))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(512, nc))

    def forward(self, x):
        x = self.entry(self.srm(x))
        x = self.d1(self.s1(x))
        x = self.d2(self.s2(x))
        x = self.d3(self.s3(x))
        return self.head(self.s4(x))


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
def load_model(path, allow_partial=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SRNet().to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state" in ckpt:
        state = ckpt["state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor)
                                         for v in ckpt.values()):
        state = ckpt
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: "
                         f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)

    print(f"Device            : {device}")
    print(f"Checkpoint tensors: {len(state)}")

    real_missing = [k for k in result.missing_keys if not k.endswith("srm.w")]
    if result.missing_keys:
        print(f"MISSING keys      : {len(result.missing_keys)}")
        for k in result.missing_keys:
            print(f"    missing  {k}")
    if result.unexpected_keys:
        print(f"UNEXPECTED keys   : {len(result.unexpected_keys)}")
        for k in result.unexpected_keys:
            print(f"    unused   {k}")
    if any(k.endswith("srm.w") for k in state):
        print("SRM filters       : loaded from checkpoint")

    if real_missing and not allow_partial:
        raise RuntimeError(
            f"{len(real_missing)} layers have random weights — "
            f"architecture mismatch. First few: {real_missing[:4]}")

    model.eval()
    print("Status            : OK — all layers loaded\n")
    return model, device


# ─────────────────────────────────────────────────────────────────
# PREPROCESS  (matches training _to_tensor exactly)
# ─────────────────────────────────────────────────────────────────
def preprocess(img_bgr, size=256):
    """BGR uint8 -> (1, 3, H, W) float32 tensor in [-1, 1], center-cropped."""
    h, w = img_bgr.shape[:2]
    h0   = max(0, (h - size) // 2)
    w0   = max(0, (w - size) // 2)
    if h >= size and w >= size:
        patch = img_bgr[h0:h0 + size, w0:w0 + size]
    else:
        patch = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
    t   = (t - 0.5) / 0.5                          # [-1, 1]
    return t.unsqueeze(0)                           # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────
# SINGLE-IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def detect(model, img_path, device):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    prob = torch.softmax(model(preprocess(img).to(device)), dim=1)[0].cpu().numpy()
    pred = int(np.argmax(prob))
    print(f"  {os.path.basename(img_path):<30} "
          f"{CLASS_NAMES[pred]:<6}  "
          f"cover={prob[0]*100:.1f}%  stego={prob[1]*100:.1f}%")
    return pred, prob


# ─────────────────────────────────────────────────────────────────
# BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, cover_dir, stego_dir, device):
    """
    Evaluate over matched cover/stego folder pairs.
    Reports accuracy + ROC-AUC (requires sklearn).

    Matching rules (tried in order):
      1. Exact same filename in both dirs
      2. cover_XXX.ext  <->  stego_XXX.ext  (strip cover_/stego_ prefix)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def _stem_key(fname):
        s = os.path.splitext(fname)[0].lower()
        for pfx in ("cover_", "stego_", "cover", "stego"):
            if s.startswith(pfx):
                return s[len(pfx):]
        return s

    cover_map = {}
    for f in os.listdir(cover_dir):
        if f.lower().endswith(IMG_EXT):
            cover_map[_stem_key(f)] = os.path.join(cover_dir, f)

    stego_files = sorted(f for f in os.listdir(stego_dir)
                         if f.lower().endswith(IMG_EXT))

    y_true, y_score, y_pred = [], [], []
    skipped = 0

    for nm in stego_files:
        cp = cover_map.get(_stem_key(nm))
        sp = os.path.join(stego_dir, nm)
        if cp is None:
            skipped += 1
            continue
        for path, label in ((cp, 0), (sp, 1)):
            img = cv2.imread(path)
            if img is None:
                continue
            prob = torch.softmax(
                model(preprocess(img).to(device)), dim=1
            )[0].cpu().numpy()
            y_true.append(label)
            y_score.append(float(prob[1]))
            y_pred.append(int(prob[1] >= 0.5))

    if skipped:
        print(f"  Warning: {skipped} stego file(s) had no matching cover — skipped.")

    n = len(y_true)
    if n == 0:
        print("No matched pairs found. Check folder paths and naming.")
        return None

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = np.array(y_pred)
    acc     = float((y_true == y_pred).mean())

    n_cover = int((y_true == 0).sum())
    n_stego = int((y_true == 1).sum())

    print(f"\n{'='*56}")
    print(f"  BATCH RESULT  ({n} images: {n_cover} cover / {n_stego} stego)")
    print(f"{'='*56}")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  (50% = random chance = your method evades the detector)")

    if roc_auc_score is not None and len(set(y_true.tolist())) == 2:
        auc = roc_auc_score(y_true, y_score)
        print(f"  ROC-AUC           : {auc:.4f}  (0.50 = perfect evasion)")

    print(f"  mean P(stego|cover): {y_score[y_true==0].mean():.3f}")
    print(f"  mean P(stego|stego): {y_score[y_true==1].mean():.3f}")

    delta = y_score[y_true==1].mean() - y_score[y_true==0].mean()
    if abs(delta) < 0.02:
        print(f"  VERDICT: model cannot distinguish cover from stego (delta={delta:+.3f})")
    elif delta > 0:
        print(f"  VERDICT: model leans toward detecting stego (delta={delta:+.3f})")
    else:
        print(f"  VERDICT: model scores stego LOWER than cover (delta={delta:+.3f}) "
              f"— completely blind")
    print(f"{'='*56}\n")
    return acc


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── checkpoint (fine-tuned on DCT_Adaptive) ────────────────────
    CKPT = os.path.join(_DIR, "steganalyzer_finetuned.pth")

    # ── your original 5 hand-picked pairs ─────────────────────────
    COVER_PATH = r"F:\Thesis\finetunecover\cover_p2_s46.png"
    STEGO_PATH = r"F:\Thesis\finetunestego\stego_p2_s46.png"

    COVER_DIR  = r"F:\Thesis\finetunecover"
    STEGO_DIR  = r"F:\Thesis\finetunestego"

    # ── downloaded + embedded dataset (200 pairs) ──────────────────
    EXTRA_COVER_DIR = r"F:\Thesis\finetune_extra\covers"
    EXTRA_STEGO_DIR = r"F:\Thesis\finetune_extra\stego"

    # ── load finetuned model ───────────────────────────────────────
    model, device = load_model(CKPT)

    # ── single-image check on every image in finetunecover/stego ──
    print("Single-image check (your images):")
    print(f"  {'file':<30} {'pred':<6}  cover%   stego%")
    print(f"  {'-'*52}")
    detect(model, r"F:\Thesis\finetunecover\cover_p1_s44.png", device)
    detect(model, r"F:\Thesis\finetunestego\stego_p1_s44.png", device)
    detect(model, r"F:\Thesis\finetunecover\cover_p2_s46.png", device)
    detect(model, r"F:\Thesis\finetunestego\stego_p2_s46.png", device)
    detect(model, r"F:\Thesis\finetunecover\cover_p2_s47.png", device)
    detect(model, r"F:\Thesis\finetunestego\stego_p2_s47.png", device)
    detect(model, r"F:\Thesis\finetunecover\cover_p5_s53.png", device)
    detect(model, r"F:\Thesis\finetunestego\stego_p5_s53.png", device)
    detect(model, r"F:\Thesis\finetunecover\cover_p8_s59.png", device)
    detect(model, r"F:\Thesis\finetunestego\stego_p8_s59.png", device)

    # ── batch: your 5 hand-picked pairs ───────────────────────────
    print("\nDataset 1 — your finetunecover / finetunestego (5 pairs):")
    evaluate(model, COVER_DIR, STEGO_DIR, device)

    # ── batch: full 200-pair downloaded dataset ────────────────────
    if os.path.isdir(EXTRA_COVER_DIR) and os.path.isdir(EXTRA_STEGO_DIR):
        print("Dataset 2 — downloaded + DCT_Adaptive embedded (200 pairs):")
        evaluate(model, EXTRA_COVER_DIR, EXTRA_STEGO_DIR, device)
