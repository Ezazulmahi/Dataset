r"""Shared paths, test-set definitions, image loading and the two detector architectures.

Detectors
  srnet   SRNet (Boroumand, Chen, Fridrich, IEEE TIFS 2019), written from the paper, three-channel
          input, random initialisation.
  ucdfnet UC-DFNet (github.com/sowmyathelukonti/UC-DFNet-Steganalysis), rewritten here: Dual-Path Enhancement
          and Fractal Downsampling blocks named after Fu et al., IEEE TIFS 2025, plus Coordinate Attention.
          A third-party interpretation of that paper, not the authors' code. Random initialisation.
  effnet  EfficientNet-B0 with the stride of the stem convolution removed and ImageNet weights
          (Yousfi, Butora, Fridrich, Tsang, IH&MMSec 2021, the ALASKA2 recipe). This is the
          strongest public detector family in recent comparisons.
"""
import os, hashlib
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
TRAIN_COVERS = Path(r"F:\Steganalyzer_UniversalDCT\covers")   # 5,000 covers, seeds 100000+, disjoint from the paper
TRAIN_STEGO = ROOT / "train_stego"                             # DEDS stego of those covers (step 01)
CACHE = ROOT / "cache"; RUNS = ROOT / "runs"; RESULTS = ROOT / "results"
R = Path(r"F:\scirep results")
UNIV = Path(r"F:\Steganalyzer_UniversalDCT")   # universal (generic) DCT embedding: covers/, stego/, split.csv (4000/500/500)
# The original UNIV covers are uncompressed while its stego images are JPEG-50 requantised, so a detector learns
# compression, not the message. universal_fixed/ (00_build_universal_fixed.py) has the same stego images with
# covers given the same requantisation; only the message differs. Training and testing use universal_fixed.
UNIV_FIXED = ROOT / "universal_fixed"

def universal_split(rate=None):
    """Rows (image, cover path, stego path, split) of the fixed universal-DCT set, using the original fixed split.
    rate: curriculum stego of 00c_build_universal_curriculum.py (universal_r<rate>\\stego); None = the real set."""
    import csv
    sd = UNIV_FIXED / "stego" if rate is None else ROOT / f"universal_r{rate}" / "stego"
    return [(r["image"], UNIV_FIXED / "covers" / r["image"], sd / r["image"], r["split"])
            for r in csv.DictReader(open(UNIV / "split.csv", newline=""))]

MATCHED = ROOT / "matched_0.4"      # 06_build_matched.py / 07_build_matched_deds.py: grayscale Y, 0.4 bpp random bits
MATCHED_COVERS_PER_GEN = 2000       # matched comparison uses universal covers ft/bs_00000-01999 only (4,000 covers:
                                    # 3,195 train / 399 val / 406 test with the fixed split); chosen to shorten the DEDS build

def matched_universal_names():
    import csv
    names = [r["image"] for r in csv.DictReader(open(UNIV / "split.csv", newline="")) if int(r["image"][3:8]) < MATCHED_COVERS_PER_GEN]
    return sorted(names, key=lambda n: (n[3:8], n[:2]))      # bs_i, ft_i interleaved

def matched_split(method, payload=0.4):
    """Training data of a matched detector: 4,000 universal covers (Y plane, MATCHED_COVERS_PER_GEN per generator)
    and their <method> stego, with the fixed split of the universal set (3,195 / 399 / 406).
    payload other than 0.4: the curriculum stage, stego from matched_<payload>\\ (06_build_matched.py --payload)."""
    import csv
    sd = ROOT / f"matched_{payload}" / method / "universal"
    return [(r["image"], MATCHED / "covers_y" / "universal" / r["image"], sd / r["image"], r["split"])
            for r in csv.DictReader(open(UNIV / "split.csv", newline="")) if int(r["image"][3:8]) < MATCHED_COVERS_PER_GEN]

V5_TEST = MATCHED / "deds_v5_sub2"   # 09_prepare_v5_test.py: Y of the v5 stego in F:\scirep submission 2.0

def _matched_test(method):
    sets = {"universal_test500": [(c, s) for _, c, s, sp in matched_split(method) if sp == "test"]}
    for name, n in (("paper_ft", 1000), ("paper_bs", 1000), ("unseen_ft", 200), ("unseen_bs", 200)):
        sets[name] = [(MATCHED / "covers_y" / name / f"{i:03d}.png", MATCHED / method / name / f"{i:03d}.png") for i in range(n)]
    if V5_TEST.exists():   # the manuscript's DEDS-v5 stego (paper payloads), scored by every matched detector
        for name, n in (("paper_ft", 1000), ("paper_bs", 1000), ("unseen_ft", 200), ("unseen_bs", 200)):
            sets[f"v5_{name}"] = [(MATCHED / "covers_y" / name / f"{i:03d}.png", V5_TEST / name / f"{i:03d}.png") for i in range(n)]
    return sets

def _paper(cov_dir, stego_dirs, n):
    pairs = []
    for i in range(n):
        name = f"{i:03d}.png"
        st = next((d / name for d in stego_dirs if (d / name).exists()), None)
        pairs.append((cov_dir / name, st))
    return pairs

def test_sets(data=None):
    """name -> list of (cover path, stego path). None of these images is used in training.
    For a matched run (data 'matched-<method>') the paper's covers embedded with that same method."""
    if data and data.startswith("matched-"):
        return _matched_test(data.split("-", 1)[1])
    c = R / "09_expansion_1000" / "corpus1000"
    u = R / "08_unseen_prompts"
    return {
        "universal_dct_test500": [(c, s) for _, c, s, sp in universal_split() if sp == "test"],   # in-domain check
        "paper1000_finetuned": _paper(c / "finetuned", [c / "stego_finetuned"], 1000),
        "paper1000_baseline": _paper(c / "baseline", [R / "03_revision_analysis" / "stego_base",
                                                      R / "09_expansion_1000" / "new600" / "stego_baseline"], 1000),
        "unseen200_finetuned": _paper(u / "finetuned", [u / "stego_finetuned"], 200),
        "unseen200_baseline": _paper(u / "baseline", [u / "stego_baseline"], 200),
        # the manuscript's DEDS-v5 fine-tuned stego (colour, as WSERNet was trained), Table 3 / Fig. 4c
        "v5_paper_ft": _paper(c / "finetuned", [Path(r"F:\scirep submission 2.0\4_evaluation\stego_finetuned")], 1000),
    }

N_FOLDS = 5

def paper_folds(n=1000, seed=2026):
    """Fold (0-4) of each paper index 0..n-1: a fixed random permutation cut into five equal parts."""
    f = np.empty(n, int); f[np.random.default_rng(seed).permutation(n)] = np.arange(n) % N_FOLDS
    return f

def read_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(im[:, :, ::-1])   # HWC uint8 RGB

def load_pairs(pairs, cache_file=None):
    """Return uint8 arrays (N,H,W,3) for covers and stego; cached as .npz so a resume loads in seconds."""
    if cache_file is not None and Path(cache_file).exists():
        z = np.load(cache_file)
        if len(z["cover"]) == len(pairs):
            return z["cover"], z["stego"]
    cov = np.stack([read_rgb(c) for c, _ in pairs]); stg = np.stack([read_rgb(s) for _, s in pairs])
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache_file) + ".tmp.npz"; np.savez(tmp, cover=cov, stego=stg); os.replace(tmp, cache_file)
    return cov, stg

def to_tensor(batch_u8, dev):
    x = torch.from_numpy(batch_u8).to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x - 127.5) / 64.0

def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

# ---------------------------------------------------------------- SRNet
class _T1(nn.Sequential):
    def __init__(s, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class _T2(nn.Module):
    def __init__(s, c):
        super().__init__(); s.b = nn.Sequential(_T1(c, c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
    def forward(s, x):
        return x + s.b(x)

class _T3(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.Sequential(_T1(i, o), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.AvgPool2d(3, 2, 1))
        s.sc = nn.Sequential(nn.Conv2d(i, o, 1, stride=2, bias=False), nn.BatchNorm2d(o))
    def forward(s, x):
        return s.b(x) + s.sc(x)

class SRNet(nn.Module):
    def __init__(s, in_ch=3):
        super().__init__()
        s.f = nn.Sequential(_T1(in_ch, 64), _T1(64, 16), *[_T2(16) for _ in range(5)],
                            _T3(16, 16), _T3(16, 64), _T3(64, 128), _T3(128, 256),
                            _T1(256, 512), nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.fc = nn.Linear(512, 2)
        for m in s.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    def forward(s, x):
        return s.fc(s.f(x))

# KV high-pass kernel (Ker & Bohme 2008; the fixed front of Qian et al. 2015 and Ye et al. 2017)
KV = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
                  dtype=torch.float32) / 12

class SRNetKV(nn.Module):
    """SRNet behind a fixed (not trained) KV high-pass filter, applied per channel and scaled by 4.
    Plain SRNet from scratch never left chance on the 3,195 matched pairs (S-UNIWARD, even at 1.0 bpp: val AUC
    0.500 after 500 steps and after 7 epochs at 0.4 bpp), while this variant reached val AUC 0.70 within 300 steps
    (19 Oct 2026). Report it as "SRNet with a fixed KV high-pass front end"."""
    def __init__(s):
        super().__init__()
        s.hp = nn.Conv2d(3, 3, 5, padding=2, bias=False, groups=3); s.hp.weight.data[:] = KV; s.hp.weight.requires_grad = False
        s.net = SRNet()
    def forward(s, x):
        return s.net(s.hp(x) * 4)

# ---------------------------------------------------------------- UC-DFNet
# Architecture of github.com/sowmyathelukonti/UC-DFNet-Steganalysis (models/ucdfnet.py, attention.py,
# fractal_block.py), rewritten here: Dual-Path Enhancement and Fractal Downsampling blocks named after
# Fu et al., IEEE TIFS 2025 (doi 10.1109/TIFS.2024.3493615), plus Coordinate Attention (Hou et al., CVPR 2021).
# It is a third-party interpretation, not the TIFS authors' network.
def _cbr(i, o, k=3, s=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o)]
    return nn.Sequential(*layers, nn.ReLU(inplace=True)) if act else nn.Sequential(*layers)

class DualPathEnhancement(nn.Module):
    def __init__(s, i, o):
        super().__init__(); m = o // 2
        s.res = nn.Sequential(_cbr(i, o), _cbr(o, o, act=False))
        s.sc = _cbr(i, o, 1, act=False) if i != o else nn.Identity()
        s.d1 = _cbr(i, m); s.d2 = _cbr(i + m, o)
        s.fuse = _cbr(2 * o, o, 1)
    def forward(s, x):
        p1 = torch.relu(s.res(x) + s.sc(x))
        p2 = s.d2(torch.cat([x, s.d1(x)], 1))
        return s.fuse(torch.cat([p1, p2], 1))

class CoordinateAttention(nn.Module):
    def __init__(s, c, reduction=16):
        super().__init__(); m = max(8, c // reduction)
        s.reduce = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m), nn.Hardswish(inplace=True))
        s.ah = nn.Conv2d(m, c, 1); s.aw = nn.Conv2d(m, c, 1)
    def forward(s, x):
        h, w = x.shape[2:]
        y = s.reduce(torch.cat([x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(2, 3)], 2))
        yh, yw = torch.split(y, [h, w], 2)
        return x * torch.sigmoid(s.ah(yh)) * torch.sigmoid(s.aw(yw.transpose(2, 3)))

class FractalDownsampling(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.b1 = _cbr(i, o, 3, 2)
        s.b2 = nn.Sequential(_cbr(i, o, 1), _cbr(o, o, 3, 2))
        s.b3 = nn.Sequential(nn.AvgPool2d(2, 2), _cbr(i, o))
        s.fuse = _cbr(3 * o, o, 1)
    def forward(s, x):
        return s.fuse(torch.cat([s.b1(x), s.b2(x), s.b3(x)], 1))

class UCDFNet(nn.Module):
    def __init__(s, in_ch=3, num_classes=2):
        super().__init__()
        stages = []
        for i, o in ((32, 64), (64, 128), (128, 256)):
            stages += [DualPathEnhancement(i, o), CoordinateAttention(o), FractalDownsampling(o, o)]
        s.f = nn.Sequential(_cbr(in_ch, 32), *stages, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.fc = nn.Linear(256, num_classes)
    def forward(s, x):
        return s.fc(s.f(x))

def build_model(kind, pretrained=True):
    if kind == "srnet":
        return SRNet()
    if kind == "srnet_kv":
        return SRNetKV()
    if kind == "ucdfnet":
        return UCDFNet()
    if kind == "wsernet":
        from wsernet import WSERNet
        return WSERNet()
    if kind == "effnet":
        import timm
        m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=2)
        m.conv_stem.stride = (1, 1)          # keep full resolution in the stem (Yousfi et al. 2021)
        return m
    raise ValueError(kind)

DEFAULTS = {  # epochs, pairs per batch, learning rate, weight decay
    # SRNet: lr 1e-3 collapsed to a constant output (loss ln 2, val AUC 0.50) on matched S-UNIWARD 0.4 bpp
    # (17-19 Oct 2026), so 2e-4, and the matched runs start from a 1.0 bpp curriculum stage (08_run_matched.py).
    "srnet": dict(epochs=80, pairs=16, lr=2e-4, wd=2e-4),
    "srnet_kv": dict(epochs=80, pairs=16, lr=2e-4, wd=2e-4),
    "effnet": dict(epochs=40, pairs=12, lr=5e-4, wd=1e-5),
    "ucdfnet": dict(epochs=60, pairs=8, lr=5e-4, wd=1e-4),
    # WSERNet recipe of the paper: SGD momentum 0.9, lr 0.01, wd 5e-4 (weights only), 16 pairs per batch,
    # lr / 10 at epochs 80, 140, 180 of 300; no early stopping. Here compressed proportionally to 100 epochs
    # (drops at 27, 47, 60 = 80/300, 140/300, 180/300 of training): the universal-DCT task converges far faster.
    # Deviation: 8 pairs per batch (16 do not fit in 8 GB) with lr halved to 0.005 (linear scaling rule).
    "wsernet": dict(epochs=100, pairs=8, lr=0.005, wd=5e-4, optim="sgd", milestones=[27, 47, 60], patience=10**6),
}
