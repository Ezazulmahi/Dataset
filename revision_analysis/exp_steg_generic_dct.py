"""Detector trained on a generic ('normal') DCT embedding, tested on the DEDS embedder.

Generic embedding = the authors' embed_dct_random (gen_dataset.py): luminance block DCT,
standard JPEG luminance table, LSB replacement of quantised coefficients larger than 1.
Two payload settings: (a) random rate 5-15% of the embeddable coefficients, as used when the
authors' CNN was first trained; (b) as many bits as the DEDS message, capped by availability."""
import importlib.util, numpy as np, pandas as pd, cv2, os
import common as C
import exp_steg as E            # reuses features, grouping, run(), nbits, names
spec = importlib.util.spec_from_file_location("gen_dataset", "F:/Thesis/gen_dataset.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
N = 400; names = E.names; nbits = E.nbits

def y_of(bgr):
    return cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0]

def n_embeddable(bgr):
    Y = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    cnt = 0
    for y in range(0, 256, 8):
        for x in range(0, 256, 8):
            q = np.around(cv2.dct(Y[y:y+8, x:x+8]) / G.JPEG_STD_LUM_QUANT_TABLE).ravel()
            cnt += int((np.round(q[1:]) > 1).sum())
    return cnt

def gen(i, mode):
    cover = cv2.imread(C.FT_DIR + "/" + names[i])
    np.random.seed(7000 + i)
    if mode == "rate":
        rate = np.random.uniform(0.05, 0.15)
    else:
        avail = max(1, n_embeddable(cover)); rate = min(1.0, nbits[i] / avail)
    return y_of(G.embed_dct_random(cover, rate)), rate

info = []
def feats(tag, mode):
    p = f"feat_{tag}.npy"
    if os.path.exists(p): return np.load(p)
    X = []
    for i in range(N):
        y, rate = gen(i, mode); X.append(C.spam686(y)); info.append((tag, i, rate))
    X = np.stack(X); np.save(p, X); return X

F = E.F
F["ft_gdct_rate"] = feats("ft_gdct_rate", "rate")
F["ft_gdct_match"] = feats("ft_gdct_match", "match")
avail = np.array([n_embeddable(cv2.imread(C.FT_DIR + "/" + n)) for n in names])
print("embeddable coefficients (standard table, >1) per fine-tuned cover: mean %.0f min %d max %d ; DEDS bits mean %.0f ; covers where available >= DEDS bits: %d of 400"
      % (avail.mean(), avail.min(), avail.max(), nbits.mean(), (avail >= nbits).sum()))
rows = []
for name, c, st, se in [
    ("Generic DCT (5-15% rate), fine-tuned covers (matched detector)", "ft_cover", "ft_gdct_rate", "ft_gdct_rate"),
    ("DEDS embedder, fine-tuned covers (detector trained on generic DCT, 5-15% rate)", "ft_cover", "ft_gdct_rate", "ft_deds"),
    ("Generic DCT (payload-matched), fine-tuned covers (matched detector)", "ft_cover", "ft_gdct_match", "ft_gdct_match"),
    ("DEDS embedder, fine-tuned covers (detector trained on generic DCT, payload-matched)", "ft_cover", "ft_gdct_match", "ft_deds")]:
    r = E.run(F[c], F[st], F[se])
    rows.append({"experiment": name, "PE": r["PE"], "PE_sd": r["PE_sd"], "AUC": r["AUC"], "AUC_lo": r["AUC_lo"], "AUC_hi": r["AUC_hi"]})
    print(f"{name:85s} PE={r['PE']:.3f}  AUC={r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}]", flush=True)
pd.DataFrame(rows).to_csv("steganalysis_generic_dct.csv", index=False)
