"""WOW as an additional spatial reference embedder: same luminance planes, same number of payload bits
per image as the DEDS message, simulated at the entropy bound, SPAM-686 + LDA with the subject-grouped
protocol of exp_steg.py. Also: DEDS stego images scored by a detector trained on WOW."""
import numpy as np, pandas as pd
import common as C, exp_steg as E, wow as W

F = E.F
F["ft_wow"] = E.feats("ft_wow", lambda i: W.wow(E.Yft[i], E.nbits[i], 3000 + i))
F["bs_wow"] = E.feats("bs_wow", lambda i: W.wow(E.Ybs[i], E.nbits[i], 3000 + i))
rows = []
for name, c, st, se in [("WOW, fine-tuned covers (matched detector)", "ft_cover", "ft_wow", "ft_wow"),
                        ("WOW, baseline covers (matched detector)", "bs_cover", "bs_wow", "bs_wow"),
                        ("DEDS embedder, fine-tuned covers (detector trained on WOW)", "ft_cover", "ft_wow", "ft_deds")]:
    r = E.run(F[c], F[st], F[se])
    rows.append({"experiment": name, "PE": r["PE"], "PE_sd": r["PE_sd"], "AUC": r["AUC"], "AUC_lo": r["AUC_lo"], "AUC_hi": r["AUC_hi"]})
    print(f"{name:65s} PE={r['PE']:.3f}  AUC={r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}]", flush=True)
pd.DataFrame(rows).to_csv("steganalysis_wow.csv", index=False)
