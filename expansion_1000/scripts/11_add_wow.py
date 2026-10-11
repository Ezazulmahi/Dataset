r"""Add the WOW reference embedder (analysis1000/steganalysis_wow.csv) to main.tex: Table 3 rows, the
detectability text, the Table 3 legend, Methods and the limitation sentence that said WOW was not run."""
import pandas as pd
SUB = "F:/Publication/scirep_submission/"
w = pd.read_csv("F:/scirep results/09_expansion_1000/analysis1000/steganalysis_wow.csv").set_index("experiment")
row = lambda emb, cov, det, r: f"{emb} & {cov} & {det} & {r.PE:.3f} & {r.AUC:.3f} ({r.AUC_lo:.3f} to {r.AUC_hi:.3f}) \\\\"
d_w = row("DEDS", "fine-tuned", "SPAM (WOW, same source)", w.loc["DEDS embedder, fine-tuned covers (detector trained on WOW)"])
w_f = row("WOW", "fine-tuned", "SPAM (WOW, same source)", w.loc["WOW, fine-tuned covers (matched detector)"])
w_b = row("WOW", "baseline", "SPAM (WOW, same source)", w.loc["WOW, baseline covers (matched detector)"])
s = open(SUB + "main.tex", encoding="utf-8").read()
R = [
 # Table 3 rows: DEDS scored by a WOW-trained detector after the S-UNIWARD-trained row; WOW rows after the S-UNIWARD rows
 ("DEDS & fine-tuned & SPAM (LSB matching, same source)", d_w + "\nDEDS & fine-tuned & SPAM (LSB matching, same source)"),
 ("LSB matching & fine-tuned & SPAM (LSB matching, same source)", w_f + "\n" + w_b + "\nLSB matching & fine-tuned & SPAM (LSB matching, same source)"),
 (r"S-UNIWARD and LSB matching are simulated at the entropy bound", r"S-UNIWARD, WOW and LSB matching are simulated at the entropy bound"),
 (r"Trained instead on S-UNIWARD\cite{holub2014uniward} or LSB-matching stego images", r"Trained instead on S-UNIWARD\cite{holub2014uniward}, WOW or LSB-matching stego images"),
 (r"For reference, S-UNIWARD and non-adaptive LSB matching were simulated", r"For reference, S-UNIWARD, WOW\cite{holub2012wow} and non-adaptive LSB matching were simulated"),
 (r"S-UNIWARD is substantially less detectable than the DEDS embedder on the same covers. Fine-tuned covers did not make S-UNIWARD or LSB matching harder",
  r"S-UNIWARD and WOW are substantially less detectable than the DEDS embedder on the same covers. Fine-tuned covers did not make them or LSB matching harder"),
 (r"and other spatial schemes such as WOW\cite{holub2012wow} were not run; the comparison is a reference point only.", r"and the spatial comparison is a reference point only."),
 (r"and LSB matching used uniform costs.", r"WOW\cite{holub2012wow} costs used the same filter bank with the aggregation exponent $p=-1$ of its reference implementation, and LSB matching used uniform costs."),
]
for o, n in R:
    assert s.count(o) == 1, (o[:70], s.count(o)); s = s.replace(o, n)
open(SUB + "main.tex", "w", encoding="utf-8", newline="").write(s)
print(d_w); print(w_f); print(w_b)
