"""Final text patches after the re-embedding run: replication sentence, detection wording,
Supplementary Table S4 rows, Data availability statement."""
import pandas as pd
s = open("../main.tex", encoding="utf-8").read()
t = open("../supplementary.tex", encoding="utf-8").read()

def rep(text, old, new):
    assert old in text, old[:70]
    return text.replace(old, new)

# 1. replication of the distortion pattern with independent messages
old = "Neither metric is a measure of security."
new = ("The re-embedding run, which used different messages, reproduced this pattern: the mean PSNR difference per bin was "
       "$+0.60$, $+0.39$, $+0.10$, $+0.11$, $-0.39$ and $-0.50$\\,dB from the lowest to the highest payload bin, the mean SSIM difference rose from $+0.050$ to $+0.108$, "
       "and SSIM was again higher for the fine-tuned cover in all 400 pairs. Neither metric is a measure of security.")
if "reproduced this pattern" not in s:
    s = rep(s, old, new)

# 2. detection: both cover types
old = "With knowledge of the embedder it detects DEDS stego images almost perfectly."
new = "With knowledge of the embedder it detects DEDS stego images almost perfectly, on fine-tuned and on baseline covers alike."
if "on baseline covers alike" not in s:
    s = rep(s, old, new)

# 3. Data availability
a = s.index("\\section*{Data availability}"); b = s.index("\\section*{Funding}")
s = s[:a] + ("\\section*{Data availability}\n"
    "The fine-tuning, embedding and evaluation code is available at \\url{https://github.com/Imraj-Rabbani/P3-Image-Steganography-Pipeline}. "
    "The 400-pair evaluation corpus, the stego images, the LoRA adapter weights, the prompt bank, the evaluation scripts and the per-image result files are available at \\url{https://github.com/Ezazulmahi/Dataset}; "
    "the folder \\texttt{revision\\_analysis} of that repository contains the scripts and result files for the cover statistics, the re-embedding run, the feature-based steganalysis, the reference embedders, the loss-weight grid and all figures of this article, the folder \\texttt{universal\\_dct\\_steganalyser} contains the trained convolutional detector with its data-generation, training and evaluation scripts and per-image results, and the folder \\texttt{repeated\\_message\\_test} contains the code and results of the repeated-message test. "
    "The training images are drawn from the public \\texttt{gmongaras/Imagenet21K\\_Recaption} dataset on Hugging Face and can be regenerated with the released filtering script. "
    "The reference photographs used for the spectral comparison were obtained from the Lorem Picsum service; their identifiers are listed in the repository.\n\n") + s[b:]
open("../main.tex", "w", encoding="utf-8").write(s)

# 4. Supplementary Table S4 rows
rt = pd.read_csv("runtime_table.csv")
rows = []
for _, r in rt.iterrows():
    rows.append(f"{r.bpp} & {r.covers} & {int(r.n)} & {r.embed_med:.0f} ({r.q25:.0f} to {r.q75:.0f}) & {r['max']:.0f} & {r.fix_med:.0f} & {r.extract_med:.2f} \\\\")
a = t.index("%S4START"); b = t.index("%S4END")
t = t[:a] + "%S4START\n" + "\n".join(rows) + "\n" + t[b:]
open("../supplementary.tex", "w", encoding="utf-8").write(t)
print("\n".join(rows))
