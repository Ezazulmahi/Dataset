r"""Update main.tex and supplementary.tex from 400 to 1,000 pairs: every number that does not depend on
the detectors. Values come from report_1000.txt (07_report_1000.py). Each replacement must match once."""
import re
SUB = "F:/Publication/scirep_submission/"
s = open(SUB + "main.tex", encoding="utf-8").read()
t2 = open("F:/scirep results/09_expansion_1000/table2_rows.tex", encoding="utf-8").read().strip()

R = [
# macros (re-embedding run of pairs 0-399 plus the expansion run of pairs 400-999)
(r"\newcommand{\RecFT}{400 of 400}", r"\newcommand{\RecFT}{1,000 of 1,000}"),
(r"\newcommand{\RecBase}{400 of 400}", r"\newcommand{\RecBase}{1,000 of 1,000}"),
(r"\newcommand{\FixMaxFT}{16}", r"\newcommand{\FixMaxFT}{17}"),
(r"\newcommand{\PSNRcolFT}{$32.11\pm2.20$}", r"\newcommand{\PSNRcolFT}{$32.16\pm2.16$}"),
(r"\newcommand{\PSNRcolBase}{$31.63\pm2.50$}", r"\newcommand{\PSNRcolBase}{$31.51\pm2.40$}"),
# abstract
(r"On 400 prompt- and seed-matched $256\times256$ cover pairs, fine-tuning raised the mean number of eligible coefficients per $8\times8$ block from 1.52 to 16.51 and the mean cover-derived capacity from 0.053 (233 baseline covers) to 0.386 bits per pixel (bpp).",
 r"On 1,000 prompt- and seed-matched $256\times256$ cover pairs, fine-tuning raised the mean number of eligible coefficients per $8\times8$ block from 1.57 to 16.64 and the mean cover-derived capacity from 0.056 (566 baseline covers) to 0.389 bits per pixel (bpp)."),
# introduction
(r"The third is a controlled evaluation on 400 prompt- and seed-matched cover pairs", r"The third is a controlled evaluation on 1,000 prompt- and seed-matched cover pairs"),
# corpus
(r"The corpus contains 400 baseline/fine-tuned cover pairs at $256\times256$ pixels, one pair for each prompt in a fixed bank of 400 prompts (40 texture subjects, each combined with 10 photographic modifiers).",
 r"The corpus contains 1,000 baseline/fine-tuned cover pairs at $256\times256$ pixels, one pair for each prompt in a fixed bank of 1,000 prompts (100 texture subjects, each combined with 10 photographic modifiers; Methods)."),
# DCT statistics
(r"Fine-tuning raised $\bar e$ from $1.52\pm1.29$ to $16.51\pm2.66$ (mean $\pm$ s.d., $n=400$; Table~\ref{tab:covers}), an increase observed in all 400 pairs (paired Wilcoxon signed-rank test, $p=2.7\times10^{-67}$). The coefficient of variation of $\bar e$ fell from 0.85 to 0.16,",
 r"Fine-tuning raised $\bar e$ from $1.57\pm1.40$ to $16.64\pm2.60$ (mean $\pm$ s.d., $n=1{,}000$; Table~\ref{tab:covers}), an increase observed in all 1,000 pairs (paired Wilcoxon signed-rank test, $p=3.3\times10^{-165}$). The coefficient of variation of $\bar e$ fell from 0.89 to 0.16,"),
(r"rose from 0.051 to 0.352.", r"rose from 0.051 to 0.354."),
(r"no position in $\mathcal{M}$ is eligible in more than 16\% of blocks, whereas in fine-tuned covers the positions nearest the anti-diagonal are eligible in 62 to 82\% of blocks and the highest-frequency positions in 8 to 13\%.",
 r"no position in $\mathcal{M}$ is eligible in more than 17\% of blocks, whereas in fine-tuned covers the positions nearest the anti-diagonal are eligible in 62 to 82\% of blocks and the highest-frequency positions in 8 to 14\%."),
# Table 1
(r"\caption{\textbf{Cover statistics before embedding} ($n=400$ baseline/fine-tuned pairs", r"\caption{\textbf{Cover statistics before embedding} ($n=1{,}000$ baseline/fine-tuned pairs"),
(r"Block-DCT energy ratio, $3\le u+v\le10$ & $0.051\pm0.028$ & $0.352\pm0.066$ \\", r"Block-DCT energy ratio, $3\le u+v\le10$ & $0.051\pm0.030$ & $0.354\pm0.066$ \\"),
(r"Eligible coefficients per block, $\bar e$ & $1.52\pm1.29$ & $16.51\pm2.66$ \\", r"Eligible coefficients per block, $\bar e$ & $1.57\pm1.40$ & $16.64\pm2.60$ \\"),
(r"\quad median (range) & 1.17 (0.01 to 7.04) & 16.62 (7.35 to 23.10) \\", r"\quad median (range) & 1.17 (0.00 to 10.75) & 16.68 (6.74 to 24.81) \\"),
(r"Covers with $\lfloor\bar e\rfloor=0$ (fallback $K$ applied) & 167 of 400 & 0 of 400 \\", r"Covers with $\lfloor\bar e\rfloor=0$ (fallback $K$ applied) & 434 of 1,000 & 0 of 1,000 \\"),
(r"$K$, all covers (fallback included) & $11.41\pm10.77$ & $23.74\pm4.02$ \\", r"$K$, all covers (fallback included) & $11.88\pm10.74$ & $23.92\pm3.94$ \\"),
(r"$K$, cover-derived only & $2.38\pm1.91$ ($n=233$) & $23.74\pm4.02$ ($n=400$) \\", r"$K$, cover-derived only & $2.59\pm2.14$ ($n=566$) & $23.92\pm3.94$ ($n=1{,}000$) \\"),
(r"Raw slot capacity (bpp), all covers & $0.193\pm0.168$ & $0.386\pm0.063$ \\", r"Raw slot capacity (bpp), all covers & $0.201\pm0.167$ & $0.389\pm0.061$ \\"),
(r"Raw slot capacity (bpp), cover-derived only & $0.053\pm0.030$ ($n=233$) & $0.386\pm0.063$ ($n=400$) \\", r"Raw slot capacity (bpp), cover-derived only & $0.056\pm0.033$ ($n=566$) & $0.389\pm0.061$ ($n=1{,}000$) \\"),
(r"\quad range, cover-derived only & 0.031 to 0.172 & 0.172 to 0.546 \\", r"\quad range, cover-derived only & 0.031 to 0.250 & 0.156 to 0.577 \\"),
(r"Covers with cover-derived capacity $\ge0.30$\bpp & 0 of 233 & 363 of 400 \\", r"Covers with cover-derived capacity $\ge0.30$\bpp & 0 of 566 & 924 of 1,000 \\"),
# Fig 2 caption
(r"averaged over 400 baseline (\textbf{a}) and 400 fine-tuned (\textbf{b}) covers.", r"averaged over 1,000 baseline (\textbf{a}) and 1,000 fine-tuned (\textbf{b}) covers."),
(r"the 167 baseline covers that fall back to the fixed $K=24$ are omitted.", r"the 434 baseline covers that fall back to the fixed $K=24$ are omitted."),
# CLIP / FID
(r"is 0.830 (95\% confidence interval 0.824 to 0.835), and the Fr\'echet inception distance (FID)\cite{heusel2017} between the two sets is 271; with 400 images per set this FID is a coarse indicator only. By the same measures embedding changes a fine-tuned cover little (CLIP similarity 0.9993, minimum 0.9948; FID 4.9).",
 r"is 0.830 (95\% confidence interval 0.827 to 0.833), and the Fr\'echet inception distance (FID)\cite{heusel2017} between the two sets is 238; with 1,000 images per set this FID is a coarse indicator only. By the same measures embedding changes a fine-tuned cover little (CLIP similarity 0.9993, minimum 0.9921; FID 3.6)."),
(r"the log-log spectral slope is $-3.60$ for baseline covers, $-1.86$ for fine-tuned covers", r"the log-log spectral slope is $-3.57$ for baseline covers, $-1.85$ for fine-tuned covers"),
(r"Five of the 400 pairs, selected by the authors", r"Five of the 1,000 pairs, selected by the authors"),
# capacity
(r"a mean raw slot capacity of 25,279 bits (0.386\bpp; range 0.172 to 0.546), and 363 of 400 exceed 0.30\bpp{}", r"a mean raw slot capacity of 25,471 bits (0.389\bpp; range 0.156 to 0.577), and 924 of 1,000 exceed 0.30\bpp{}"),
(r"The corresponding figure for baseline covers, 12,680 bits (0.193\bpp), overstates what those covers support: 167 baseline covers", r"The corresponding figure for baseline covers, 13,163 bits (0.201\bpp), overstates what those covers support: 434 baseline covers"),
(r"For the 233 baseline covers whose $K$ is derived from the cover, mean capacity is 0.053\bpp{}", r"For the 566 baseline covers whose $K$ is derived from the cover, mean capacity is 0.056\bpp{}"),
(r"(mean 0.447; 19,688 to 39,304 bits; $K$ from 19 to 38). For 273 of the 400 fine-tuned covers", r"(mean 0.447; 19,680 to 39,304 bits; $K$ from 19 to 38). For 688 of the 1,000 fine-tuned covers"),
# recovery
(r"With messages of half each fine-tuned cover's adaptive capacity (5,616 to 17,880 bits), all 400 messages were recovered exactly. To verify recovery over the 0.30 to 0.60\bpp{} sweep with known ground truth, every cover was re-embedded with a seed-reproducible message at its assigned payload:",
 r"With messages of half each fine-tuned cover's adaptive capacity (5,104 to 18,904 bits), all 1,000 messages were recovered exactly. To verify recovery over the 0.30 to 0.60\bpp{} sweep with known ground truth, every cover was embedded with a seed-reproducible message at its assigned payload:"),
# distortion
(r"Table~\ref{tab:quality} bins the 400 pairs by payload", r"Table~\ref{tab:quality} bins the 1,000 pairs by payload"),
(r"which for 273 fine-tuned covers exceeds the capacity", r"which for 688 fine-tuned covers exceeds the capacity"),
(r"from $+0.051$ at 0.30 to 0.35\bpp{} to $+0.112$ at 0.55 to 0.60\bpp{} (all $p<10^{-9}$; fine-tuned SSIM was higher in every one of the 400 pairs). Over the whole corpus, SSIM was $0.990\pm0.009$ (range 0.934 to 0.999) for fine-tuned covers against $0.912\pm0.054$ (0.698 to 0.992) for baseline covers. PSNR behaves differently: the fine-tuned gain decreases from $+0.58$\,dB at 0.30 to 0.35\bpp{} to a non-significant $+0.04$\,dB at 0.45 to 0.50\bpp{} ($p=0.443$), and reverses above 0.50\bpp, where fine-tuned covers are 0.28 to 0.41\,dB worse ($p<10^{-5}$). Averaged over all pairs the PSNR difference is $+0.08$\,dB, and luminance PSNR lies between 30.0 and 36.8\,dB for fine-tuned covers.",
 r"from $+0.053$ at 0.30 to 0.35\bpp{} to $+0.112$ at 0.55 to 0.60\bpp{} (all $p<10^{-25}$; fine-tuned SSIM was higher in every one of the 1,000 pairs). Over the whole corpus, SSIM was $0.990\pm0.009$ (range 0.934 to 0.999) for fine-tuned covers against $0.909\pm0.055$ (0.650 to 0.993) for baseline covers. PSNR behaves differently: the fine-tuned gain decreases from $+0.53$\,dB at 0.30 to 0.35\bpp{} to $+0.07$\,dB at 0.45 to 0.50\bpp{} ($p=0.046$), and reverses above 0.50\bpp, where fine-tuned covers are 0.21 to 0.39\,dB worse ($p<10^{-7}$). Averaged over all pairs the PSNR difference is $+0.08$\,dB, and luminance PSNR lies between 29.9 and 36.8\,dB for fine-tuned covers."),
(r"The re-embedding run, which used different messages, reproduced this pattern:", r"For the first 400 pairs, a second embedding run with different messages reproduced this pattern:"),
(r"(\textbf{a}) and SSIM (\textbf{b}) of each stego image against its cover as a function of embedded payload, for baseline and fine-tuned covers embedded with the same message and the same $K$ ($n=400$ pairs;",
 r"(\textbf{a}) and SSIM (\textbf{b}) of each stego image against its cover as a function of embedded payload, for baseline and fine-tuned covers embedded with the same message and the same $K$ ($n=1{,}000$ pairs;"),
(r"binned by assigned payload ($n=400$ pairs).", r"binned by assigned payload ($n=1{,}000$ pairs)."),
# processing tests
(r"Every standard distortion reduced exact message recovery to 0 of 400 images,", r"Every standard distortion reduced exact message recovery to 0 of 1,000 images,"),
(r"The header survived in 3 of 400 images under Gaussian noise", r"The header survived in 4 of 1,000 images under Gaussian noise"),
(r"a $1^\circ$ rotation and a two-pixel crop also gave 0 of 400 exact recoveries", r"a $1^\circ$ rotation and a two-pixel crop also gave 0 of 1,000 exact recoveries"),
(r"only 10 of 400 messages were recovered exactly in the unscaled case", r"only 20 of 1,000 messages were recovered exactly in the unscaled case"),
(r"(Methods; Supplementary Table~S6; 100 fine-tuned covers).", r"(Methods; Supplementary Table~S6; 100 fine-tuned covers from the first 400 pairs)."),
(r"($n=400$ fine-tuned covers; payload equal to half of each cover's adaptive capacity, 5,616 to 17,880 bits).", r"($n=1{,}000$ fine-tuned covers; payload equal to half of each cover's adaptive capacity, 5,104 to 18,904 bits)."),
(r"Header intact and exact recovery are counts out of 400.", r"Header intact and exact recovery are counts out of 1,000."),
(r"None (PNG round trip) & 100.0 & 400 of 400 & 400 of 400 \\", r"None (PNG round trip) & 100.0 & 1,000 of 1,000 & 1,000 of 1,000 \\"),
(r"Resize to 0.90 / 0.75 / 0.50 and back & 50.2 / 49.9 / 50.1 & 0 & 0 \\", r"Resize to 0.90 / 0.75 / 0.50 and back & 50.2 / 49.9 / 50.2 & 0 & 0 \\"),
(r"Gaussian noise, $\sigma=5$ / 10 / 15 & 79.1 / 56.7 / 50.8 & 3 / 0 / 0 & 0 \\", r"Gaussian noise, $\sigma=5$ / 10 / 15 & 79.1 / 56.7 / 50.8 & 4 / 0 / 0 & 0 \\"),
(r"& 99.0 / 96.2 / 99.9 & 208 / 43 / 392 & 0 / 0 / 10 \\", r"& 99.0 / 96.2 / 99.9 & 519 / 104 / 988 & 0 / 0 / 20 \\"),
# computational cost: timing kept from the uninterrupted run of the first 400 pairs
(r"on fine-tuned covers the median embedding time was \EmbLowFT\,s", r"in the embedding run of the first 400 pairs the median time on fine-tuned covers was \EmbLowFT\,s"),
# discussion
(r"raised the mean eligible count from 1.5 to 16.5 coefficients per block on every one of 400 matched pairs", r"raised the mean eligible count from 1.6 to 16.6 coefficients per block on every one of 1,000 matched pairs"),
(r"(SSIM 0.990 against 0.912)", r"(SSIM 0.990 against 0.909)"),
(r"The PSNR gain is small below 0.45\bpp, absent at 0.45 to 0.50\bpp{} and slightly negative above 0.50\bpp.", r"The PSNR gain is small below 0.45\bpp, close to zero at 0.45 to 0.50\bpp{} and slightly negative above 0.50\bpp."),
(r"returned only 10 of 400 messages exactly.", r"returned only 20 of 1,000 messages exactly."),
(r"The corpus contains 400 pairs from one prompt bank of textures at one resolution,", r"The corpus contains 1,000 pairs from one prompt bank of 100 texture subjects at one resolution,"),
# methods
(r"Prompt $i$ of the 400-prompt bank was sampled with seed $42+i$ for both the baseline and the fine-tuned model,",
 r"The prompt bank has 1,000 prompts: the 400 of the original evaluation (40 texture subjects) and 600 added during revision from 60 further texture subjects with the same ten modifiers, generated with the same code and weights. Prompt $i$ was sampled with seed $42+i$ for both the baseline and the fine-tuned model,"),
(r"For the re-embedding run that verifies recovery, the message for image $i$ was drawn from a pseudorandom generator seeded with $424242+i$, the stego image was written to PNG, reloaded and extracted, and the output compared byte-for-byte with the input.",
 r"For the first 400 pairs, recovery was verified in a separate re-embedding run in which the message for image $i$ was drawn from a pseudorandom generator seeded with $424242+i$; for the 600 added pairs the distortion run itself used seed-reproducible payloads and messages. In both cases the stego image was written to PNG, reloaded and extracted, and the output compared byte-for-byte with the input."),
(r"The test used 100 fine-tuned covers (every fourth image of the corpus)", r"The test used 100 fine-tuned covers (every fourth image of the first 400 pairs)"),
(r"and the three conditions were run on the 400 fine-tuned covers with the same protocol", r"and the three conditions were run on the 1,000 fine-tuned covers with the same protocol"),
]
for old, new in R:
    assert s.count(old) == 1, ("main", old[:90], s.count(old))
    s = s.replace(old, new)
# Table 2 rows
a = s.index("0.30 to 0.35 & 70 & 35.19"); b = s.index(r"\bottomrule", a)
s = s[:a] + t2 + "\n" + s[b:]
open(SUB + "main.tex", "w", encoding="utf-8", newline="").write(s)

t = open(SUB + "supplementary.tex", encoding="utf-8").read()
RS = [(r"\caption{\textbf{Rule-selected examples from the 400-pair evaluation corpus.}", r"\caption{\textbf{Rule-selected examples from the 1,000-pair evaluation corpus.}"),
      (r"on the 400 stego images of the matched-payload sweep", r"on the 1,000 stego images of the matched-payload sweep")]
for old, new in RS:
    assert t.count(old) == 1, ("supp", old[:80]); t = t.replace(old, new)
n = t.count("of 400 \\\\") + t.count("of 400 at each budget")
t = t.replace("400 of 400 \\\\", "1,000 of 1,000 \\\\").replace("0 of 400 \\\\", "0 of 1,000 \\\\").replace("0 of 400 at each budget", "0 of 1,000 at each budget")
open(SUB + "supplementary.tex", "w", encoding="utf-8", newline="").write(t)
print("main.tex: %d replacements and Table 2; supplementary: %d captions and %d Table S1 cells" % (len(R), len(RS), n))
