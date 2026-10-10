r"""Update the detection text, Methods and Data availability of main.tex to the 1,000-pair results
(results_1000\universal_dct_steganalysis_report.txt, analysis1000\steganalysis_*.csv, report_1000.txt)."""
SUB = "F:/Publication/scirep_submission/"
s = open(SUB + "main.tex", encoding="utf-8").read()
R = [
(r"flagged, at its trained threshold, no stego image in fine-tuned covers but 83\% in baseline covers,", r"flagged, at its trained threshold, no stego image in fine-tuned covers but 84\% in baseline covers,"),
(r"At that threshold it flagged none of the 400 DEDS stego images of fine-tuned covers and no cover, which is 50\% accuracy ($P_\mathrm{E}=0.500$); for the same embedder on baseline covers it flagged 332 of 400 stego images and no cover. At the conventional threshold of 0.5 it flagged 2 of 400 fine-tuned and 391 of 400 baseline stego images, again with no cover flagged.",
 r"At that threshold it flagged none of the 1,000 DEDS stego images of fine-tuned covers and no cover, which is 50\% accuracy ($P_\mathrm{E}=0.500$); for the same embedder on baseline covers it flagged 835 of 1,000 stego images and no cover. At the conventional threshold of 0.5 it flagged 2 of 1,000 fine-tuned and 981 of 1,000 baseline stego images, again with no cover flagged."),
(r"(AUC) of 0.859 (95\% confidence interval 0.832 to 0.886), and the stego score exceeds the paired cover score in 88\% of pairs,", r"(AUC) of 0.851 (95\% confidence interval 0.833 to 0.868), and the stego score exceeds the paired cover score in 87\% of pairs,"),
(r"By payload bin the AUC is 0.84 to 0.99 below 0.55\bpp{} and 0.43 at 0.55 to 0.60\bpp. A threshold chosen on these test scores would give $P_\mathrm{E}=0.17$,",
 r"By payload bin the AUC is 0.95 to 0.98 below 0.50\bpp, 0.81 at 0.50 to 0.55\bpp{} and 0.41 at 0.55 to 0.60\bpp. A threshold chosen on these test scores would give $P_\mathrm{E}=0.18$,"),
(r"does not transfer ($P_\mathrm{E}=0.27$).", r"does not transfer ($P_\mathrm{E}=0.25$)."),
(r"decreased after embedding in 384 of 400 pairs;", r"decreased after embedding in 967 of 1,000 pairs;"),
(r"\caption{\textbf{Detectability on the 400-pair corpus at payloads of 0.30 to 0.60\bpp.}", r"\caption{\textbf{Detectability on the 1,000-pair corpus at payloads of 0.30 to 0.60\bpp.}"),
(r"the same embedder was flagged in 83\% of baseline stego images and in none of the fine-tuned ones.", r"the same embedder was flagged in 84\% of baseline stego images and in none of the fine-tuned ones."),
(r"The scores of that detector still rank stego above cover (AUC 0.86),", r"The scores of that detector still rank stego above cover (AUC 0.85),"),
(r"one feature set and 400 evaluation pairs,", r"one feature set and 1,000 evaluation pairs,"),
(r"It was then applied without further training to the 400 fine-tuned covers and their DEDS stego images, and to the 400 baseline covers and the DEDS stego images of the re-embedding run. Scoring the fine-tuned stego images of the re-embedding run gave the same result (no image flagged; AUC 0.874).",
 r"It was then applied without further training to the 1,000 fine-tuned covers and their DEDS stego images, and to the 1,000 baseline covers and their DEDS stego images (for the first 400 pairs, those of the re-embedding run). The 60 subjects added during revision were not among its training prompts. For the first 400 pairs, scoring the fine-tuned stego images of the re-embedding run gave the same result (no image flagged; AUC 0.874)."),
(r"The 400 pairs were split into five folds by prompt subject (40 subjects of 10 images),", r"The 1,000 pairs were split into five folds by prompt subject (100 subjects of 10 images),"),
(r"that embedding was applied to the 400 fine-tuned covers of the corpus.", r"that embedding was applied to the 1,000 fine-tuned covers of the corpus."),
(r"The 400-pair evaluation corpus, the stego images,", r"The 1,000-pair evaluation corpus, the stego images,"),
(r"and the folder \texttt{repeated\_message\_test} contains the code and results of the repeated-message test.",
 r"the folder \texttt{repeated\_message\_test} contains the code and results of the repeated-message test, and the folder \texttt{expansion\_1000} contains the 600 pairs added during revision with their prompts and the generation, embedding and analysis scripts and per-image result files of the full corpus."),
]
for old, new in R:
    assert s.count(old) == 1, (old[:90], s.count(old))
    s = s.replace(old, new)
open(SUB + "main.tex", "w", encoding="utf-8", newline="").write(s)
print(len(R), "replacements")
