# JISA reviewer-response matrix (checked against main.tex of 11 October 2026)

## R1.1  [PARTIALLY FULFILLED]
**Comment.** Novelty limited: the framework combines established techniques (SD v1.5, LoRA, DCT embedding, LSB, Huffman, CRC, adaptive selection); the DCT reward is not shown to be a substantial advance over frequency-aware generative steganography; a clearer technical distinction is needed.
**Concern.** Novelty and positioning
**Revision.** Introduction states the embedder is assembled from standard components and serves as a test bed; contributions are restated as a reward-guided fine-tuning objective plus a controlled evaluation; diffusion is used only as the cover source, unlike methods that embed in the diffusion process (CRoSS, GSD/StegoDiffusion, PSyDUCK, Stable Diffusion latent methods); reward fine-tuning precedent (DDPO, DRaFT) is cited.
**Where.** Introduction paragraphs 3 to 6
**Evidence.** Text in main.tex; no experiment against frequency-aware generative methods
**Remaining.** The distinction is argued in text only. No same-data experiment against a frequency-aware generative scheme (they solve a different problem: lossy-channel recovery).
**Response.** We agree that the components are established. We now state this explicitly and restrict the contribution to a reward-guided fine-tuning objective for cover generation and a controlled measurement of what it changes; we explain why diffusion-as-channel methods are not directly comparable.

## R1.2  [FULFILLED - VERIFIED]
**Comment.** Steganalysis shows a security weakness: near-random accuracy at the native SRNet threshold is threshold miscalibration; calibrated accuracy 71.8 to 82.7%, AUC up to 0.836; security claims unsupported.
**Concern.** Unsupported security claims; threshold miscalibration
**Revision.** All security claims withdrawn (Introduction scope paragraph, Discussion). The old detector is replaced by (a) an SRNet-style CNN trained from scratch on a universal DCT embedding with its threshold fixed on its own validation set, and (b) SPAM detectors trained on DEDS with subject-grouped CV. Threshold-free AUC and P_E reported; matched detectors reach AUC 1.000; text states the trained-threshold result does not show indistinguishability.
**Where.** Results, Detectability; Table 3; Fig. 4c; Discussion
**Evidence.** claim_ledger.csv C22 to C31; universal_dct_steganalysis_results.xlsx; steganalysis_spam.csv
**Remaining.** None.
**Response.** We agree. All security claims are removed. We report threshold-free AUC and P_E, and detectors trained on our embedder detect it almost perfectly (AUC 1.000).

## R1.3  [PARTIALLY FULFILLED]
**Comment.** Comparison not rigorous or fair: results from different studies compared directly; J-UNIWARD, S-UNIWARD, WOW and J-MiPOD not evaluated on the same cover/stego data with the same steganalyser.
**Concern.** Fair same-pipeline baselines
**Revision.** Cross-paper numeric comparison removed (Discussion states no ranking is implied). S-UNIWARD, WOW and LSB matching run on the same 1,000 pairs, same luminance planes, same number of payload bits per image, same SPAM detector protocol (entropy-bound simulation). J-UNIWARD and J-MiPOD not run; justified because the stego object is a losslessly stored spatial image.
**Where.** Results, Detectability; Table 3; Methods, Reference embedders; Discussion, limitations
**Evidence.** steganalysis_spam.csv; analysis1000/steganalysis_wow.csv; wow.py verified against brute force (relative error 1e-13)
**Remaining.** Spatial baselines: done (simulated at the entropy bound, not STC-coded; baselines were not scored by the CNN). JPEG-domain baselines: not run. A reviewer may argue DEDS also requantises luminance with a JPEG-like table; a contextual J-UNIWARD run on JPEG versions of the covers at matched bits would close this.
**Response.** S-UNIWARD, WOW and LSB matching were run on the same covers, payloads and detector; both adaptive schemes are markedly less detectable than DEDS. J-UNIWARD and J-MiPOD embed in JPEG files, whereas our stego object is a lossless spatial image, so no like-for-like comparison exists; we state this as a limitation.

## R1.4  [TEXTUALLY ADDRESSED ONLY]
**Comment.** Extremely poor robustness: 0% full recovery under JPEG 90, resizing, noise, blur, cropping, geometric changes; Hamming(7,4) fails; method restricted to a clean lossless channel.
**Concern.** Practical applicability
**Revision.** No robustness claim; scope restricted to lossless storage (Abstract, Introduction, Discussion). Table 4 on 1,000 covers; Supplementary Table S1 includes the Hamming(7,4) simulation; new repeated-message test (Supplementary Table S6) shows recovery only after own-table requantisation or at about 27 bytes.
**Where.** Abstract; Results, Message recovery after lossy processing; Table 4; Supp. Tables S1, S6; Discussion
**Evidence.** robustness_results.xlsx; customq_requantisation_1000.csv; robust_mode_results_r3_9_101.xlsx
**Remaining.** The weakness is inherent to the embedder and is not fixed; it is measured and stated. Additional measurements were added.
**Response.** We agree. The paper now claims exact recovery on a lossless channel only, reports the failures in full, and adds a repeated-message test that shows the limits of redundancy.

## R1.5  [PARTIALLY FULFILLED]
**Comment.** Figures look stylised/generic, have small unclear text, and do not expose operations, data flow or module interactions; redesign as technically informative, editable diagrams.
**Concern.** Figure quality
**Revision.** Fig. 1 redrawn as a vector schematic with equations, tensor shapes and gradient path (script fig_schematic.py, editable); Figs. 2 to 4 are data plots generated from the result files.
**Where.** Figs. 1 to 4; analysis/fig_schematic.py, make_figs.py, final_numbers.py
**Evidence.** Vector PDFs in figures/
**Remaining.** Legibility of the smallest labels (about 6 to 7 pt) must be checked in the compiled PDF by the authors.
**Response.** All figures were redrawn from scripts as vector graphics; Fig. 1 now shows each operation with its equation number.

## R1.6  [FULFILLED - VERIFIED]
**Comment.** Practical benefit of the costly diffusion stage not established; no comprehensive computational complexity/runtime comparison; trade-off not compelling given detectability and fragility.
**Concern.** Cost-benefit
**Revision.** Computational cost section: generation 1.9/3.0 s, DEDS embedding 15 to 85 s, extraction 0.12 s, fine-tuning 18 to 24 h (authors' records), and measured S-UNIWARD/WOW cost computation 0.12 s per image on the same processor; Discussion states DEDS is far more expensive and that its benefit is limited to capacity and to one detector.
**Where.** Results, Computational cost; Supp. Table S4; Discussion
**Evidence.** embed_rerun.csv; timing measured 11 Oct 2026 (20 covers, single process)
**Remaining.** Fine-tuning time comes from the authors' records, not a log.
**Response.** We now report measured runtimes for every stage and for S-UNIWARD and WOW on the same processor, and we state plainly that DEDS is far more expensive.

## R4.1  [PARTIALLY FULFILLED]
**Comment.** Novelty real but incremental; core embedding (LSB of quantised DCT coefficients) well established; SOTA techniques should be compared and analysed.
**Concern.** Novelty; SOTA comparison
**Revision.** See R1.1 and R1.3.
**Where.** Introduction; Table 3
**Evidence.** As R1.1, R1.3
**Remaining.** As R1.3 (JPEG-domain schemes; deep SOTA detectors).
**Response.** See responses to Reviewer 1, comments 1 and 3.

## R4.2  [FULFILLED - VERIFIED]
**Comment.** Scientific contribution small: that DCT capacity depends on mid-frequency energy is well established (watermarking; payload distribution of J-UNIWARD/J-MiPOD).
**Concern.** Established fact presented as finding
**Revision.** Introduction states the dependence is well established and not a finding of this work; the research question is restated as whether a generator can be trained to supply such coefficients and what this does to capacity, distortion, cover statistics and detectability. The set M is identified as the upper-frequency half, not a mid-band.
**Where.** Introduction paragraph 4; Results, first subsection; Methods, DCT reward loss
**Evidence.** main.tex text
**Remaining.** None.
**Response.** We agree and say so explicitly; the paper no longer presents this relationship as a result.

## R4.3  [NOT APPLICABLE - JUSTIFIED]
**Comment.** Composite loss (diffusion + DCT reward + anchor) is technically sound.
**Concern.** Positive comment
**Revision.** No change required.
**Where.** Methods, Training objective
**Evidence.** n/a
**Remaining.** None.
**Response.** We thank the reviewer.

## R4.4  [PARTIALLY FULFILLED]
**Comment.** 0.3 to 0.5 bpp is a regular rate for J-UNIWARD/J-MiPOD with much better security; this comparison is missing; Huffman compression is standard practice.
**Concern.** Capacity framing; missing comparison
**Revision.** Title and abstract say 'increased' capacity, not 'high'; payload range 0.30 to 0.60 bpp compared with S-UNIWARD and WOW at identical bits on identical covers, both less detectable. No compression is used in any experiment; Huffman is mentioned only as an unused option.
**Where.** Title; Abstract; Embedding capacity; Table 3; Methods, Payload
**Evidence.** comparison_results.xlsx; steganalysis_spam.csv
**Remaining.** Spatial comparison done; J-UNIWARD/J-MiPOD not run (see R1.3).
**Response.** See R1.3; compression is no longer used and bpp counts message bits only.

## R4.5  [TEXTUALLY ADDRESSED ONLY]
**Comment.** Comparison with recent closely related methods (BAFLE-DCT, PSyDUCK, ImSA, StegoDiffusion) missing.
**Concern.** Related-work coverage
**Revision.** BAFLE-DCT, PSyDUCK and GSD (StegoDiffusion) are cited and positioned in the Introduction; no experiments with them. ImSA could not be identified in any accessible scholarly source (the editor removed several reviewer-suggested references) and is not cited.
**Where.** Introduction paragraph 3
**Evidence.** Web search 11 Oct 2026: StegoDiffusion = GSD (arXiv 2305.03472); ImSA not found
**Remaining.** No experiments (different problem settings and code availability). ImSA: BLOCKED pending a reference from the authors.
**Response.** BAFLE-DCT, PSyDUCK and StegoDiffusion are now discussed; they hide data in the diffusion process or select coefficients in existing images, which is a different problem. We could not identify 'ImSA' and would be grateful for the reference.

## R4.6  [NOT APPLICABLE - JUSTIFIED]
**Comment.** Transparency about robustness failure and detectability is excellent but the limitation is too significant for a security journal.
**Concern.** Fit to venue
**Revision.** Submitted to Scientific Reports with the scope restated as capacity and exact lossless recovery; limitations reported in full.
**Where.** Introduction scope paragraph; Discussion
**Evidence.** n/a
**Remaining.** None.
**Response.** We have narrowed the claims and moved to a general journal; the limitations are reported in full.

## R4.7  [FULFILLED - VERIFIED]
**Comment.** Reproducibility efforts acknowledged (code, corpus, LoRA weights, prompts released).
**Concern.** Positive comment
**Revision.** Release expanded: 1,000-pair corpus, all scripts, merged result files, claim ledger.
**Where.** Data availability
**Evidence.** GitHub Ezazulmahi/Dataset (commits 71037f5, 68b1abc)
**Remaining.** None.
**Response.** We thank the reviewer; the release now covers every number in the paper.

## R4.8  [PARTIALLY FULFILLED]
**Comment.** n = 400 is small compared with ALASKA for security evaluation.
**Concern.** Sample size
**Revision.** Corpus expanded to 1,000 pairs (60 new texture subjects, authors' generation code, pixel-exact reproduction check) plus a 200-pair check on 20 non-texture subjects; still far smaller than ALASKA, stated as a limitation.
**Where.** Evaluation corpus; Methods, Cover generation; Supp. Table S7; Discussion
**Evidence.** report_1000.txt; claim_ledger.csv
**Remaining.** 1,000 pairs remains small for steganalysis benchmarks.
**Response.** We expanded the corpus to 1,000 pairs and added a 200-pair check on unseen subjects; we state that this is still small.

## R4.9  [PARTIALLY FULFILLED]
**Comment.** Use a well-established detector (maxSRM with fast linear classifier), less sensitive to cover-source mismatch; CSM literature not reported.
**Concern.** Detector choice; cover-source mismatch
**Revision.** CSM literature cited (Giboulot et al. 2020) to explain why detectors trained on photographs fail. SPAM (686) with shrinkage LDA used instead of maxSRM; Discussion states rich-model features would give tighter estimates.
**Where.** Results, Detectability (last paragraph); Discussion, limitations
**Evidence.** n/a
**Remaining.** maxSRM not run. Minimum work: SRM/maxSRM feature extraction (e.g. a Python port such as the one in the Aletheia toolbox) on 2,000 to 4,000 images and an FLD ensemble with the same subject-grouped CV; roughly 3 to 5 hours.
**Response.** We cite the CSM literature and use detectors trained on the same source. We did not run maxSRM; a simple SPAM detector already detects DEDS with AUC 1.000, so a stronger detector cannot change that conclusion.

## R4.10  [FULFILLED - VERIFIED]
**Comment.** PSNR improvement is small (about 0.5 dB) and near zero at high payload; serious given the 'high payload' framing.
**Concern.** Overstated distortion benefit
**Revision.** Table 2 reports PSNR per bin with p values and effect sizes (+0.53 to -0.39 dB); text says the PSNR gain is small, close to zero at 0.45 to 0.50 bpp and negative above 0.50; SSIM gain reported separately; 'high payload' framing removed.
**Where.** Results, Embedding distortion; Table 2; Discussion
**Evidence.** claim_ledger.csv C14 to C18; table2_rows.tex
**Remaining.** None.
**Response.** We agree; the PSNR result is reported as small and reversing above 0.50 bpp.

## R4.11  [FULFILLED - VERIFIED]
**Comment.** 51.4% accuracy at the native 0.5 threshold mostly reflects miscalibration; abstract/introduction could be read as undetectability; report calibrated accuracy.
**Concern.** Misleading detection framing
**Revision.** As R1.2; the paper reports AUC, the optimistic best-threshold P_E (0.18) and explicitly says the trained-threshold result is not indistinguishability.
**Where.** Results, Detectability; Abstract
**Evidence.** claim_ledger.csv C25 to C27
**Remaining.** None.
**Response.** See R1.2.

## R4.12  [FULFILLED - VERIFIED]
**Comment.** Claim that DEDS has the 'same general detectability range as classical baselines' is speculative because J-MiPOD/J-UNIWARD were not run; also BAFLE-DCT, PSyDUCK, ImSA, StegoDiffusion comparisons missing.
**Concern.** Speculative comparative claim
**Revision.** The claim is removed; same-pipeline experiments show DEDS is more detectable than S-UNIWARD and WOW.
**Where.** Results, Detectability; Discussion
**Evidence.** Claim scan of main.tex (no such sentence)
**Remaining.** Related-method comparison: see R4.5.
**Response.** The claim is withdrawn; measured comparisons show the opposite.

## Editor  [NOT APPLICABLE - JUSTIFIED]
**Comment.** Six reviewer-suggested references (three self-citations) were removed by the editor as unjustified.
**Concern.** Citation ethics
**Revision.** None of the removed references were added; ImSA (possibly among them) is not cited.
**Where.** References
**Evidence.** n/a
**Remaining.** None.
**Response.** No action needed.

## R5.1  [FULFILLED - VERIFIED]
**Comment.** Claims of 'chance level detection' / 'near random guessing' against SRNet contradict Table 10 (accuracy 71.8 to 82.7%, AUC 0.709 to 0.836).
**Concern.** Unsupported undetectability claim
**Revision.** All such claims removed (claim scan: 'chance' appears only in definitions); see R1.2.
**Where.** Whole manuscript
**Evidence.** Claim scan 11 Oct 2026
**Remaining.** None.
**Response.** These claims have been removed.

## R5.2  [FULFILLED - VERIFIED]
**Comment.** BPP computed on the Huffman-compressed bitstream depends on the message distribution, so it is not comparable with fixed-payload work.
**Concern.** Payload definition
**Revision.** No compression in any experiment; bpp = message bits / pixels, header excluded; message entropy stated (6.07 bits per byte); random-bit messages give the same results (Supplementary Note 2).
**Where.** Embedding capacity; Methods, Payload; Supp. Note 2
**Evidence.** t11_report.txt
**Remaining.** None.
**Response.** Payload is now uncompressed message bits; we also show that random-bit messages give the same results.

## R5.3  [FULFILLED - VERIFIED]
**Comment.** A reward on fixed-band coefficient magnitudes can produce repetitive high-frequency texture, aliasing and unnatural images.
**Concern.** Cover naturalness
**Revision.** Confirmed and measured: spectral slopes, high-frequency power about thirty times that of photographs, CLIP 0.83, FID; Discussion states the covers are dense repetitive textures that would attract attention.
**Where.** Characteristics of the generated covers; Fig. 2d; Discussion
**Evidence.** make_figs.py output; claim_ledger.csv C19 to C21
**Remaining.** The underlying problem remains and is stated as the main open question.
**Response.** We agree and now measure it; it is reported as the main limitation.

## Status count

- PARTIALLY FULFILLED: 7
- FULFILLED - VERIFIED: 10
- TEXTUALLY ADDRESSED ONLY: 2
- NOT APPLICABLE - JUSTIFIED: 3