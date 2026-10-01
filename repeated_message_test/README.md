# Repeated-message test (DCT_Adaptive with an outer repetition code)

`DCT_Adaptive.py` is an unchanged copy of the embedder. `dct_adaptive_robust.py` imports it and
adds, at message level only, a frame (16-bit byte count, message, 16-bit check) written r times
in a row, majority-vote decoding, and a search over K when the 64-bit header is damaged.
`evaluate_robust.py` runs the end-to-end test; results are in `results/`.

Exact recovery out of 100 fine-tuned covers (every fourth image of the 400-pair corpus):

| Condition | r = 3 (1,052 bytes) | r = 9 (348 bytes) | r = 101 (27 bytes) |
|---|---|---|---|
| None (PNG round trip) | 100 | 100 | 100 |
| Own table, scale 1.0 | 100 | 100 | 100 |
| Own table, scale 0.6 | 0 | 80 | 100 |
| Own table, scale 0.2 | 0 | 100 | 100 |
| JPEG 95 | 0 | 9 | 100 |
| JPEG 90 | 0 | 0 | 100 |
| JPEG 70, JPEG 50 | 0 | 0 | 0 |
| Gaussian noise sigma 5 | 0 | 0 | 100 |
| Gaussian noise sigma 10 | 0 | 0 | 28 |
| Blur 3x3, resize 0.90, crop 5% | 0 | 0 | 0 |

The method is not robust to image processing. Exact recovery is obtained after requantisation
with the embedder's own table at a reduced payload, and after JPEG at quality 90 or above only
at about 27 bytes per image.

The scripts in this folder were written with the assistance of Claude (Anthropic).
