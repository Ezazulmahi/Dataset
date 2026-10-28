# Derivations for embedder v4 and v5 (our own)

Notation. A payload coefficient $c$ at a position with quantisation step $Q$ has
quantised value $q=\mathrm{round}(c/Q)$ and carries the bit $q \bmod 2$. Write
$x = c - qQ \in [-Q/2, Q/2]$ for its offset inside its bin. Under the usual
high-resolution assumption $x$ is uniform on $[-Q/2,Q/2]$, and the message bit is
independent of $x$, so half of the slots already hold the right parity.

## 1. Expected distortion of the v3 rule

v3 writes every slot at a bin centre $q'Q$.

* Parity already right (probability 1/2): the coefficient moves by $|x|$,
  $E[e^2] = \frac{1}{Q}\int_{-Q/2}^{Q/2} x^2\,dx = Q^2/12$.
* Parity wrong: it moves to the centre of the nearer neighbouring bin, a distance
  $Q-|x|$ with $|x|\sim U(0,Q/2)$:
  $E[e^2] = \frac{2}{Q}\int_0^{Q/2}(Q-t)^2dt = 7Q^2/12$.

$$D_{v3} = \tfrac12\left(\tfrac{Q^2}{12}+\tfrac{7Q^2}{12}\right) = \frac{Q^2}{3}.$$

## 2. Expected distortion of the v4 margin rule

v4 moves a coefficient only into the admissible set of the target bin,
$[(q'-\tfrac12+\mu)Q,\;(q'+\tfrac12-\mu)Q]$, $0<\mu\le\tfrac12$.

* Parity right: no move unless $|x| > (\tfrac12-\mu)Q$ (probability $2\mu$), and
  then by $t\sim U(0,\mu Q)$: $E[e^2] = 2\mu\cdot\mu^2Q^2/3 = \tfrac23\mu^3Q^2$.
* Parity wrong: the coefficient crosses the nearer bin edge, at distance
  $s = Q/2-|x| \sim U(0,Q/2)$, and stops $\mu Q$ inside:
  $E[e^2] = \frac{2}{Q}\int_0^{Q/2}(s+\mu Q)^2ds = \tfrac23\left[(\tfrac12+\mu)^3-\mu^3\right]Q^2$.

$$\boxed{D_{v4}(\mu) = \frac{(\tfrac12+\mu)^3}{3}\,Q^2}$$

Checks: $\mu=\tfrac12$ gives $Q^2/3=D_{v3}$; the gain over v3 is
$10\log_{10}\!\big(1/(\tfrac12+\mu)^3\big)$ dB, i.e. 3.7 dB at $\mu=0.25$,
5.6 dB at $\mu=0.15$ and 9.0 dB as $\mu\to0$. Measured gains are smaller
(about 1.7–1.9 dB at $\mu=0.25$ in the pilot) because the uniform-offset
assumption fails for the many slots with $q=0$, because some slots are pushed to
larger margins by the verify-fix escalation (Section 4), and because pixel
clipping and the colour repair add error that the margin does not control.

## 3. What the margin buys: a guaranteed tolerance

Every written slot satisfies $|x| \le (\tfrac12-\mu)Q$, so its parity survives
**any** later perturbation of the coefficient with $|e| < \mu Q$.

* Pixel rounding after the inverse DCT adds an error of variance $1/12$ per pixel;
  the orthonormal DCT keeps that variance per coefficient, s.d. $0.29$. With
  $Q\ge 8$ on $\mathcal M$, $\mu=0.15$ already gives $\mu Q\ge1.2\approx4$ s.d.
* Requantisation with the embedder's own table scaled by $s$ (step
  $Q_s\approx sQ$) moves a coefficient by at most $Q_s/2$, so all written
  bits survive when $\mu > s/2$. Prediction: $\mu=0.25$ protects scale 0.2 but not
  scale 0.6; $\mu=0.5$ (v3) protects both; at scale 1.0 the requantisation maps
  every coefficient to its own bin centre and never changes parity, for any
  $\mu$. This is the PSNR/robustness trade-off in one inequality. The guarantee
  ignores pixel clipping: after requantisation, reconstructed values outside
  0..255 are clipped, which perturbs coefficients beyond the bound. The covers
  here have many saturated pixels (31 % in fine-tuned cover 000), which is why the
  pilot still loses about 4 % of bits at scale 0.2 with $\mu=0.25$.

## 4. Convergence under clipping: per-slot margin escalation

When many pixels of a block sit at 0 or 255, the pixel clipping after the inverse
DCT removes part of a small coefficient move, and a slot written with a small
margin can read wrong again. v4 keeps a failure count $f_k$ per slot and rewrites a
failing slot with margin
$$\mu_k = \min\!\left(\tfrac12,\;\mu\,2^{\,f_k-1}\right),$$
so after at most $\lceil\log_2(1/(2\mu))\rceil+1$ failures the slot is written at
its bin centre, the v3 rule, which converged on all 2,000 covers of the 1,000-pair
corpus. Slots that never fail keep the small margin, so the extra distortion is
confined to the few stubborn slots. (Offsetting the write by the clipping loss
measured on the previous pass was also tried and oscillated on saturated covers;
see results/compensation_test.txt.)

## 5. v5 secure mode: the slot cost

With syndrome coding the embedder chooses which slots change. The cost of
changing slot $k$ in block $b$ is
$$\rho_k = d_k^2\;\frac{1+\gamma\,s_b}{\sigma_b^2+\tau}.$$

* $d_k$ is the move the v4 rule needs to flip slot $k$ (Section 2, with margin
  $\mu$). By Parseval for the orthonormal DCT, $d_k^2$ is also the squared
  pixel-domain change of block $b$ before rounding, so one quantity measures the
  change in both domains.
* $\sigma_b^2$ is the mean squared Laplacian residual of the cover luminance in
  block $b$. For a change of energy $E$ in a block whose content behaves like
  noise of variance $\sigma^2$, the deflection of a likelihood-ratio detector grows
  as $E/\sigma^2$, so the same change is cheaper where the block is busy; $\tau$
  keeps flat blocks finite.
* $s_b$ is the fraction of saturated pixels in block $b$; clipping partly undoes
  changes there and forces extra repair passes, so they are penalised by
  $\gamma s_b$.

## 6. v5 robust mode: soft reliability of a read

At extraction let $f = c/Q - \mathrm{round}(c/Q)\in[-\tfrac12,\tfrac12]$ and
$a = 1-2|f|\in[0,1]$. A bit written at a bin centre ($f=0$) is read wrong only if
the channel moved it by more than $Q/2$, i.e. past the point where $a$ reaches 0.
For channel errors that are symmetric and decreasing in magnitude, the
probability that a read is right therefore increases monotonically with $a$, and
$a$ serves as a reliability weight: each of the $r$ copies of frame bit $i$ votes
$(2b-1)\,a$ and the bit is the sign of the sum. With all $a=1$ this is the plain
majority vote used in Supplementary Table S6.
