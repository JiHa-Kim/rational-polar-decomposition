# DWH2 Small-Side Diagnosis

This note tracks the bounded small-side DWH2 investigation.

## Goal

Make the DWH2 small-side path faster than the rectangular path without adding brittle routing or quality regressions on hard spectra.

## Current Best Integrated Kernel

The current `smallside_bounded` mode in [dwh2.py](../src/polar_decomposition/algorithms/dwh2.py):

- builds the small Gram with balanced split-K tree reduction
- forms `H0 = (I + c0 G0)^(-1)` from a Cholesky factor
- uses the bounded recurrence
  `K = H0 + (c1 / c0) M0 (I - H0) M0`
- evaluates the second bounded `K`-build multiply as
  `alpha * buf + beta * (H @ buf)` instead of a raw `M @ buf`
- applies the second solve as two triangular solves

Integrated 11-case benchmark at `16384 x 4096`, `seed=0`,
`spectral_bound` normalization:

- `rectangular` median runtime: `389.50 ms`
- `smallside_bounded` median runtime: `388.55 ms`
- faster on `6/11`
- better `q_fro_error` on `8/11`
- remaining worse cases: `rank_1_heavy`, `lowrank_noise`, `ar1_cols`

## Confirmed Diagnoses

### 1. The old instability was structural, not a generic Cholesky issue.

What actually broke before the bounded rewrite was the explicit propagated Gram-like state. The bounded recurrence fixed the SPD-loss pathology without needing fallback branches in the good path.

Confirmed by the current integrated sweep:

- `smallside_bounded` had `0` shifted Cholesky calls on all 11 cases
- the rectangular baseline still uses shifted calls

### 2. The remaining hard-case loss is not caused by the small-side Gram build.

On the hard cases, the TF32 split-K Gram is already very close to the FP32 split-K Gram:

- `rank_1_heavy`: relative Gram error about `1.35e-5`
- `ar1_cols`: relative Gram error about `1.35e-5`
- `lowrank_noise`: relative Gram error about `1.89e-5`

Propagating the same Gram through a higher-accuracy bounded evaluator nearly removes the gap. The first-order issue is not the Gram itself.

### 3. The second-stage two-triangular-solve backend is not the real problem.

Using the same small Gram and turning TF32 off inside the bounded evaluator:

- `ref`: Cholesky solve on the second stage
- `tri_only`: two triangular solves on the second stage

These matched to measurement on the hard cases:

- `rank_1_heavy`: `q_ref ~= q_tri_only`
- `ar1_cols`: `q_ref ~= q_tri_only`
- `lowrank_noise`: `q_ref ~= q_tri_only`

So replacing the second `cholesky_solve` with two triangular solves is acceptable by itself.

### 4. The remaining loss comes from TF32 matmuls inside the bounded evaluator.

On the same small Gram:

- `ref`: FP32 small-side evaluator
- `tf32_only`: TF32 small-side evaluator with Cholesky solve still used
- `fast`: TF32 small-side evaluator plus the two-triangular second solve

`tf32_only` was already almost as bad as `fast` on the hard rows. That means the dominant remaining error is from TF32 small-side matmuls, not from the solve backend.

This is the most important current diagnosis.

More specifically, the hard-case gap is dominated by the bounded small-side
matmuls, not by the second solve backend:

- on the same small Gram, replacing the second `cholesky_solve` with two
  triangular solves alone was essentially neutral on the hard rows
- on the same small Gram, turning the bounded evaluator matmuls back to TF32
  recreated most of the loss immediately

For `ar1_cols`, the main bad actor is the `K`-build itself. The final
small-side affine multiply is comparatively minor there.

### 5. The first inverse and the second solve want different kernels.

The best branch-free split so far is:

- first full inverse `H0 = A0^{-1}`: `torch.cholesky_solve(I, L0)`
- second apply `K^{-1} H0`: two triangular solves

Why:

- using two triangular solves for the first full inverse degrades the hard rows
- using them only for the second apply preserves the speed win and keeps quality near the previous bounded kernel

## What Helped

- bounded recurrence instead of explicit congruence propagation
- balanced split-K tree reduction for the small Gram
- no unconditional Cholesky jitter in the good path
- accurate first inverse, faster second solve
- explicit symmetrization of the small-side SPD states
- affine decomposition of the second bounded `K`-build multiply:
  `M @ buf = alpha * buf + beta * (H @ buf)`

## What Did Not Help Enough

- more Cholesky jitter
- centering the second solve around `T = I - H`
  it was neutral on some cases and clearly worse on `ar1_cols`
- factorized 3-solve SPD evaluation of the bounded polynomial
  it improved `ar1_cols`, but runtime jumped too much
- blaming the problem on the second solve backend
  the isolated tests show that was not the main cause
- blaming the problem on the TF32 Gram build
  the measured Gram error is too small to explain the final gap

## Additional failed rewrites

These are worth recording because they looked principled on paper but were not
actually viable.

### Cubic-in-$T$ rewrite of the bounded step

Using

$$
T = I - H,\qquad
K = I + c_1 T + c_2 T^2 + c_3 T^3
$$

is algebraically equivalent to the bounded `K` update and keeps the same two
small-side matmuls. In TF32, however, it lost SPD on the hard rows and the
subsequent Cholesky failed.

Conclusion:

- mathematically elegant
- not numerically acceptable in the current low-precision regime

### $M^2 T$ or $T M^2$ rewrite

Since `M` and `T` commute in exact arithmetic, another natural idea was to form
`M^2` first and then multiply by `T`.

That also lost SPD on the hard rows in TF32 and failed the next Cholesky.

Conclusion:

- exact commutation is not enough
- forcing a “more symmetric” rewrite can still be worse numerically

### Symmetrizing the first `T M` intermediate

The intermediate product `T M` is symmetric in exact arithmetic, so explicitly
symmetrizing it looked like a cheap way to suppress TF32 commutation drift.

In practice it was catastrophic on the hard rows:

- `rank_1_heavy`, `ar1_cols`, and `lowrank_noise` all became much worse

Conclusion:

- the intermediate should not be projected back to symmetry that way
- the current bounded form is already closer to the right structure than that
  “fix”

## Current best structural fix

The best low-cost improvement so far is to rewrite the second bounded `K`-build
multiply

$$
M\,\mathrm{buf}
$$

as

$$
\alpha\,\mathrm{buf} + \beta(H\,\mathrm{buf}),
\qquad
M = \alpha I + \beta H.
$$

This matters because the large identity contribution is then handled exactly in
scalar arithmetic, and only the bounded `H @ buf` term goes through TF32.

Empirical effect on the integrated `smallside_bounded` kernel:

- keeps the median speed win against `rectangular`
- improves the remaining correlated-hard cases slightly without introducing
  routing or site-wide FP32

Latest 11-case sweep at `16384 x 4096`, `seed=0`, `spectral_bound`:

- `rectangular` median runtime: `389.50 ms`
- `smallside_bounded` median runtime: `388.55 ms`
- faster on `6/11`
- better `q_fro_error` on `8/11`

Representative rows:

- `ar1_cols`: `0.03585 -> 0.03508` versus the previous bounded variant
- `lowrank_noise`: `0.10528 -> 0.10510`
- `rank_1_heavy`: `0.01594 -> 0.01589`

This does not fully close the remaining `rank_1_heavy` gap to the rectangular
kernel, but it is currently the best speed/quality tradeoff found without
introducing brittle branching or broad FP32 fallback.

## Hard-Case Interpretation

The remaining difficult cases are not all the same:

- `rank_1_heavy`: the relevant correction lives in a very small subspace, so relative error in the bounded correction is amplified
- `ar1_cols`: the hard part is not tiny-rank; the TF32 bounded `K`-build itself is visibly damaging
- `lowrank_noise`: mixed behavior, but again the TF32 bounded evaluator is the dominant remaining source

So the next fix should not be another branch. It should target the TF32 bounded matmul pattern directly.

## Next Targeted Experiments

In order of priority:

1. Split the bounded evaluator by matmul site and identify which TF32 multiply is doing the damage.
   Most likely suspect: the `K`-build `M0 (I - H0)` / `M0 (...)` pair.

2. Rewrite the bounded correction so the fragile part is evaluated in a numerically centered form but without changing the algorithm or adding routing.

3. Only if needed, isolate one specific small-side multiply for higher precision.
   This is acceptable only if it is a single structural hotspot, not a pile of per-case branches.

## Potential Structural Improvements

These are the most plausible next ideas that still fit the current design goals.

### 1. Pull more identity mass out of TF32 matmuls

The latest affine rewrite helped because it kept the large identity term out of
the matmul path. The same principle may still apply to the remaining bounded
products.

What to look for:

- rewrites where TF32 only sees bounded operators like `H` or `I - H`
- affine decompositions that avoid multiplying by `alpha I + beta H` directly

Why it is promising:

- it directly targets the confirmed hotspot
- it is branch-free
- it preserves the same algorithmic map in exact arithmetic

### 2. Reparameterize the bounded state around a contractive variable

`H` and `T = I - H` both stay bounded, while the old explicit Gram-like state
did not. A better parameterization of the second bounded step may reduce TF32
damage further if it keeps all intermediate spectra in a compact interval.

What to look for:

- recurrences written only in `H` and `T`
- equivalent forms whose intermediate matrices remain PSD or contractive by
  construction

Why it is promising:

- it is structural rather than heuristic
- it attacks the remaining sensitivity without fallback branches

### 3. Replace one raw matmul with a factor-apply if it stays cheap

The bounded mode already uses the right split for solves: accurate first
inverse, fast second apply. One remaining possibility is to replace a single
fragile matmul site by a factor-based apply that is still cheaper than broad
FP32 promotion.

What to look for:

- one small-side hotspot where factor application is measurably better than TF32
- a replacement that does not introduce retries or case routing

Why it is promising:

- it stays surgical
- it may recover the last hard rows without giving back the speed win

## Current Working Rule

Until a better rewrite lands:

- keep the bounded mode branch-free
- keep the first inverse as `cholesky_solve`
- keep the second solve as two triangular solves
- focus future work on the TF32 small-side matmuls, not on the solve backend
