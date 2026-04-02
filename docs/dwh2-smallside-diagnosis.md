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
- applies the second solve as two triangular solves

Integrated 11-case sweep at `16384 x 4096`, `seed=0`, `spectral_bound` normalization:

- `rectangular` median runtime: `391.35 ms`
- `smallside_bounded` median runtime: `374.03 ms`
- faster on `11/11`
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

## Current Working Rule

Until a better rewrite lands:

- keep the bounded mode branch-free
- keep the first inverse as `cholesky_solve`
- keep the second solve as two triangular solves
- focus future work on the TF32 small-side matmuls, not on the solve backend
