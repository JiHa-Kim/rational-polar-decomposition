# Minimal DWH2 vs PE5 benchmark

This repository is a deliberately small head-to-head comparison between two fast approximate polar-factor methods on tall matrices relevant to Muon-like optimizers:

- `dwh2`: 2-step dynamically weighted Halley (DWH), implemented as the direct rectangular iteration with small-side Cholesky solves.
- `pe5`: 5-step degree-5 Polar Express, implemented with the fast small-side rectangular trick and restart interval 3.

## Goals

The benchmark is designed to answer three questions as directly as possible:

1. On tall matrices, does 2-step DWH beat 5-step Polar Express on runtime for similar quality?
2. Under realistic matrix-only information, how much quality does each method retain?
3. On hard low-precision cases, is small-side SPD preconditioning actually needed?

This repo intentionally avoids oracle inputs such as known singular values.

## What is implemented

### DWH2

We use the DWH scalar schedule from Nakatsukasa, Bai, and Gygi,

\[
a_k = h(\ell_k),\qquad
b_k = \frac{(a_k - 1)^2}{4},\qquad
c_k = a_k + b_k - 1,
\]

with

\[
\ell_{k+1} = \ell_k \frac{a_k + b_k \ell_k^2}{1 + c_k \ell_k^2}.
\]

The matrix update is

\[
X_{k+1} = X_k (a_k I + b_k X_k^\top X_k)(I + c_k X_k^\top X_k)^{-1}.
\]

The implementation uses the exact affine-resolvent identity

\[
(aI + bG)(I + cG)^{-1} = \frac{b}{c} I + \left(a - \frac{b}{c}\right)(I + cG)^{-1},
\]

which saves a large matrix multiply per step.

### PE5

We use the Polar Express offline degree-5 coefficient generator with:

- the paper's cushioning,
- recentering around 1,
- the `1.01` safety factor on all but the final polynomial.

The online method is the fast rectangular small-side formulation. For a polynomial

\[
p(x) = x(a + b x^2 + c x^4),
\]

we apply it through the small-side Gram `Y = X^T X`, maintaining a small-side factor `Q` so that the large matrix is only multiplied at the end of each restart block.

We use restart interval 3. The paper suggests adding `1e-3 I` to the first Gram for more conservative low-precision stability; this minimal implementation leaves that off by default because it noticeably hurts easy-case accuracy and is not usually needed in float32/TF32.

## Matrix-only setup

The methods do not inspect singular values.

All inputs are normalized by

\[
X_0 = \frac{A}{\|A\|_F + 10^{-3}}.
\]

Both methods use the same fixed design lower bound

\[
\ell_0 = 10^{-3}
\]

by default. This is a design parameter, not an oracle estimate.

## SPD preconditioning

Every DWH solve uses the same small-side SPD solve stack:

1. symmetrize,
2. lightly floor the diagonal,
3. symmetric unit-diagonal scaling,
4. unshifted `cholesky_ex` first,
5. if needed, retry on the scaled matrix with geometric jitter.

The benchmark logs how often retries or shifts were actually needed.

## Cases

The default benchmark uses five matrix generators:

- `gaussian`: iid Gaussian baseline.
- `lognormal_cols`: Gaussian matrix with large random column scales.
- `ar1_cols`: strongly correlated columns.
- `duplicate_cols`: repeated columns plus small noise, giving a nearly rank-deficient hard case.
- `lowrank_noise`: low-rank matrix plus weak dense noise.

These are realistic matrix mechanisms rather than spectra scripted from known singular values.

## Metrics

The benchmark logs:

- `runtime_ms_median`
- `runtime_ms_min`
- `ortho_fro = ||Q^T Q - I||_F / sqrt(n)`
- `q_fro_error`, if float64 reference polar computation is enabled
- `objective_ratio = <Q, A> / <Q_ref, A>`, if reference is enabled
- DWH Cholesky health stats:
  - `chol_calls`
  - `chol_shifted_calls`
  - `chol_total_retries`
  - `chol_max_jitter`
  - `diag_floored`

Reference quality evaluation is optional and is computed only from the matrix itself via a float64 eigendecomposition of `A^T A`.

## Run

Default run:

```bash
python bench.py --device cuda --tf32
```

Write to a JSONL file:

```bash
python bench.py --device cuda --tf32 --output runs/run1/results.jsonl --quiet
```

Skip the float64 reference pass:

```bash
python bench.py --device cuda --tf32 --no-reference
```

## File layout

- `dwh2.py`: DWH2 kernel and exact scalar schedule.
- `pe5.py`: PE5 offline coefficient generator and fast online kernel.
- `precond.py`: shared SPD preconditioned Cholesky solve.
- `bench.py`: realistic benchmark driver and JSONL logging.
- `README.md`: this file.
