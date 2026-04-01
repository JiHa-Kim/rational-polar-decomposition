# Minimal DWH2 vs PE5 benchmark

This repository is a deliberately small head-to-head comparison between two fast approximate polar-factor methods on tall matrices relevant to Muon-like optimizers:

- `dwh2`: 2-step dynamically weighted Halley (DWH), implemented via small-side Gram accumulation with Cholesky-based SPD inverse.
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

$$
a_k = h(\ell_k),\qquad
b_k = \frac{(a_k - 1)^2}{4},\qquad
c_k = a_k + b_k - 1,
$$

with

$$
\ell_{k+1} = \ell_k \frac{a_k + b_k \ell_k^2}{1 + c_k \ell_k^2}.
$$

The matrix update is

$$
X_{k+1} = X_k M_k, \quad M_k = \alpha_k I + \beta_k (I + c_k G_k)^{-1},
$$

where $G_k = X_k^\top X_k$ and $\alpha_k = b_k/c_k$, $\beta_k = a_k - \alpha_k$.

**Small-side accumulation.** Since $M_k$ and $G_k$ share eigenvectors (both are functions of the same SPD matrix), we track the Gram entirely in $n \times n$ space:

$$
G_{k+1} = M_k\, G_k\, M_k, \qquad
X_{\text{final}} = X_0 (M_0 M_1).
$$

This reduces the number of large $O(mn^2)$ matmuls from 4 to 2 (one initial Gram, one final projection), with all intermediate work being $O(n^3)$.

### PE5

We use the Polar Express offline degree-5 coefficient generator with:

- the paper's cushioning,
- recentering around 1,
- the `1.01` safety factor on all but the final polynomial.

The online method is the fast rectangular small-side formulation. For a polynomial

$$
p(x) = x(a + b x^2 + c x^4),
$$

we apply it through the small-side Gram `Y = X^T X`, maintaining a small-side factor `Q` so that the large matrix is only multiplied at the end of each restart block.

We use restart interval 3. The paper suggests adding `1e-3 I` to the first Gram for more conservative low-precision stability; this minimal implementation leaves that off by default because it noticeably hurts easy-case accuracy and is not usually needed in float32/TF32.

## Matrix-only setup

The methods do not inspect singular values.

All inputs are normalized by

$$
X_0 = \frac{A}{\|A\|_F + 10^{-3}}.
$$

Both methods use the same fixed design lower bound

$$
\ell_0 = 10^{-3}
$$

by default. This is a design parameter, not an oracle estimate.

## SPD preconditioning

Every DWH solve uses the same small-side SPD inverse stack:

1. unit-diagonal scaling via `rsqrt(diag)`,
2. explicit symmetrization to fix float32 rounding from sequential scaling,
3. unconditional jitter of $O(n \cdot \epsilon)$ in the scaled space,
4. `cholesky` + `cholesky_inverse`.

A safe path with `cholesky_ex` retries and geometric jitter is available via `robust=True`.

## Cases

The default benchmark uses eleven matrix generators:

- `gaussian`: iid Gaussian baseline.
- `lognormal_cols`: Gaussian matrix with large random column scales.
- `ar1_cols`: strongly correlated columns.
- `duplicate_cols`: repeated columns plus small noise (stress case).
- `lowrank_noise`: low-rank matrix plus weak dense noise (stress case).
- `ill_conditioned`: systematically decaying singular values.
- `heavy_tail_t`: Student-t distribution with heavy tails.
- `sparse_like`: 95% sparsity pseudo-sparse matrix.
- `orthogonal_noisy`: nearly orthogonal columns.
- `rank_1_heavy`: extreme low-rank plus noise.
- `adversarial_condition`: exact condition number bound via mixing.

These are realistic matrix mechanisms rather than spectra scripted from known singular values.

## Metrics

The benchmark logs:

- `runtime_ms_median`
- `runtime_ms_min`
- `ortho_fro = ||Q^T Q - I||_F / sqrt(n)`
- `q_fro_error`, if reference polar computation is enabled
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
uv run bench --device cuda --tf32
```

Write to a JSONL file:

```bash
uv run bench --device cuda --tf32 --output runs/run1/results.jsonl --quiet
```

Skip the float64 reference pass:

```bash
uv run bench --device cuda --tf32 --reference none
```

## File layout

- `dwh2.py`: DWH2 kernel (small-side Gram accumulation) and exact scalar schedule.
- `pe5.py`: PE5 offline coefficient generator and fast online kernel.
- `precond.py`: SPD inverse via Cholesky with diagonal scaling and jitter. Shared `PolarResult` type.
- `bench.py`: realistic benchmark driver and JSONL logging.
- `profile_gpu.py`: GPU profiler with `torch.profiler` trace export.
- `README.md`: this file.
