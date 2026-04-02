# Minimal DWH2 vs PE5 benchmark

This repository is a deliberately small head-to-head comparison between two fast approximate polar-factor methods on tall matrices relevant to Muon-like optimizers:

- `dwh2`: 2-step dynamically weighted Halley (DWH), implemented in rectangular form with recomputed small-side Gram solves so full-TF32 runs stay numerically stable.
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

**Rectangular recomputation.** The current kernel recomputes the small-side Gram from the actual iterate each step,

$$
G_k = X_k^\top X_k, \qquad
X_{k+1} = X_k M_k.
$$

This uses four large $O(mn^2)$ matmuls across the two DWH steps. On the target GPU that turned out to be both faster and more stable than the theoretically leaner accumulated-Gram variant once full TF32 tensor-core matmuls were enabled.

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
4. `cholesky`,
5. inverse formation via a recursive block inverse on large CUDA `float32` blocks when
   TF32 matmuls are enabled, and two triangular solves otherwise.

The scale-and-symmetrize step also has a Triton fast path for large contiguous CUDA `float32` matrices.

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
- `objective_ratio = <Q, A> / <Q_*, A>`, if reference is enabled
- `objective_proj`, if audit is enabled: objective ratio after projecting `Q` to the nearest orthogonal factor
- DWH Cholesky health stats:
  - `chol_calls`
  - `chol_shifted_calls`
  - `chol_total_retries`
  - `chol_max_jitter`
  - `diag_floored`

Reference quality evaluation is optional and is computed only from the matrix itself via a float64 eigendecomposition of `A^T A`.

The reference path is also small-side now: it stores only the inverse square root of `A^T A` and derives `q_fro_error` chunkwise from `A @ (A^T A)^{-1/2}` instead of materializing the full reference polar factor.

The audit path is intentionally low-memory. Instead of forming a tall float64 SVD of `Q`, it accumulates `Q^T Q` and `Q^T A` in row chunks and does only the final `n x n` eigendecomposition in float64. That keeps the audit feasible on GPUs where the timed benchmark itself fits but a naive audit does not.

## Final benchmark report

Fresh current-`HEAD` benchmark on this machine:

- benchmark command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --output runs/final_current_serial_20260401.jsonl`
- shape: 16384 x 4096
- cases: 11 default cases
- measurement: warmup=1, trials=3
- execution policy: one benchmark job at a time; no overlapping runs

| Method | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **391.05 ms** | **0.06084** | **0.19371** |
| `pe5` | 664.96 ms | 0.09083 | 0.39122 |

`dwh2` is 1.70x faster by median runtime and lower on `q_fro_error` in 10/11 cases.

`q_fro_error` is the main quality metric for the raw approximate factor. Raw
`objective_ratio` is also logged, but it is not a projected metric; use `--audit`
if you want the projected-objective comparison.

| Case | DWH2 ms | PE5 ms | Speedup | DWH2 `q_fro_error` | PE5 `q_fro_error` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adversarial_condition` | **397.96** | 674.79 | 1.70x | **0.06084** | 0.12410 |
| `ar1_cols` | **388.40** | 656.94 | 1.69x | **0.10790** | 0.23701 |
| `duplicate_cols` | **390.79** | 663.66 | 1.70x | 0.24178 | **0.19284** |
| `gaussian` | **390.77** | 658.84 | 1.69x | **0.02810** | 0.08953 |
| `heavy_tail_t` | **393.73** | 670.08 | 1.70x | **0.02758** | 0.09026 |
| `ill_conditioned` | **393.92** | 668.50 | 1.70x | **0.12110** | 0.19181 |
| `lognormal_cols` | **388.83** | 661.04 | 1.70x | **0.15298** | 0.25366 |
| `lowrank_noise` | **391.05** | 664.96 | 1.70x | **0.07714** | 0.09083 |
| `orthogonal_noisy` | **396.85** | 671.16 | 1.69x | **0.03179** | 0.04190 |
| `rank_1_heavy` | **395.25** | 672.54 | 1.70x | **0.01331** | 0.01443 |
| `sparse_like` | **388.20** | 662.47 | 1.71x | **0.02801** | 0.08958 |

Detailed per-operation DWH2 profile for DWH2 on the 16384 x 4096 Gaussian case:

- profile basis: eager mode, 2 warm-up calls, 5 profiled iterations
- grouping rule: `aten::mm` split by exact input shapes; inclusive `device_time_total` for high-level nodes

| Operation                           | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :---------------------------------- | -------------: | ----: | ----------: | --------: |
| **GEMM 4096x16384x4096**            |       241.4122 |   2.0 |    120.7061 |    36.01% |
| **GEMM 16384x4096x4096**            |       216.9831 |   2.0 |    108.4915 |    32.37% |
| **Triangular Solve (small-side)**   |        99.5103 |   4.0 |     24.8776 |    14.85% |
| **Cholesky (small-side)**           |        84.5002 |   4.0 |     21.1251 |    12.84% |
| Memory / Element-wise               |        21.8806 |  40.0 |      0.5470 |     3.33% |
| Other overhead                      |         2.9031 |  14.0 |      0.2122 |     0.44% |

Detailed per-operation breakdown for PE5 (16384 x 4096, 5 iterations):

| Operation                           | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :---------------------------------- | -------------: | ----: | ----------: | --------: |
| **GEMM 4096x4096x4096**            |       681.9190 |  20.0 |     34.0960 |    59.92% |
| **GEMM 16384x4096x4096**            |       220.5398 |   2.0 |    110.2699 |    19.38% |
| **GEMM 4096x16384x4096**            |       214.6005 |   2.0 |    107.3003 |    18.86% |
| Memory / Element-wise               |        17.4540 |  32.0 |      0.5454 |     1.53% |
| Other overhead                      |         3.4432 |   1.0 |      3.4432 |     0.30% |

**Note on Ratios**: With the profiling artifact fixed, the observed latency ratio between rectangular expansion ($16k \times 4k \times 4k$) and small-side square ($4k^3$) is **~3.2x**. This is physically consistent with the $4\times$ change in FLOPs, as larger GEMMs achieve higher TFLOPS utilization.

## Run

Default run:

```bash
uv run bench --device cuda --tf32
```

For apples-to-apples benchmark numbers, run one benchmark job at a time.

Write to a JSONL file:

```bash
uv run bench --device cuda --tf32 --output runs/run1/results.jsonl --quiet
```

Skip the float64 reference pass:

```bash
uv run bench --device cuda --tf32 --reference none
```

Run the low-memory projected-objective audit on GPU:

```bash
uv run bench --device cuda --tf32 --reference fp32 --audit --audit-device same --audit-chunk-rows 512
```

Lower `--audit-chunk-rows` further if your GPU is still tight on memory.

If you only need `q_fro_error` and `objective_ratio`, `--reference fp32` already uses the low-memory small-side reference representation and avoids storing the full reference `Q_ref`.

## File layout

- `dwh2.py`: DWH2 kernel (rectangular full-TF32-stable implementation) and exact scalar schedule.
- `pe5.py`: PE5 offline coefficient generator and fast online kernel.
- `precond.py`: SPD inverse stack with fast recursive/solve paths and safe retry path. Shared `PolarResult` type.
- `triton_ops.py`: optional Triton kernels for small-side symmetrization and affine-diagonal updates.
- `bench.py`: realistic benchmark driver and JSONL logging.
- `profile_gpu.py`: GPU profiler with Chrome trace and detailed per-op breakdown.
- `README.md`: this file.
