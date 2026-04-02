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
This is algebraically the same as the paper's

$$
X_{k+1} = X_k (a_k I + b_k G_k)(I + c_k G_k)^{-1},
$$

just rewritten to avoid explicitly forming the $a_k I + b_k G_k$ factor.

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

we apply it through the small-side Gram $Y = X^\top X$, maintaining a small-side factor $Q$ so that the large matrix is only multiplied at the end of each restart block.

We use restart interval 3. The paper suggests adding $10^{-3} I$ to the first Gram for more conservative low-precision stability; this minimal implementation leaves that off by default because it noticeably hurts easy-case accuracy and is not usually needed in float32/TF32.

## Matrix-only setup

The methods do not inspect singular values.

The default benchmark normalizes inputs by a one-sided 2-moment spectral bound.
For a tall view $X$ and the computed small-side Gram $\widehat G = X^\top X$, we use

$$
t_1 = \|X\|_F^2,\qquad
t_{2,\mathrm{ub}} = \bigl(\|\widehat G\|_F + \eta \|X\|_F^2\bigr)^2,
$$

followed by the PSD moment bound

$$
\alpha_{\mathrm{ub}}
=
\sqrt{
\frac{
t_1 + \sqrt{(n - 1)(n t_{2,\mathrm{ub}} - t_1^2)}
}{n}
}.
$$

The normalized input is then

$$
X_0 = \frac{A}{\alpha_{\mathrm{ub}} + 10^{-3}}.
$$

On CUDA `float32` with TF32 tensor-core matmuls enabled, $\eta$ is the simple
dot-product envelope $2^{-10} + \gamma_m^{(32)}$, where $m$ is the Gram
inner-product length. So the default scale is conservative for a written-down
finite-precision model, not an empirical ridge knob.

The shared CLI design lower bound is

$$
\ell_0 = 10^{-3}
$$

by default. This is a design parameter, not an oracle estimate.

### Baseline normalizer

The only alternative CLI baseline is:

- `fro`: the QDWH-style practical upper bound $\|A\|_F$

Fresh corrected sweep on this machine with the shared $\ell_0 = 10^{-3}$ setting and
the scale-relative reference cutoff:

- sweep command: `uv run norm-sweep --device cuda --tf32 --quiet --output runs/norm_sweep_spectral_bound_full_20260402.jsonl`
- shape: 16384 x 4096
- `spectral_bound` conservatism: `0/22` underestimates vs true spectral norm in the method-case comparison, with minimum estimated/true spectral ratio `1.00092217`
- versus `fro`: improved `q_fro_error` on `7/11` DWH2 cases and `10/11` PE5 cases
- DWH2 median `q_fro_error`: `0.10788 -> 0.03147` vs `fro`
- PE5 median `q_fro_error`: `0.12519 -> 0.08877` vs `fro`

So the current default is `spectral_bound`, with `fro` kept only as the simple
baseline.

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
- `ortho_fro`: $\|Q^\top Q - I\|_F / \sqrt{n}$
- `q_fro_error`, if reference polar computation is enabled
- `objective_ratio`: $\langle Q, A \rangle / \langle Q_*, A \rangle$, if reference is enabled
- `objective_proj`, if audit is enabled: objective ratio after projecting `Q` to the nearest orthogonal factor
- DWH Cholesky health stats:
  - `chol_calls`
  - `chol_shifted_calls`
  - `chol_total_retries`
  - `chol_max_jitter`
  - `diag_floored`

Reference quality evaluation is optional and is computed only from the matrix itself via a float64 eigendecomposition of $A^\top A$.

The reference path is also small-side now: it stores only the inverse square root of $A^\top A$ and derives `q_fro_error` chunkwise from $A (A^\top A)^{-1/2}$ instead of materializing the full reference polar factor.

The reference cutoff is scale-relative:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon \lambda_{\max}(A^\top A)).
$$

This keeps `q_fro_error` comparable across different normalizers. The earlier
absolute cutoff $\epsilon \max(\lambda_{\max}, 1)$ made Fro-like
over-normalization look artificially good on low-rank cases.

The audit path is intentionally low-memory. Instead of forming a tall float64 SVD of $Q$, it accumulates $Q^\top Q$ and $Q^\top A$ in row chunks and does only the final $n \times n$ eigendecomposition in float64. That keeps the audit feasible on GPUs where the timed benchmark itself fits but a naive audit does not.

## Final benchmark report

Fresh current-`HEAD` benchmark on this machine with the shared $\ell_0 = 10^{-3}$
setting for both methods:

- benchmark command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --output runs/final_spectral_bound_serial_20260402.jsonl`
- shape: 16384 x 4096
- cases: 11 default cases
- measurement: warmup=1, trials=3
- normalizer: `spectral_bound`
- execution policy: one benchmark job at a time; no overlapping runs

| Method | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **405.90 ms** | **0.03147** | **0.07454** |
| `pe5` | 690.11 ms | 0.08877 | 0.18633 |

`dwh2` is 1.70x faster by median runtime and lower on `q_fro_error` in 11/11 cases.

`q_fro_error` is the main quality metric for the raw approximate factor. Raw
`objective_ratio` is also logged, but it is not a projected metric; use `--audit`
if you want the projected-objective comparison.

| Case | DWH2 ms | PE5 ms | Speedup | DWH2 `q_fro_error` | PE5 `q_fro_error` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adversarial_condition` | **402.74** | 696.63 | 1.73x | **0.51762** | 0.53158 |
| `ar1_cols` | **402.39** | 666.37 | 1.66x | **0.02869** | 0.08877 |
| `duplicate_cols` | **398.13** | 677.87 | 1.70x | **0.33440** | 0.36799 |
| `gaussian` | **406.94** | 678.19 | 1.67x | **0.03147** | 0.08849 |
| `heavy_tail_t` | **418.73** | 699.41 | 1.67x | **0.03118** | 0.08836 |
| `ill_conditioned` | **405.90** | 690.11 | 1.70x | **0.78345** | 0.83943 |
| `lognormal_cols` | **399.90** | 681.63 | 1.70x | **0.13339** | 0.20085 |
| `lowrank_noise` | **411.94** | 696.05 | 1.69x | **0.10389** | 0.11190 |
| `orthogonal_noisy` | **417.95** | 698.65 | 1.67x | **0.02867** | 0.08751 |
| `rank_1_heavy` | **400.25** | 676.08 | 1.69x | **0.01343** | 0.01437 |
| `sparse_like` | **406.28** | 699.98 | 1.72x | **0.03140** | 0.08841 |

Representative per-operation breakdown for DWH2 on the 16384 x 4096 Gaussian
case:

- profile basis: eager mode, 2 warm-up calls, 5 profiled iterations
- grouping rule: `aten::mm` split by exact input shapes; inclusive `device_time_total` for high-level nodes

| Operation                           | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :---------------------------------- | -------------: | ----: | ----------: | --------: |
| **GEMM 4096x16384x4096**            |       167.3036 |     2 |     83.6518 |    36.41% |
| **GEMM 16384x4096x4096**            |       144.0689 |     2 |     72.0344 |    31.35% |
| **Cholesky (small-side)**           |        71.6286 |     4 |     17.9072 |    15.59% |
| **GEMM 4096x4096x4096**             |        36.9212 |     2 |     18.4606 |     8.04% |
| Memory / Element-wise               |        13.6238 |   192 |      0.0710 |     2.96% |
| **Triangular Solve (small-side)**   |         5.1873 |    16 |      0.3242 |     1.13% |

Detailed per-operation breakdown for PE5 (16384 x 4096, 5 iterations):

| Operation                           | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :---------------------------------- | -------------: | ----: | ----------: | --------: |
| **GEMM 4096x4096x4096**             |       381.8374 |    20 |     19.0919 |    52.71% |
| **GEMM 4096x16384x4096**            |       166.9725 |     2 |     83.4863 |    23.05% |
| **GEMM 16384x4096x4096**            |       153.0491 |     2 |     76.5246 |    21.13% |
| Memory / Element-wise               |        18.7676 |    32 |      0.5865 |     2.59% |
| Other overhead                      |         3.8470 |     1 |      3.8470 |     0.53% |

The profile picture is consistent with the benchmark table: DWH2 is dominated by
two rectangular updates plus two small-side solves, while PE5 spends most of
its time in the 20 small-side $4096^3$ GEMMs from the polynomial steps.

## Run

Default run:

```bash
uv run bench --device cuda --tf32
```

Try the Frobenius baseline:

```bash
uv run bench --device cuda --tf32 --normalizer fro
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

Run the normalization sweep:

```bash
uv run norm-sweep --device cuda --tf32 --quiet --output runs/norm_sweep/results.jsonl
```

## File layout

- `dwh2.py`: DWH2 kernel (rectangular full-TF32-stable implementation) and exact scalar schedule.
- `pe5.py`: PE5 offline coefficient generator and fast online kernel.
- `normalization.py`: Frobenius and one-sided spectral-bound input scaling helpers.
- `precond.py`: SPD inverse stack with fast recursive/solve paths and safe retry path. Shared `PolarResult` type.
- `triton_ops.py`: optional Triton kernels for small-side symmetrization and affine-diagonal updates.
- `bench.py`: realistic benchmark driver and JSONL logging.
- `norm_sweep.py`: normalization recipe sweep harness and conservatism audit.
- `profile_gpu.py`: GPU profiler with Chrome trace and detailed per-op breakdown.
- `README.md`: this file.
