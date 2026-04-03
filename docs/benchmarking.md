# Benchmarking Notes

## Cases

The default benchmark uses eleven matrix generators:

- `gaussian`: iid Gaussian baseline.
- `lognormal_cols`: Gaussian matrix with large random column scales.
- `ar1_cols`: strongly correlated columns.
- `duplicate_cols`: repeated columns plus small noise.
- `lowrank_noise`: low-rank matrix plus weak dense noise.
- `ill_conditioned`: systematically decaying singular values.
- `heavy_tail_t`: Student-t distribution with heavy tails.
- `sparse_like`: 95% sparsity pseudo-sparse matrix.
- `orthogonal_noisy`: nearly orthogonal columns.
- `rank_1_heavy`: extreme low-rank plus noise.
- `adversarial_condition`: exact condition number bound via mixing.

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

The reference path stores only the inverse square root of $A^\top A$ and derives `q_fro_error` chunkwise from $A (A^\top A)^{-1/2}$ instead of materializing the full reference polar factor.

The reference cutoff is scale-relative:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon \lambda_{\max}(A^\top A)).
$$

This keeps `q_fro_error` comparable across benchmark runs.

## Current benchmark snapshot

Fresh current-`HEAD` benchmark on this machine with the shared $\ell_0 = 10^{-3}$
setting for both methods:

- benchmark command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --output runs/final_smallside_bounded_finalsolve_ref_20260403.jsonl`
- shape: 16384 x 4096
- cases: 11 default cases
- measurement: warmup=1, trials=1
- normalization: `spectral_additive`
- DWH2 kernel: bounded small-side update
- execution policy: one benchmark job at a time

| Method | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **344.71 ms** | **0.02963** | **0.06763** |
| `pe5` | 666.72 ms | 0.08874 | 0.18627 |

`dwh2` is 1.93x faster by median runtime and lower on `q_fro_error` in 11/11
cases.

## Detailed per-case results

| Case | DWH2 ms | PE5 ms | Speedup | DWH2 `q_fro_error` | PE5 `q_fro_error` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adversarial_condition` | **347.98** | 676.51 | 1.94x | **0.49690** | 0.51884 |
| `ar1_cols` | **343.10** | 659.94 | 1.92x | **0.02845** | 0.08874 |
| `duplicate_cols` | **344.27** | 665.24 | 1.93x | **0.32876** | 0.36716 |
| `gaussian` | **341.71** | 666.72 | 1.95x | **0.02951** | 0.08845 |
| `heavy_tail_t` | **346.73** | 670.00 | 1.93x | **0.02963** | 0.08861 |
| `ill_conditioned` | **346.38** | 668.03 | 1.93x | **0.77064** | 0.83908 |
| `lognormal_cols` | **343.44** | 665.59 | 1.94x | **0.13201** | 0.19996 |
| `lowrank_noise` | **344.71** | 666.10 | 1.93x | **0.10410** | 0.11144 |
| `orthogonal_noisy` | **346.59** | 675.80 | 1.95x | **0.00030** | 0.08398 |
| `rank_1_heavy` | **350.95** | 679.18 | 1.94x | **0.01397** | 0.01437 |
| `sparse_like` | **341.19** | 661.83 | 1.94x | **0.02947** | 0.08843 |

## Representative historical profiles

These profile tables were captured before the bounded mode became the default.
They are still useful for rough operation mix, but they are not the current
headline benchmark numbers.

Representative per-operation breakdown for DWH2 on the 16384 x 4096 Gaussian
case:

- profile basis: eager mode, 2 warm-up calls, 5 profiled iterations
- grouping rule: `aten::mm` split by exact input shapes; inclusive `device_time_total` for high-level nodes

| Operation | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :--- | ---: | ---: | ---: | ---: |
| **GEMM 4096x16384x4096** | 167.3036 | 2 | 83.6518 | 36.41% |
| **GEMM 16384x4096x4096** | 144.0689 | 2 | 72.0344 | 31.35% |
| **Cholesky (small-side)** | 71.6286 | 4 | 17.9072 | 15.59% |
| **GEMM 4096x4096x4096** | 36.9212 | 2 | 18.4606 | 8.04% |
| Memory / Element-wise | 13.6238 | 192 | 0.0710 | 2.96% |
| **Triangular Solve (small-side)** | 5.1873 | 16 | 0.3242 | 1.13% |

Detailed per-operation breakdown for PE5 on the same shape:

| Operation | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :--- | ---: | ---: | ---: | ---: |
| **GEMM 4096x4096x4096** | 381.8374 | 20 | 19.0919 | 52.71% |
| **GEMM 4096x16384x4096** | 166.9725 | 2 | 83.4863 | 23.05% |
| **GEMM 16384x4096x4096** | 153.0491 | 2 | 76.5246 | 21.13% |
| Memory / Element-wise | 18.7676 | 32 | 0.5865 | 2.59% |
| Other overhead | 3.8470 | 1 | 3.8470 | 0.53% |

## Historical note

The repository used to carry multiple DWH2 kernel variants and normalization
benchmarking machinery. Those were removed once the bounded small-side DWH2
kernel plus additive spectral normalization became the clear default path. The
current benchmark tables above are the supported surface.
