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

This keeps `q_fro_error` comparable across different normalizers.

## Current benchmark snapshot

Fresh current-`HEAD` benchmark on this machine with the shared $\ell_0 = 10^{-3}$
setting for both methods:

- benchmark command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --output runs/final_spectral_additive_20260402.jsonl`
- shape: 16384 x 4096
- cases: 11 default cases
- measurement: warmup=1, trials=3
- normalizer: `spectral_additive`
- DWH2 mode: `rectangular` (default)
- execution policy: one benchmark job at a time

| Method | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **395.97 ms** | **0.03062** | **0.07421** |
| `pe5` | 673.85 ms | 0.08874 | 0.18627 |

`dwh2` is 1.70x faster by median runtime and lower on `q_fro_error` in 11/11
cases.

## Detailed per-case results

| Case | DWH2 ms | PE5 ms | Speedup | DWH2 `q_fro_error` | PE5 `q_fro_error` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adversarial_condition` | **414.78** | 690.83 | 1.67x | **0.50496** | 0.51884 |
| `ar1_cols` | **394.76** | 680.92 | 1.72x | **0.02868** | 0.08874 |
| `duplicate_cols` | **403.48** | 679.98 | 1.69x | **0.33345** | 0.36716 |
| `gaussian` | **390.48** | 667.68 | 1.71x | **0.03062** | 0.08845 |
| `heavy_tail_t` | **395.97** | 673.85 | 1.70x | **0.03062** | 0.08861 |
| `ill_conditioned` | **395.49** | 672.77 | 1.70x | **0.78306** | 0.83908 |
| `lognormal_cols` | **393.10** | 669.56 | 1.70x | **0.13236** | 0.19996 |
| `lowrank_noise` | **396.84** | 671.42 | 1.69x | **0.10402** | 0.11144 |
| `orthogonal_noisy` | **399.16** | 677.44 | 1.70x | **0.00077** | 0.08398 |
| `rank_1_heavy` | **399.63** | 695.34 | 1.74x | **0.01343** | 0.01437 |
| `sparse_like` | **389.76** | 667.68 | 1.71x | **0.03055** | 0.08843 |

## Rectangular reference snapshot

Fresh DWH2-only reference runs on the same benchmark path:

- rectangular command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --methods dwh2 --dwh2-mode rectangular --output runs/dwh2_rectangular_head_20260402.jsonl`
- bounded command: `uv run bench --device cuda --tf32 --reference fp32 --quiet --methods dwh2 --dwh2-mode smallside_bounded --output runs/dwh2_smallside_bounded_head_20260402.jsonl`

| Mode | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `rectangular` | 389.50 ms | **0.03147** | 0.07454 |
| `smallside_bounded` | **388.55 ms** | 0.03537 | **0.07029** |

In isolated DWH2-only reference runs, `smallside_bounded` is faster on `6/11`
cases and lower on `q_fro_error` on `8/11` cases, but the median speed edge is
only about `1.002x`. That is too small to justify keeping the more complex path
as the default, so `rectangular` remains the baseline kernel.

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

## Normalizer sweep snapshot

Fresh corrected sweep on this machine with the shared $\ell_0 = 10^{-3}$ setting,
the scale-relative reference cutoff, and the default DWH2 `rectangular`
mode:

- sweep command: `uv run norm-sweep --device cuda --tf32 --quiet --output runs/norm_sweep_additive_20260402.jsonl`
- shape: 16384 x 4096
- `spectral_additive` conservatism: `0/22` underestimates vs true spectral norm in the method-case comparison
- versus `spectral_bound`: improved `q_fro_error` on `9/11` DWH2 cases and `8/11` PE5 cases
- versus `fro`: improved `q_fro_error` on `7/11` DWH2 cases and `10/11` PE5 cases
- DWH2 median `q_fro_error`: `0.10788 -> 0.03062` vs `fro`
- PE5 median `q_fro_error`: `0.12519 -> 0.08874` vs `fro`
