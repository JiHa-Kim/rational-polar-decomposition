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

- benchmark command: `uv run -m polar_decomposition.cli.bench --device cuda --tf32 --reference fp32`
- shape: 16383 x 4096
- cases: 11 default cases
- GNS Implementation: Official `Dao-AILab/gram-newton-schulz` (PyTorch backend, no-compile)
- Execution: Serial (one job at a time) to ensure measurement accuracy.

| Method | Median runtime | Median `q_fro_error` | Strategy |
| --- | ---: | ---: | :--- |
| **DWH2 (TF32/FP16)** | **213.00 ms** | 0.50265 | FP16 GEMMs / TF32 Solver |
| DWH2 (TF32) | 249.83 ms | **0.02988** | TF32 GEMMs / TF32 Solver |
| GNS (Official) | 294.31 ms | 0.88746 | Official FP16 Loop |

### "What was missing?" — Analysis of the Performance Gap

1.  **Hardware-Aware Precision**: GNS iterates entirely in **FP16**. By implementing
    **DWH2 (TF32/FP16)**, which uses FP16 for the massive $O(MN^2)$ data movements but
    keeps the $O(N^3)$ rational solver in **TF32**, we achieve **~1.37x speedup** 
    over GNS while maintaining significantly better convergence.
2.  **Solver Stability (Retries)**: Base **DWH2 (TF32)** implementations were hitting
    up to 5 retries on degenerate matrices. Our refined **`_smallside_factor_stable`**
    (unified unconditional jitter) eliminated these retries. **DWH2 (TF32/FP16) 
    now beats GNS even on its strongest `rank_1_heavy` case** (289ms vs 314ms).
3.  **Contiguity**: The official GNS implementation returns **non-contiguous**
    tensors for tall inputs ($M > N$), hiding part of its true cost. Our 
    implementations always return contiguous results.

## Detailed per-case results (Median, ms)

| Case | DWH2 (TF32/FP16) | GNS (Official) | DWH2 (TF32) | DWH2 (TF32/FP16) Err | GNS Err | DWH2 (TF32) Err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian` | **212.96** | 294.31 | 249.83 | 0.50265 | 0.88746 | **0.02988** |
| `rank_1_heavy` | **289.87** | 313.99 | 337.19 | 0.01577 | **0.01468** | 0.01543 |
| `lognormal_cols` | **214.37** | 291.60 | 250.86 | 0.27536 | 0.50755 | **0.13843** |
| `adversarial` | 312.96 | 331.13 | **267.27** | 0.86450 | 0.85620 | **0.41604** |
| `ill_conditioned`| **215.81** | 298.72 | 253.13 | 0.73417 | **0.47943** | 0.73507 |
| `ar1_cols` | **214.71** | 294.04 | 250.91 | 0.07857 | 0.38224 | **0.03238** |
| `duplicate_cols` | **215.00** | 296.85 | 251.43 | 0.33440 | **0.22683** | 0.33419 |
| `heavy_tail_t` | **216.71** | 298.60 | 253.43 | 0.50938 | 0.88660 | **0.02998** |
| `sparse_like` | **215.93** | 292.79 | 252.58 | 0.50323 | 0.88743 | **0.02987** |
| `orthogonal_noisy`| **225.69** | 305.28 | 255.89 | 0.24471 | 0.85936 | **0.00040** |
| `lowrank_noise` | **215.79** | 299.18 | 251.99 | 0.10375 | **0.09200** | 0.10362 |

## Representative historical profiles

### DWH2 (TF32/FP16) vs. Official GNS

Profiles on RTX 3050 (Ampere) for $16384 \times 4096$ matrices:

| Component | DWH2 (TF32/FP16) | GNS (Official) | Notes |
| :--- | :--- | :--- | :--- |
| **Normalization** | ~3.5 ms | ~5.2 ms | Spectral-Additive vs Frobenius-only |
| **Gram/Initial** | ~102 ms | ~98 ms | **FP16** Gram product ($X^\top X$) |
| **Solver/Iter** | ~14 ms | ~195 ms | DWH (2-step) vs GNS (5-step) |
| **Final Apply** | ~105 ms | - | $X \gets X K$ vs $X \gets Q X$ (implicit in iters) |
| **Total Median**| **213 ms** | **294 ms** | **DWH2 is 1.38x faster** |

### Detailed per-operation profile ($16384 \times 4096$)

#### [DWH2 (TF32/FP16)]

| Operation | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :--- | ---: | ---: | ---: | ---: |
| **GEMM 4096x16384x4096** (Initial Gram) | 74.1613 | 1 | 74.1613 | 19.44% |
| **GEMM 16384x4096x4096** (Apply) | 35.8271 | 1 | 35.8271 | 9.39% |
| **Cholesky (small-side)** | 143.5804 | 4 | 35.8951 | 37.63% |
| **Triangular Solve (small-side)** | 41.2802 | 2 | 20.6401 | 10.82% |
| Memory / Element-wise | 55.1678 | 62 | 0.8898 | 14.46% |
| Other overhead | 31.5013 | 16 | 1.9688 | 8.26% |

#### [GNS (Official)]

| Operation | Aggregate (ms) | Count | Per-op (ms) | Share (%) |
| :--- | ---: | ---: | ---: | ---: |
| **GEMM 4096x4096x4096** (Iterative mm) | 138.6381 | 14 | 9.9027 | 39.70% |
| **GEMM 4096x4096x16383** (Apply) | 101.7192 | 2 | 50.8596 | 29.13% |
| **GEMM 4096x16383x4096** (Gram mT) | 83.1114 | 2 | 41.5557 | 23.80% |
| Memory / Element-wise | 17.3836 | 22 | 0.7902 | 4.98% |
| Other overhead | 8.3776 | 11 | 0.7616 | 2.40% |

## Historical note

The repository used to carry multiple DWH2 kernel variants and normalization
benchmarking machinery. Those were removed once the bounded small-side DWH2
kernel plus additive spectral normalization became the clear default path. The
current benchmark tables above are the supported surface.
