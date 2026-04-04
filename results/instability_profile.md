# 📊 Instability Profile Report

This report documents the numerical instability sources and precision loss in the DWH2 (Rational Polar Decomposition) implementation, based on comprehensive profiling of "hard" cases across `fp16` and `bf16` precisions.

## 📈 Summary of Hard Cases

| Case | Dtype | Ortho (Ref) | Alignment (Rel) | P-Err (Rel) | Cholesky Failures | Backtracks | Max $\alpha$ (Solve Amp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **rank_1_heavy** | fp16 | 0.9999 | 0.3514 | 0.3900 | 0 | 1 | 10022.5 |
| **rank_1_heavy** | bf16 | 0.9994 | 0.7801 | 0.5690 | 1 (Shifted) | 0 | 5.57 |
| **ill_conditioned** | fp16 | 0.7077 | 0.0969 | 0.1753 | 0 | 0 | 0.85 |
| **ill_conditioned** | bf16 | 0.6705 | 0.1059 | 0.1752 | 0 | 0 | 0.83 |
| **lowrank_noise** | fp16 | 0.9916 | 0.0319 | 0.1277 | 0 | 0 | 1.13 |
| **lowrank_noise** | bf16 | 0.9635 | 0.0477 | 0.1349 | 0 | 0 | 1.13 |
| **adversarial_cond** | fp16 | 0.0695 | 0.0593 | 0.0578 | 0 | 0 | 0.28 |
| **adversarial_cond** | bf16 | 0.0695 | 0.0593 | 0.0578 | 0 | 0 | 0.28 |

> [!NOTE]
> **Ortho (Ref)** is the relative Frobenius error compared to the identity matrix (for full rank) or the range projector (for rank-deficient). For these cases, it shows how well $Q^T Q$ matches the expected projector.
> **Alignment (Rel)** is $||A - Q(Q^T A)|| / ||A||$.
> **P-Err (Rel)** is $||P^2 - A^T A|| / ||A^T A||$.

---

## 🔍 Key Findings

### 1. The "Rank-1 Heavy" Instability
In the `rank_1_heavy` case, the matrix is almost rank-1 with a small noise. 
- **fp16 Behavior:** Triggered a backtrack due to a massive `alpha` (10022.5) during the second stage solve. The backtracking mechanism correctly reduced `theta` to 0.5, bringing `alpha` down to 1.33. However, the final error remains high (~39%).
- **bf16 Behavior:** Failed the initial Cholesky decomposition of the second stage buffer. The `_chol_spd_inplace_ex` mechanism added jitter (0.0107) to stabilize it. This prevented the backtrack but resulted in higher final alignment error (0.78).

> [!WARNING]
> The current backtracking limit `_BACKTRACK_ALPHA_LIMIT = 10.0` is robust for many cases but may be too lenient or triggered too late for extremely rank-deficient inputs in lower precision.

### 2. Cholesky Fragility in bf16
`bf16` consistently shows more Cholesky failures than `fp16` for the same inputs.
- In `rank_1_heavy (bf16)`, `chol1_probe` reported `info=4079`, indicating non-positive definiteness even after the theoretical $I + cG$ construction.
- The jitter-based stabilization (`_chol_spd_inplace_ex`) keeps the algorithm running but injects direct error into the solve, explaining the degraded alignment.

### 3. Solve Amplification ($\alpha$)
The metric `alpha = ||tmp|| / ||m0||` is a strong indicator of instability.
- When $\alpha \gg 1$, the linear system in the second stage is extremely ill-conditioned. 
- This is most prominent when the gram matrix $G$ has eigenvalues very close to the boundaries handled by the rational approximation.

### 4. Skewness and Symmetrization
While `s_c` (matrix skew) was 0.0 in most logs, this is because `_symmetrize_` is called aggressively. However, the *implicit* skew before symmetrization in the cross-term construction can still lead to precision loss.

---

## 🛠️ Recommendations

1.  **Adaptive Ridge Tuning:** The `_ADAPTIVE_RIDGE_SCALE` (1e-6) could be slightly increased for `bf16` specifically to avoid the Cholesky retry loop.
2.  **Backtrack Sensitivity:** Lowering `_BACKTRACK_ALPHA_LIMIT` might force more iterations or better `theta` scaling earlier in the process.
3.  **Cross-Term Precision:** The `cross_term` calculation is a known hotspot for precision loss. Investigating a more stable form for `tmp @ scratch` in `fp16` could help.

---

### Verification
- All results generated using `profile_instability.py` on `cuda` with `TF32` enabled (default).
- Reference `fp32` runs confirm that the mathematical logic is sound; the issues are primarily precision-depth related.
