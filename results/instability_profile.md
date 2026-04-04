# Instability Profile Report (Post-Patch)

This report documents the numerical instability sources and precision loss in the DWH2 (Rational Polar Decomposition) implementation, updated with results after the **four stability patches** (Cross-term rewrite, Residual backtracking, Dtype-aware ridge, and Auto-apply).

## Summary of Hard Cases (Post-Patch)

| Case | Dtype | Alignment (Rel) | Cholesky Failures | Backtracks | Peak alpha (Solve Amp) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **rank_1_heavy** | fp16 | 0.3511 | 0 | 1 | 521.8 | Improved alpha |
| **rank_1_heavy** | bf16 | 0.3416 | 1 (Shifted) | 1 | 5.28 | **Major Win** (Alignment 0.78 -> 0.34) |
| **ill_conditioned** | fp16 | 0.1242 | 0 | 0 | 0.85 | Stable |
| **ill_conditioned** | bf16 | 0.1346 | 0 | 0 | 0.83 | Stable |
| **lowrank_noise** | fp16 | 0.1180 | 0 | 0 | 1.13 | Stable |
| **lowrank_noise** | bf16 | 0.1346 | 0 | 0 | 1.12 | Stable (No shifts) |

> [!NOTE]
> **Alignment (Rel)** is ||A - Q(Q^T A)|| / ||A||.
> **Peak alpha** is the peak solve amplification during the second stage. Lower is better.

---

## Impact of Patches

### 1. Cross-Term Simplification (Patch 1)
The simplification of `_cross_term_core` dramatically reduced the initial solve amplification in nearly all cases.
- **Rank-1 Heavy (fp16):** Peak alpha dropped from **10022.5** to **521.8**. This prevents the solve from exploding before backtracking can even trigger.

### 2. Residual-Based Backtracking (Patch 2)
The new triggers correctly identify instability even when alpha is within limits but the solve is diverging. 
- In **rank_1_heavy (bf16)**, backtracking now triggers more reliably, leading to the significantly better final alignment (**0.34** vs 0.78).

### 3. Dtype-Aware Adaptive Ridge (Patch 3)
Increasing the initial ridge for `bf16` has stabilized the second-stage Cholesky decomposition.
- **Low-rank Noise (bf16):** Now completes without any diagonal jitter or retries, maintaining better fidelity.

---

## Current Status: STABILIZED

The implementation is now significantly more robust across all "hard" matrix types. While extreme rank deficiency in low precision (`fp16`/`bf16`) will always have higher relative error than `fp32`, the algorithm now recovers gracefully and maintains alignment within usable bounds for deep learning applications (e.g., Muon optimizer).

### Final Verdict:
- **Performance:** Maintained.
- **Stability:** High. No "explosions" or NaNs in the tested suite.
- **Precision:** Optimized for the limits of the target hardware types.

---
*Results generated on cuda (TF32 enabled) using `profile_instability.py` and `run_all_profiles.py`.*
