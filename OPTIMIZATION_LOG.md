# Optimization & Anti-Optimization Log

This file tracks implementation attempts, their outcomes, and reasons for rejection to avoid redundant work.

## Performance Rejections (Anti-Optimizations)

### `torch.cholesky_inverse`
- **Attempted**: Replaced manual recursive inversion (`_spd_inv_from_cholesky`) with `torch.cholesky_inverse`.
- **Outcome**: REGRESSION.
- **Metrics**: Increased median time from ~185ms to ~213ms on hard cases.
- **Reason**: While a single LAPACK call, it seems to inhibit certain Inductor optimizations or introduces kernel launch overhead that outweighs its theoretical efficiency in this specific iterative context.
- **Status**: DO NOT USE. Manual recursive inversion is faster under `torch.compile`.

### `torch.cholesky_solve` (for Stage 2)
- **Attempted**: Replaced `solve_triangular` double-pass with `cholesky_solve`.
- **Outcome**: REGRESSION (~214ms).
- **Reason**: Inductor currently optimizes the triangular solve pattern better than the `cholesky_solve` op on this environment.
- **Status**: DO NOT USE.
