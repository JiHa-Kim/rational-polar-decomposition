from __future__ import annotations

import torch

from .precond import CholStats, PolarResult, spd_inverse_fast, spd_inverse_safe


def dwh_coefficients(ell: float) -> tuple[float, float, float, float]:
    ell = float(max(min(ell, 1.0), 1e-12))
    gamma = (4.0 * (1.0 - ell * ell) / (ell**4)) ** (1.0 / 3.0)
    root = (1.0 + gamma) ** 0.5
    a = (
        root
        + 0.5
        * (8.0 - 4.0 * gamma + 8.0 * (2.0 - ell * ell) / (ell * ell * root)) ** 0.5
    )
    b = ((a - 1.0) ** 2) / 4.0
    c = a + b - 1.0
    ell_next = ell * (a + b * ell * ell) / (1.0 + c * ell * ell)
    return a, b, c, ell_next


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = 1e-3,
    robust: bool = False,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
) -> PolarResult:
    """Two-step rectangular DWH.

    The mathematically lean small-side Gram recurrence is attractive on paper,
    but on this GPU full-TF32 tensor-core matmuls make it noticeably less stable
    than simply recomputing G_k = X_k^T X_k each step. The rectangular form is
    also simpler: two Gram builds, two small-side SPD inverses, and two large
    projections.
    """
    assert a.ndim == 2
    transposed = a.shape[0] < a.shape[1]
    x = a.mT.contiguous() if transposed else a

    n = x.shape[1]
    stats = CholStats()
    fast_inverse_kwargs = {
        "diag_floor_rel": diag_floor_rel,
        "scaled_jitter_scale": scaled_jitter_scale,
    }

    # n×n buffers
    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    buf = torch.empty((n, n), device=x.device, dtype=x.dtype)
    inv = torch.empty((n, n), device=x.device, dtype=x.dtype)
    ell = float(ell0)
    for _ in range(2):
        # Large matmul: Gram
        torch.mm(x.mT, x, out=gram)

        aa, bb, cc, ell = dwh_coefficients(ell)
        alpha = bb / cc
        beta = aa - alpha

        # Form (I + c*G) in buf, keeping gram intact.
        buf.copy_(gram)
        buf.mul_(cc)
        buf.diagonal().add_(1.0)

        if robust:
            spd_inverse_safe(
                buf,
                stats,
                out=inv,
                diag_floor_rel=1e-6,
            )
        else:
            try:
                spd_inverse_fast(
                    buf,
                    stats,
                    out=inv,
                    **fast_inverse_kwargs,
                )
            except torch._C._LinAlgError:
                buf.copy_(gram)
                buf.mul_(cc)
                buf.diagonal().add_(1.0)
                spd_inverse_safe(
                    buf,
                    stats,
                    out=inv,
                    diag_floor_rel=1e-6,
                )

        # M_k = alpha*I + beta*inv  (reuse inv buffer)
        inv.mul_(beta)
        inv.diagonal().add_(alpha)

        # Large matmul: rectangular update
        x = x @ inv

    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)
