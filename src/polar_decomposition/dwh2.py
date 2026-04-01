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


def _dwh_schedule(ell0: float, steps: int = 2) -> list[tuple[float, float, float]]:
    coeffs: list[tuple[float, float, float]] = []
    ell = float(ell0)
    for _ in range(steps):
        a, b, c, ell = dwh_coefficients(ell)
        coeffs.append((a, b, c))
    return coeffs


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = 1e-3,
    tf32: bool = True,
    robust: bool = False,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
) -> PolarResult:
    """Two-step DWH via small-side accumulation.

    All O(n³) work is done on the n×n Gram side. Only two O(mn²) matmuls
    are performed: the initial Gram and the final projection. This halves
    the large-matmul count vs the naive rectangular iteration.

    X_{k+1} = X_k @ M_k  where  M_k = α_k I + β_k (I + c_k G_k)^{-1}
    G_{k+1} = M_k @ G_k @ M_k   (M_k and G_k commute)
    X_final = X_0 @ (M_0 @ M_1)
    """
    assert a.ndim == 2
    transposed = a.shape[0] < a.shape[1]
    x = a.mT.contiguous() if transposed else a

    n = x.shape[1]
    stats = CholStats()
    inverse_fn = spd_inverse_safe if robust else spd_inverse_fast

    # n×n buffers
    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    buf = torch.empty((n, n), device=x.device, dtype=x.dtype)
    inv = torch.empty((n, n), device=x.device, dtype=x.dtype)
    m_acc = torch.eye(n, device=x.device, dtype=x.dtype)

    # Single large matmul: Gram
    torch.mm(x.mT, x, out=gram)

    coeffs = _dwh_schedule(ell0=ell0, steps=2)
    for aa, bb, cc in coeffs:
        alpha = bb / cc
        beta = aa - alpha

        # Form (I + c*G) in buf, keeping gram intact
        buf.copy_(gram)
        buf.mul_(cc)
        buf.diagonal().add_(1.0)

        # inv = (I + c*G)^{-1}
        inverse_fn(
            buf,
            stats,
            tf32=tf32,
            out=inv,
            diag_floor_rel=(1e-6 if robust else diag_floor_rel),
            **({} if robust else {"scaled_jitter_scale": scaled_jitter_scale}),
        )

        # M_k = alpha*I + beta*inv  (reuse inv buffer)
        inv.mul_(beta)
        inv.diagonal().add_(alpha)

        # G_{k+1} = M_k @ G_k @ M_k  (all n×n)
        torch.mm(gram, inv, out=buf)
        torch.mm(inv, buf, out=gram)

        # Accumulate: m_acc = m_acc @ M_k
        torch.mm(m_acc, inv, out=buf)
        m_acc, buf = buf, m_acc

    # Single large matmul: final projection
    result = x @ m_acc

    if transposed:
        result = result.mT.contiguous()
    return PolarResult(q=result, stats=stats)
