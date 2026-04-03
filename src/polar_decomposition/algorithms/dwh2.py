from __future__ import annotations

import torch

PAPER_MUON_ELL = 1e-3
PAPER_NORM_EPS = 1e-3
from ..utils.precond import (
    CholStats,
    PolarResult,
    _form_u,
)

_SMALLSIDE_GRAM_BLOCK_ROWS = 1024


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
        aa, bb, cc, ell = dwh_coefficients(ell)
        coeffs.append((aa, bb, cc))
    return coeffs


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    return a


def _block_gram_tree_(
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int = _SMALLSIDE_GRAM_BLOCK_ROWS,
) -> torch.Tensor:
    """Balanced split-K Gram accumulation with O(log num_blocks) live buffers."""
    partials: list[torch.Tensor | None] = []
    for start in range(0, x.shape[0], block_rows):
        stop = min(start + block_rows, x.shape[0])
        carry = x[start:stop].mT @ x[start:stop]
        level = 0
        while True:
            if level == len(partials):
                partials.append(carry)
                break
            prev = partials[level]
            if prev is None:
                partials[level] = carry
                break
            prev.add_(carry)
            partials[level] = None
            carry = prev
            level += 1

    out.zero_()
    for part in partials:
        if part is not None:
            out.add_(part)
    return out


def _smallside_factor_stable(
    a: torch.Tensor,
    stats: CholStats,
    *,
    scratch: torch.Tensor,
    jitter_scale: float = 2.0,
) -> torch.Tensor:
    """Stable factor for well-conditioned DWH Gram/Correction matrix.

    DWH matrices H = I + c G or A1 = I + ... are theoretically well-conditioned
    (all eig >= 1). We skip diagonal scaling and use a small unconditional
    shift to ensure SPD-ness in FP32/TF32 without retries.
    """
    n = a.shape[0]
    u = _form_u(a.dtype)
    # Use a larger initial jitter — O(N*eps) is roughly the accumulation error.
    jitter = max(jitter_scale * n * u, 1e-4)

    # Symmetrize
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    # Simple unconditional shift
    a.diagonal().add_(jitter)

    l, info = torch.linalg.cholesky_ex(a, check_errors=False)

    if int(info.item()) != 0:
        # Fallback for extreme cases — larger shift
        retries = 1
        jitter_step = jitter * 10.0
        while int(info.item()) != 0 and retries < 5:
            a.diagonal().add_(jitter_step)
            jitter += jitter_step
            l, info = torch.linalg.cholesky_ex(a, check_errors=False)
            retries += 1
            jitter_step *= 10.0
        
        if int(info.item()) != 0:
            # Last ditch: huge shift
            a.diagonal().add_(0.1)
            l = torch.linalg.cholesky(a)
            jitter += 0.1
            retries += 1
            
        stats.update(shifted=True, retries=retries, jitter=float(jitter), diag_floored=0)
    else:
        stats.update(shifted=True, retries=0, jitter=jitter, diag_floored=0)

    return l


def _inverse_from_cholesky_solve(
    l: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    out.zero_()
    out.diagonal().fill_(1.0)
    out.copy_(torch.cholesky_solve(out, l))
    return out


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    gram_0: torch.Tensor | None = None,
) -> PolarResult:
    """Two-step bounded small-side DWH update.

    If *gram_0* is supplied it is used as the initial small-side Gram matrix
    G = X^T X instead of recomputing it internally.  This lets the caller
    share the Gram that was already computed during normalization, eliminating
    one full rectangular GEMM from the critical path.
    """
    assert a.ndim == 2
    transposed = a.shape[0] < a.shape[1]
    x = a.mT.contiguous() if transposed else a

    n = x.shape[1]
    stats = CholStats()
    coeffs = _dwh_schedule(ell0, steps=2)
    (aa0, bb0, cc0), (aa1, bb1, cc1) = coeffs
    alpha0 = bb0 / cc0
    beta0 = aa0 - alpha0
    alpha1 = bb1 / cc1
    beta1 = aa1 - alpha1
    delta_scale = cc1 / cc0

    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    h = torch.empty((n, n), device=x.device, dtype=x.dtype)
    m_acc = torch.empty((n, n), device=x.device, dtype=x.dtype)
    k = torch.empty((n, n), device=x.device, dtype=x.dtype)
    buf = torch.empty((n, n), device=x.device, dtype=x.dtype)
    scratch = torch.empty((n, n), device=x.device, dtype=x.dtype)

    if gram_0 is not None:
        gram.copy_(gram_0)
    else:
        _block_gram_tree_(x, gram)
    h.copy_(gram).mul_(cc0)
    h.diagonal().add_(1.0)
    l0 = _smallside_factor_stable(
        h,
        stats,
        scratch=scratch,
        jitter_scale=scaled_jitter_scale,
    )
    _inverse_from_cholesky_solve(l0, out=h)

    k.copy_(h).mul_(-1.0)
    k.diagonal().add_(1.0)
    # Build A1 = I + (c1 / c0) M0 Delta0 M0 from Delta0 = c0 G0. Using
    # Delta0 H0 = I - H0 avoids the dense-RHS second solve K^{-1} H0 entirely.
    # Only the bounded H0 @ T0 product goes through TF32.
    buf.copy_(gram)
    buf.mul_(delta_scale * cc0 * alpha0 * alpha0)
    sh = torch.sqrt(torch.clamp(h.diagonal(), min=1e-30))
    invsh = sh.reciprocal()
    scratch.copy_(k)
    scratch.mul_(sh[:, None])
    m_acc.copy_(h)
    m_acc.mul_(invsh[:, None]).mul_(invsh[None, :])
    torch.mm(m_acc, scratch, out=gram)
    gram.mul_(sh[:, None])
    buf.add_(k, alpha=delta_scale * 2.0 * alpha0 * beta0)
    buf.add_(gram, alpha=delta_scale * beta0 * beta0)
    buf.diagonal().add_(1.0)
    _symmetrize_(buf, scratch)

    m_acc.copy_(h).mul_(beta0)
    m_acc.diagonal().add_(alpha0)

    l1 = _smallside_factor_stable(
        buf,
        stats,
        scratch=scratch,
        jitter_scale=scaled_jitter_scale,
    )
    # Final update: K = alpha1 M0 + beta1 (M0 A1^{-1}).
    k.copy_(m_acc)
    torch.linalg.solve_triangular(l1.mT, k, upper=True, left=False, out=k)
    torch.linalg.solve_triangular(l1, k, upper=False, left=False, out=k)
    k.mul_(beta1)
    k.add_(m_acc, alpha=alpha1)

    x = x @ k
    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)


def dwh2_hybrid(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    gram_0: torch.Tensor | None = None,
) -> PolarResult:
    """Hybrid precision DWH2.

    Uses FP16 for massive O(MN^2) GEMMs and TF32 for O(N^3) solver steps.
    """
    assert a.ndim == 2
    transposed = a.shape[0] < a.shape[1]
    x_orig = a.mT.contiguous() if transposed else a

    # 1. Cast massive matrix to FP16
    x = x_orig.half()
    n = x.shape[1]
    stats = CholStats()

    # 2. Compute Gram in FP32 (for stability, O(MN^2) is still fast on Ampere)
    # If we use FP16 for the Gram, DWH2 (2-step) fails because it needs precision
    # in the rational coefficients. 
    gram = torch.empty((n, n), device=x.device, dtype=torch.float32)
    if gram_0 is not None:
        gram.copy_(gram_0)
    else:
        # Use a high-precision Gram accumulation
        _block_gram_tree_(x_orig, gram)

    coeffs = _dwh_schedule(ell0, steps=2)
    (aa0, bb0, cc0), (aa1, bb1, cc1) = coeffs
    alpha0 = bb0 / cc0
    beta0 = aa0 - alpha0
    alpha1 = bb1 / cc1
    beta1 = aa1 - alpha1
    delta_scale = cc1 / cc0

    h = torch.empty((n, n), device=x.device, dtype=torch.float32)
    m_acc = torch.empty((n, n), device=x.device, dtype=torch.float32)
    k = torch.empty((n, n), device=x.device, dtype=torch.float32)
    buf = torch.empty((n, n), device=x.device, dtype=torch.float32)
    scratch = torch.empty((n, n), device=x.device, dtype=torch.float32)

    h.copy_(gram).mul_(cc0)
    h.diagonal().add_(1.0)
    l0 = _smallside_factor_stable(h, stats, scratch=scratch, jitter_scale=scaled_jitter_scale)
    _inverse_from_cholesky_solve(l0, out=h)

    m_k = h.mul(-1.0)
    m_k.diagonal().add_(1.0)
    buf.copy_(gram).mul_(delta_scale * cc0 * alpha0 * alpha0)

    # Simplified hybrid m_acc logic
    sh = torch.sqrt(torch.clamp(h.diagonal(), min=1e-30))
    invsh = sh.reciprocal()
    scratch.copy_(m_k).mul_(sh[:, None])
    m_acc.copy_(h).mul_(invsh[:, None]).mul_(invsh[None, :])
    buf.addmm_(m_acc, scratch, alpha=delta_scale * beta0 * beta0, beta=1.0)
    buf.add_(m_k, alpha=delta_scale * 2.0 * alpha0 * beta0)
    buf.diagonal().add_(1.0)

    l1 = _smallside_factor_stable(buf, stats, scratch=scratch, jitter_scale=scaled_jitter_scale)

    m_acc.copy_(h).mul_(beta0).diagonal().add_(alpha0)
    k.copy_(m_acc)
    torch.linalg.solve_triangular(l1.mT, k, upper=True, left=False, out=k)
    torch.linalg.solve_triangular(l1, k, upper=False, left=False, out=k)
    k.mul_(beta1).add_(m_acc, alpha=alpha1)

    # 4. Final correction in FP16
    correction = k.half()
    res = x @ correction

    # 5. Return in original dtype
    out = res.to(x_orig.dtype)
    if transposed:
        out = out.mT.contiguous()
    return PolarResult(q=out, stats=stats)
