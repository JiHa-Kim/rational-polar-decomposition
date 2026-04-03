from __future__ import annotations

import torch

from .pe5 import PAPER_MUON_ELL
from ..utils.precond import (
    CholStats,
    PolarResult,
    _form_u,
    _scale_and_symmetrize,
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


def _smallside_factor(
    a: torch.Tensor,
    stats: CholStats,
    *,
    scratch: torch.Tensor,
    diag_floor_rel: float,
    base_shift_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a small SPD block after unit-diagonal scaling, with retry jitter."""
    mat, s, diag_floored = _scale_and_symmetrize(
        a,
        diag_floor_rel,
        scratch=scratch,
    )

    base_shift = max(base_shift_scale * a.shape[0] * _form_u(a.dtype), 1e-7)
    shifted = False
    retries = 0
    jitter = 0.0

    l, info = torch.linalg.cholesky_ex(mat, check_errors=False)
    while int(info.item()) != 0 and retries < 6:
        shifted = True
        jitter = base_shift * (10.0**retries)
        scratch.copy_(mat)
        scratch.diagonal().add_(jitter)
        l, info = torch.linalg.cholesky_ex(scratch, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        shifted = True
        jitter = base_shift * (10.0**retries)
        scratch.copy_(mat)
        scratch.diagonal().add_(jitter)
        l = torch.linalg.cholesky(scratch)
        retries += 1

    stats.update(
        shifted=shifted,
        retries=retries,
        jitter=jitter,
        diag_floored=diag_floored,
    )
    return l, s


def _smallside_inverse_from_factor(
    l: torch.Tensor,
    s: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    out.zero_()
    out.diagonal().fill_(1.0)
    out.copy_(torch.cholesky_solve(out, l))
    out.mul_(s[:, None]).mul_(s[None, :])
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
    l0, s0 = _smallside_factor(
        h,
        stats,
        scratch=scratch,
        diag_floor_rel=diag_floor_rel,
        base_shift_scale=scaled_jitter_scale,
    )
    _smallside_inverse_from_factor(l0, s0, out=h)
    _symmetrize_(h, scratch)

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

    l1, s1 = _smallside_factor(
        buf,
        stats,
        scratch=scratch,
        diag_floor_rel=diag_floor_rel,
        base_shift_scale=scaled_jitter_scale,
    )
    # Final update: K = alpha1 M0 + beta1 (M0 A1^{-1}). Applying the second
    # inverse directly to M0 avoids forming H1 explicitly and removes the last
    # small-side GEMM from the bounded path.
    k.copy_(m_acc)
    k.mul_(s1[None, :])
    torch.linalg.solve_triangular(l1.mT, k, upper=True, left=False, out=k)
    torch.linalg.solve_triangular(l1, k, upper=False, left=False, out=k)
    k.mul_(s1[None, :])
    k.mul_(beta1)
    k.add_(m_acc, alpha=alpha1)

    x = x @ k
    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)
