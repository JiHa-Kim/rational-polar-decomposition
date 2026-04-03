from __future__ import annotations

import torch

from .pe5 import PAPER_MUON_ELL
from ..utils.precond import (
    CholStats,
    PolarResult,
    _form_u,
    _scale_and_symmetrize,
    spd_inverse_fast,
    spd_inverse_safe,
)
from ..kernels.triton_ops import affine_diag, can_affine_diag

DWH2_MODES = ("rectangular", "smallside_bounded")

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


def _smallside_solve_from_factor(
    rhs: torch.Tensor,
    l: torch.Tensor,
    s: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    out.copy_(rhs)
    out.mul_(s[:, None])
    torch.linalg.solve_triangular(l, out, upper=False, left=True, out=out)
    torch.linalg.solve_triangular(l.mT, out, upper=True, left=True, out=out)
    out.mul_(s[:, None])
    return out


def _dwh2_rectangular(
    a: torch.Tensor,
    *,
    ell0: float,
    robust: bool,
    scaled_jitter_scale: float,
    diag_floor_rel: float,
) -> PolarResult:
    transposed = a.shape[0] < a.shape[1]
    x = a.mT.contiguous() if transposed else a

    n = x.shape[1]
    stats = CholStats()
    fast_inverse_kwargs = {
        "diag_floor_rel": diag_floor_rel,
        "scaled_jitter_scale": scaled_jitter_scale,
    }

    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    buf = torch.empty((n, n), device=x.device, dtype=x.dtype)
    inv = torch.empty((n, n), device=x.device, dtype=x.dtype)
    ell = float(ell0)
    for _ in range(2):
        torch.mm(x.mT, x, out=gram)

        aa, bb, cc, ell = dwh_coefficients(ell)
        alpha = bb / cc
        beta = aa - alpha

        if can_affine_diag(gram, buf):
            affine_diag(gram, buf, alpha=cc, diag_add=1.0)
        else:
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

        # Paper form: (aI + bG)(I + cG)^{-1}. We rewrite it as
        # alpha I + beta (I + cG)^{-1} with alpha = b/c and beta = a - b/c.
        if can_affine_diag(inv, inv):
            affine_diag(inv, inv, alpha=beta, diag_add=alpha)
        else:
            inv.mul_(beta)
            inv.diagonal().add_(alpha)
        x = x @ inv

    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)


def _dwh2_smallside_bounded(
    a: torch.Tensor,
    *,
    ell0: float,
    robust: bool,
    scaled_jitter_scale: float,
    diag_floor_rel: float,
) -> PolarResult:
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

    m_acc.copy_(h).mul_(beta0)
    m_acc.diagonal().add_(alpha0)

    k.copy_(h).mul_(-1.0)
    k.diagonal().add_(1.0)
    torch.mm(k, m_acc, out=buf)
    # Evaluate M @ buf as alpha * buf + beta * (H @ buf) so the large identity
    # contribution stays in exact scalar arithmetic and only the bounded H part
    # goes through TF32 matmul.
    torch.mm(h, buf, out=k)
    buf.mul_(alpha0)
    k.mul_(beta0).add_(buf)
    k.mul_(delta_scale)
    k.add_(h)
    _symmetrize_(k, scratch)

    l1, s1 = _smallside_factor(
        k,
        stats,
        scratch=scratch,
        diag_floor_rel=diag_floor_rel,
        base_shift_scale=scaled_jitter_scale,
    )
    _smallside_solve_from_factor(h, l1, s1, out=buf)
    _symmetrize_(buf, scratch)
    buf.mul_(beta1)
    buf.diagonal().add_(alpha1)
    torch.mm(m_acc, buf, out=k)

    x = x @ k
    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    mode: str = "smallside_bounded",
    robust: bool = False,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
) -> PolarResult:
    """Two-step DWH with rectangular and bounded small-side update modes."""
    assert a.ndim == 2
    if mode == "rectangular":
        return _dwh2_rectangular(
            a,
            ell0=ell0,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
        )
    if mode == "smallside_bounded":
        return _dwh2_smallside_bounded(
            a,
            ell0=ell0,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
        )
    raise ValueError(f"unknown dwh2 mode {mode}")
