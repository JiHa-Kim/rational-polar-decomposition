from __future__ import annotations

from contextlib import contextmanager

import torch

from .pe5 import PAPER_MUON_ELL
from ..utils.precond import CholStats, PolarResult, spd_inverse_fast, spd_inverse_safe
from ..kernels.triton_ops import affine_diag, can_affine_diag

DWH2_MODES = ("rectangular", "smallside_delta")


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


@contextmanager
def _fp32_small_matmuls():
    if not torch.cuda.is_available():
        yield
        return

    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        torch.set_float32_matmul_precision(prev_prec)


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    return a


def _smallside_inverse(
    a: torch.Tensor,
    stats: CholStats,
    *,
    robust: bool,
    scaled_jitter_scale: float,
    diag_floor_rel: float,
    out: torch.Tensor,
) -> torch.Tensor:
    if robust:
        spd_inverse_safe(
            a,
            stats,
            out=out,
            diag_floor_rel=max(diag_floor_rel, 1e-6),
            base_shift_scale=scaled_jitter_scale,
        )
        return out

    try:
        spd_inverse_fast(
            a,
            stats,
            out=out,
            scaled_jitter_scale=0.0,
            diag_floor_rel=diag_floor_rel,
        )
    except torch._C._LinAlgError:
        spd_inverse_safe(
            a,
            stats,
            out=out,
            diag_floor_rel=max(diag_floor_rel, 1e-6),
            base_shift_scale=scaled_jitter_scale,
        )
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


def _dwh2_smallside_delta(
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
    delta = torch.empty((n, n), device=x.device, dtype=x.dtype)
    buf = torch.empty((n, n), device=x.device, dtype=x.dtype)
    inv = torch.empty((n, n), device=x.device, dtype=x.dtype)
    m_acc = torch.empty((n, n), device=x.device, dtype=x.dtype)
    scratch = torch.empty((n, n), device=x.device, dtype=x.dtype)

    torch.mm(x.mT, x, out=gram)
    _symmetrize_(gram, scratch)
    delta.copy_(gram).mul_(cc0)

    with _fp32_small_matmuls():
        # Step 0
        buf.copy_(delta)
        buf.diagonal().add_(1.0)
        _smallside_inverse(
            buf,
            stats,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
            out=inv,
        )
        inv.mul_(beta0)
        inv.diagonal().add_(alpha0)
        m_acc.copy_(inv)

        torch.mm(delta, inv, out=buf)
        torch.mm(inv, buf, out=delta)
        delta.mul_(delta_scale)
        _symmetrize_(delta, scratch)

        # Step 1
        buf.copy_(delta)
        buf.diagonal().add_(1.0)
        _smallside_inverse(
            buf,
            stats,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
            out=inv,
        )
        inv.mul_(beta1)
        inv.diagonal().add_(alpha1)
        torch.mm(m_acc, inv, out=buf)
        m_acc.copy_(buf)

    x = x @ m_acc
    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x, stats=stats)


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    mode: str = "rectangular",
    robust: bool = False,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
) -> PolarResult:
    """Two-step DWH with pluggable rectangular vs small-side updates."""
    assert a.ndim == 2
    if mode == "rectangular":
        return _dwh2_rectangular(
            a,
            ell0=ell0,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
        )
    if mode == "smallside_delta":
        return _dwh2_smallside_delta(
            a,
            ell0=ell0,
            robust=robust,
            scaled_jitter_scale=scaled_jitter_scale,
            diag_floor_rel=diag_floor_rel,
        )
    raise ValueError(f"unknown dwh2 mode {mode}")
