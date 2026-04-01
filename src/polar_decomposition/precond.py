from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PolarResult:
    """Unified return type for all polar decomposition methods."""

    q: torch.Tensor
    stats: CholStats = field(default_factory=lambda: CholStats())


@dataclass
class CholStats:
    calls: int = 0
    shifted_calls: int = 0
    total_retries: int = 0
    max_jitter: float = 0.0
    diag_floored: int = 0

    def update(
        self, *, shifted: bool, retries: int, jitter: float, diag_floored: int
    ) -> None:
        self.calls += 1
        self.shifted_calls += int(shifted)
        self.total_retries += retries
        self.max_jitter = max(self.max_jitter, float(jitter))
        self.diag_floored += int(diag_floored)


def _form_u(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).eps)


def _scratch_like(a: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None or out.data_ptr() == a.data_ptr():
        return torch.empty_like(a)
    return out


def _scale_and_symmetrize(
    a: torch.Tensor,
    diag_floor_rel: float,
    *,
    scratch: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Unit-diagonal scaling + exact symmetrization."""
    diag = a.diagonal()
    diag_floored = 0
    if diag_floor_rel > 0.0:
        tiny = torch.finfo(a.dtype).tiny
        floor = torch.mean(diag.abs()).clamp_min(tiny) * diag_floor_rel
        diag_floored = int(torch.count_nonzero(diag < floor).item())
        diag.clamp_min_(floor)
    s = torch.rsqrt(diag)
    a.mul_(s[:, None]).mul_(s[None, :])
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    return s, diag_floored


def _inverse_from_cholesky(
    l: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Form L^{-T} L^{-1} via two triangular solves.

    On this GPU this is materially faster than torch.cholesky_inverse for the
    4k x 4k SPD blocks used by DWH2, while staying numerically equivalent for
    benchmark purposes.
    """
    inv_a = (
        torch.empty_like(l) if out is None or out.data_ptr() == l.data_ptr() else out
    )
    inv_a.zero_()
    inv_a.diagonal().fill_(1.0)
    torch.linalg.solve_triangular(l, inv_a, upper=False, left=True, out=inv_a)
    torch.linalg.solve_triangular(l.mT, inv_a, upper=True, left=True, out=inv_a)
    return inv_a


def spd_inverse_fast(
    a: torch.Tensor,
    stats: CholStats,
    *,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compile-friendly SPD inverse with unconditional jitter."""
    n = a.shape[0]
    scratch = _scratch_like(a, out)
    s, diag_floored = _scale_and_symmetrize(a, diag_floor_rel, scratch=scratch)

    # Scale jitter by matrix dimension — O(n*eps) is the expected
    # rounding error in the Gram product for n-wide inner products.
    u = _form_u(a.dtype)
    jitter = max(scaled_jitter_scale * n * u, 0.0)
    if jitter > 0.0:
        a.diagonal().add_(jitter)

    l = torch.linalg.cholesky(a)
    inv_a = _inverse_from_cholesky(l, out=out)
    inv_a.mul_(s[:, None]).mul_(s[None, :])

    stats.update(
        shifted=jitter > 0.0,
        retries=0,
        jitter=jitter,
        diag_floored=diag_floored,
    )
    return inv_a


def spd_inverse_safe(
    a: torch.Tensor,
    stats: CholStats,
    *,
    diag_floor_rel: float = 1e-6,
    base_shift_scale: float = 2.0,
    max_retries: int = 6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Defensive SPD inverse with retries for stress testing."""
    n = a.shape[0]
    scratch = _scratch_like(a, out)
    s, diag_floored = _scale_and_symmetrize(a, diag_floor_rel, scratch=scratch)

    base_shift = max(base_shift_scale * n * _form_u(a.dtype), 1e-7)
    shifted = False
    retries = 0
    jitter = 0.0

    l, info = torch.linalg.cholesky_ex(a, check_errors=False)
    while int(info.item()) != 0 and retries < max_retries:
        shifted = True
        jitter = base_shift * (10.0**retries)
        scratch.copy_(a)
        scratch.diagonal().add_(jitter)
        l, info = torch.linalg.cholesky_ex(scratch, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        shifted = True
        jitter = base_shift * (10.0**retries)
        scratch.copy_(a)
        scratch.diagonal().add_(jitter)
        l = torch.linalg.cholesky(scratch)
        retries += 1

    inv_a = _inverse_from_cholesky(l, out=out)
    inv_a.mul_(s[:, None]).mul_(s[None, :])

    stats.update(
        shifted=shifted,
        retries=retries,
        jitter=jitter,
        diag_floored=diag_floored,
    )
    return inv_a
