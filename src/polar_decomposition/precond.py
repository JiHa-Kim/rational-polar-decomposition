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


def _form_u(dtype: torch.dtype, tf32: bool) -> float:
    del tf32
    return float(torch.finfo(dtype).eps)


def _scale_and_symmetrize(a: torch.Tensor, diag_floor_rel: float) -> torch.Tensor:
    """Unit-diagonal scaling + exact symmetrization. Returns scale vector."""
    diag = a.diagonal()
    if diag_floor_rel > 0.0:
        tiny = torch.finfo(a.dtype).tiny
        floor = torch.mean(diag.abs()).clamp_min(tiny) * diag_floor_rel
        diag.clamp_min_(floor)
    s = torch.rsqrt(diag)
    a.mul_(s[:, None]).mul_(s[None, :])
    # Two sequential in-place muls can break symmetry; force it back.
    a = 0.5 * (a + a.mT)
    return a, s


def spd_inverse_fast(
    a: torch.Tensor,
    stats: CholStats,
    *,
    tf32: bool,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compile-friendly SPD inverse with unconditional jitter."""
    n = a.shape[0]
    a, s = _scale_and_symmetrize(a, diag_floor_rel)

    # Scale jitter by matrix dimension — O(n*eps) is the expected
    # rounding error in the Gram product for n-wide inner products.
    u = _form_u(a.dtype, tf32)
    jitter = max(scaled_jitter_scale * n * u, 0.0)
    if jitter > 0.0:
        a.diagonal().add_(jitter)

    l = torch.linalg.cholesky(a)
    inv_a = torch.cholesky_inverse(l, upper=False, out=out)
    inv_a.mul_(s[:, None]).mul_(s[None, :])

    stats.update(shifted=jitter > 0.0, retries=0, jitter=jitter, diag_floored=0)
    return inv_a


def spd_inverse_safe(
    a: torch.Tensor,
    stats: CholStats,
    *,
    tf32: bool,
    diag_floor_rel: float = 1e-6,
    base_shift_scale: float = 2.0,
    max_retries: int = 6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Defensive SPD inverse with retries for stress testing."""
    n = a.shape[0]
    a, s = _scale_and_symmetrize(a, diag_floor_rel)

    base_shift = max(base_shift_scale * n * _form_u(a.dtype, tf32), 1e-7)
    shifted = False
    retries = 0
    jitter = 0.0
    shifted_a = torch.empty_like(a)

    l, info = torch.linalg.cholesky_ex(a, check_errors=False)
    while int(info.item()) != 0 and retries < max_retries:
        shifted = True
        jitter = base_shift * (10.0**retries)
        shifted_a.copy_(a)
        shifted_a.diagonal().add_(jitter)
        l, info = torch.linalg.cholesky_ex(shifted_a, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        shifted = True
        jitter = base_shift * (10.0**retries)
        shifted_a.copy_(a)
        shifted_a.diagonal().add_(jitter)
        l = torch.linalg.cholesky(shifted_a)
        retries += 1

    inv_a = torch.cholesky_inverse(l, upper=False, out=out)
    inv_a.mul_(s[:, None]).mul_(s[None, :])

    stats.update(shifted=shifted, retries=retries, jitter=jitter, diag_floored=0)
    return inv_a
