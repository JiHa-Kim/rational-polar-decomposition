from __future__ import annotations

from dataclasses import dataclass

import torch


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


def _default_form_u(dtype: torch.dtype, tf32: bool) -> float:
    if dtype == torch.float32 and tf32 and torch.cuda.is_available():
        return 2.0**-10
    return float(torch.finfo(dtype).eps)




def spd_cholesky_solve_fast(
    a: torch.Tensor,
    b: torch.Tensor,
    stats: CholStats,
    *,
    tf32: bool,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast compile-friendly SPD solve for hot paths.

    This path assumes the input is already numerically close to SPD. It avoids
    Python-side error inspection and retry loops so `torch.compile` can keep the
    entire region in a single graph. A tiny unconditional diagonal jitter in the
    scaled space buys robustness without a graph break.
    """
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    assert b.ndim == 2 and b.shape[0] == a.shape[0]

    diag = a.diagonal()
    tiny = torch.finfo(a.dtype).tiny
    mean_diag = torch.mean(diag).clamp_min(tiny)
    if diag_floor_rel > 0.0:
        diag_floor = mean_diag * diag_floor_rel
        diag.clamp_min_(diag_floor)

    s = torch.rsqrt(diag)
    a.mul_(s[:, None]).mul_(s[None, :])

    jitter = max(scaled_jitter_scale * _default_form_u(a.dtype, tf32), 0.0)
    if jitter > 0.0:
        a.diagonal().add_(jitter)

    l = torch.linalg.cholesky(a)
    inv_a = torch.cholesky_inverse(l, upper=False)

    inv_a.mul_(s[:, None]).mul_(s[None, :])
    rhs = torch.mm(inv_a, b, out=out)

    stats.update(shifted=jitter > 0.0, retries=0, jitter=jitter, diag_floored=0)
    return rhs


def spd_cholesky_solve_safe(
    a: torch.Tensor,
    b: torch.Tensor,
    stats: CholStats,
    *,
    tf32: bool,
    diag_floor_rel: float = 1e-6,
    base_shift_scale: float = 2.0,
    max_retries: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Defensive SPD solve with retries.

    Keep this path for stress testing and numerical debugging. It is intentionally
    slower than `spd_cholesky_solve_fast` because it pays for Python-side error
    inspection and retry logic.
    """
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    assert b.ndim == 2 and b.shape[0] == a.shape[0]

    a = 0.5 * (a + a.mT)
    diag = a.diagonal()
    tiny = torch.finfo(a.dtype).tiny
    mean_diag = torch.mean(diag).clamp_min(tiny)
    diag_floor = mean_diag * diag_floor_rel
    diag.clamp_min_(diag_floor)

    s = torch.rsqrt(diag)
    a.mul_(s[:, None]).mul_(s[None, :])

    base_shift = max(base_shift_scale * _default_form_u(a.dtype, tf32), 1e-7)
    shifted = False
    retries = 0
    jitter = 0.0

    l, info = torch.linalg.cholesky_ex(a, check_errors=False)
    while int(info.item()) != 0 and retries < max_retries:
        shifted = True
        jitter = base_shift * (10.0**retries)
        shifted_a = a.clone()
        shifted_a.diagonal().add_(jitter)
        l, info = torch.linalg.cholesky_ex(shifted_a, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        shifted = True
        jitter = base_shift * (10.0**retries)
        shifted_a = a.clone()
        shifted_a.diagonal().add_(jitter)
        l = torch.linalg.cholesky(shifted_a)
        retries += 1

    inv_a = torch.cholesky_inverse(l, upper=False)
    inv_a.mul_(s[:, None]).mul_(s[None, :])

    rhs = torch.mm(inv_a, b, out=out)

    stats.update(shifted=shifted, retries=retries, jitter=jitter, diag_floored=0)
    return rhs
