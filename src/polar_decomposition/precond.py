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


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.mT)


def _default_form_u(dtype: torch.dtype, tf32: bool) -> float:
    if dtype == torch.float32 and tf32 and torch.cuda.is_available():
        return 2.0**-10
    return float(torch.finfo(dtype).eps)


def spd_cholesky_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    stats: CholStats,
    *,
    tf32: bool,
    diag_floor_rel: float = 1e-6,
    base_shift_scale: float = 2.0,
    max_retries: int = 4,
) -> torch.Tensor:
    """Solve A X = B for symmetric positive definite A using scaled-space Cholesky.

    Steps:
      1. Symmetrize A.
      2. Lightly floor the diagonal relative to its mean.
      3. Symmetric unit-diagonal scaling.
      4. Unshifted cholesky_ex first, then retry in scaled space with geometric jitter.
    """
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    assert b.ndim == 2 and b.shape[0] == a.shape[0]

    dtype = a.dtype

    a = _symmetrize(a)
    diag = torch.diagonal(a)
    mean_diag = torch.mean(diag).clamp_min_(torch.finfo(dtype).tiny)
    diag_floor = diag_floor_rel * mean_diag
    diag.clamp_min_(diag_floor)

    s = torch.rsqrt(diag)
    a.mul_(s[:, None]).mul_(s[None, :])
    h = a

    rhs = s[:, None] * b

    form_u = _default_form_u(dtype, tf32)
    base_shift = max(base_shift_scale * form_u, 1e-7)

    retries = 0
    jitter = 0.0
    shifted = False

    l, info = torch.linalg.cholesky_ex(h, check_errors=False)
    while int(info.item()) != 0 and retries < max_retries:
        shifted = True
        jitter = base_shift * (10.0**retries)
        h_retry = h.clone()
        h_retry.diagonal().add_(jitter)
        l, info = torch.linalg.cholesky_ex(h_retry, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        jitter = base_shift * (10.0**retries)
        h_retry = h.clone()
        h_retry.diagonal().add_(jitter)
        l = torch.linalg.cholesky(h_retry)
        shifted = True
        retries += 1

    torch.cholesky_solve(rhs, l, upper=False, out=rhs)
    rhs.mul_(s[:, None])
    stats.update(shifted=shifted, retries=retries, jitter=jitter, diag_floored=0)
    return rhs
