from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .precond import CholStats, spd_inverse_fast, spd_inverse_safe


@dataclass
class DWH2Result:
    q: torch.Tensor
    ell_final: float
    stats: CholStats


def dwh_coefficients(ell: float) -> Tuple[float, float, float, float]:
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


def _dwh_schedule(
    ell0: float, steps: int = 2
) -> Tuple[List[Tuple[float, float, float]], float]:
    coeffs: List[Tuple[float, float, float]] = []
    ell = float(ell0)
    for _ in range(steps):
        a, b, c, ell = dwh_coefficients(ell)
        coeffs.append((a, b, c))
    return coeffs, ell


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = 1e-3,
    tf32: bool = True,
    robust: bool = False,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
) -> DWH2Result:
    """Two-step direct DWH iteration optimized for repeated GPU execution.

    Main changes versus the baseline:
      * small-side Gram is written into a persistent buffer via `out=`
      * the hot solve path is branch-free and compile-friendly by default
      * no Python scalar syncs in the default path
    """
    assert a.ndim == 2
    transposed = False
    x = a.clone()
    if x.shape[0] < x.shape[1]:
        x = x.mT.contiguous()
        transposed = True

    n = x.shape[1]
    m = x.shape[0]
    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    inv_gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    rhs_t = torch.empty((m, n), device=x.device, dtype=x.dtype)
    stats = CholStats()

    coeffs, ell_final = _dwh_schedule(ell0=ell0, steps=2)
    inverse_fn = spd_inverse_safe if robust else spd_inverse_fast

    for aa, bb, cc in coeffs:
        torch.mm(x.mT, x, out=gram)
        gram.mul_(cc)
        gram.diagonal().add_(1.0)

        alpha = bb / cc
        beta = aa - alpha
        inverse_fn(
            gram,
            stats,
            tf32=tf32,
            out=inv_gram,
            diag_floor_rel=(1e-6 if robust else diag_floor_rel),
            **({"scaled_jitter_scale": scaled_jitter_scale} if not robust else {}),
        )
        torch.mm(x, inv_gram, out=rhs_t)
        x.mul_(alpha).add_(rhs_t, alpha=beta)

    if transposed:
        x = x.mT.contiguous()
    return DWH2Result(q=x, ell_final=ell_final, stats=stats)
