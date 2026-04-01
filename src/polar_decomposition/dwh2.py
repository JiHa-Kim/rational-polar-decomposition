from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch._dynamo

from .precond import CholStats, spd_cholesky_solve

torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class DWH2Result:
    q: torch.Tensor
    ell_final: float
    stats: CholStats


# Nakatsukasa-Bai-Gygi DWH scalar schedule.
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


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = 1e-3,
    tf32: bool = True,
) -> DWH2Result:
    """Two-step direct DWH iteration for tall matrices.

    Input is assumed to already be normalized by a matrix-only scale such as ||A||_F.
    The implementation is direct rectangular DWH, but every solve is done on the small side.
    """
    assert a.ndim == 2
    transposed = False
    x = a.clone()
    if x.shape[0] < x.shape[1]:
        x = x.mT.contiguous()
        transposed = True

    n = x.shape[1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    stats = CholStats()
    ell = float(ell0)

    for _ in range(2):
        aa, bb, cc, ell = dwh_coefficients(ell)
        s = torch.addmm(eye, x.mT, x, alpha=cc)
        # Affine-resolvent identity:
        # (aI + bG)(I + cG)^{-1} = (b/c)I + (a - b/c)(I + cG)^{-1}.
        alpha = bb / cc
        beta = aa - alpha
        solved_t = spd_cholesky_solve(s, x.mT, stats, tf32=tf32)
        x.mul_(alpha).add_(solved_t.mT, alpha=beta)

    if transposed:
        x = x.mT.contiguous()
    return DWH2Result(q=x, ell_final=ell, stats=stats)
