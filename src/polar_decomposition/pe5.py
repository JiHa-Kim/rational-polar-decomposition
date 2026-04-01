from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt
from typing import List, Sequence, Tuple

import numpy as np
import torch


PAPER_MUON_ELL = 1e-3
PAPER_CUSHION = 0.02407327424182761
PAPER_SAFETY = 1.01
PAPER_FIRST_GRAM_JITTER = 0.0
PAPER_NORM_EPS = 1e-3


@dataclass
class PE5Result:
    q: torch.Tensor


def _optimal_quintic(l: float, u: float = 1.0) -> Tuple[float, float, float]:
    assert 0.0 <= l <= u
    if 1.0 - 5e-6 <= l / u:
        return (15.0 / 8.0) / u, (-10.0 / 8.0) / (u**3), (3.0 / 8.0) / (u**5)

    q = (3.0 * l + u) / 4.0
    r = (l + 3.0 * u) / 4.0
    e = inf
    old_e = None
    while old_e is None or abs(old_e - e) > 1e-15:
        old_e = e
        lhs = np.array(
            [
                [l, l**3, l**5, 1.0],
                [q, q**3, q**5, -1.0],
                [r, r**3, r**5, 1.0],
                [u, u**3, u**5, -1.0],
            ],
            dtype=np.float64,
        )
        a, b, c, e = np.linalg.solve(lhs, np.ones(4, dtype=np.float64))
        roots = np.sqrt(
            (-3.0 * b + np.array([-1.0, 1.0]) * sqrt(9.0 * b * b - 20.0 * a * c))
            / (10.0 * c)
        )
        q, r = float(roots[0]), float(roots[1])
    return float(a), float(b), float(c)


def pe5_coefficients(
    *,
    ell0: float = PAPER_MUON_ELL,
    steps: int = 5,
    cushion: float = PAPER_CUSHION,
    safety: float = PAPER_SAFETY,
) -> List[Tuple[float, float, float]]:
    l = float(ell0)
    u = 1.0
    coeffs: List[Tuple[float, float, float]] = []
    for _ in range(steps):
        a, b, c = _optimal_quintic(max(l, cushion * u), u)
        pl = a * l + b * l**3 + c * l**5
        pu = a * u + b * u**3 + c * u**5
        rescale = 2.0 / (pl + pu)
        a *= rescale
        b *= rescale
        c *= rescale
        coeffs.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2.0 - l

    if safety != 1.0 and len(coeffs) > 1:
        scaled = []
        for i, (a, b, c) in enumerate(coeffs):
            if i < len(coeffs) - 1:
                scaled.append((a / safety, b / (safety**3), c / (safety**5)))
            else:
                scaled.append((a, b, c))
        coeffs = scaled
    return coeffs


def _eval_h_centered(
    y: torch.Tensor, eye: torch.Tensor, coeff: Tuple[float, float, float]
) -> torch.Tensor:
    a, b, c = coeff
    # h(z) = a + bz + cz^2, evaluated around z = 1 via E = I - Y.
    e = eye - y
    e2 = e @ e
    h0 = a + b + c
    h1 = -(b + 2.0 * c)
    h2 = c
    h = h0 * eye + h1 * e + h2 * e2
    return 0.5 * (h + h.mT)


def _apply_block(
    x: torch.Tensor,
    coeffs: Sequence[Tuple[float, float, float]],
    *,
    first_block: bool,
    first_gram_jitter: float,
) -> torch.Tensor:
    n = x.shape[1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    y = x.mT @ x
    y = 0.5 * (y + y.mT)
    if first_block and first_gram_jitter:
        y = y + first_gram_jitter * eye

    q = eye
    for coeff in coeffs:
        h = _eval_h_centered(y, eye, coeff)
        q = q @ h
        yh = y @ h
        y = yh @ h
        y = 0.5 * (y + y.mT)
    return x @ q


def pe5(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    restart_interval: int = 3,
    coeffs: Sequence[Tuple[float, float, float]] | None = None,
    first_gram_jitter: float = PAPER_FIRST_GRAM_JITTER,
) -> PE5Result:
    """Five-step Polar Express using the fast small-side formulation with restart-3.

    Input is assumed to already be normalized by a matrix-only scale. This function applies the
    paper's degree-5 offline coefficients with finite-precision modifications and the first-block
    Gram jitter described for the fast rectangular algorithm.
    """
    assert a.ndim == 2
    transposed = False
    x = a
    if x.shape[0] < x.shape[1]:
        x = x.mT.contiguous()
        transposed = True

    coeffs = list(coeffs or pe5_coefficients(ell0=ell0, steps=5))
    for i in range(0, len(coeffs), restart_interval):
        block = coeffs[i : i + restart_interval]
        x = _apply_block(
            x, block, first_block=(i == 0), first_gram_jitter=first_gram_jitter
        )

    if transposed:
        x = x.mT.contiguous()
    return PE5Result(q=x)
