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


def centered_coefficients(
    coeffs: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    return [(a + b + c, -(b + 2.0 * c), c) for (a, b, c) in coeffs]


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.mT)


def _apply_block(
    x: torch.Tensor,
    out: torch.Tensor,
    coeffs: Sequence[Tuple[float, float, float]],
    *,
    first_block: bool,
    first_gram_jitter: float,
    symmetrize_inputs: bool,
) -> torch.Tensor:
    n = x.shape[1]
    y = x.mT @ x
    if symmetrize_inputs:
        y = _symmetrize(y)
    if first_block and first_gram_jitter:
        y.diagonal().add_(first_gram_jitter)

    q = torch.eye(n, device=x.device, dtype=x.dtype)
    e = torch.empty_like(y)
    e2 = torch.empty_like(y)
    h = torch.empty_like(y)
    temp = torch.empty_like(y)
    q_new = torch.empty_like(y)

    for h0, h1, h2 in coeffs:
        torch.neg(y, out=e)
        e.diagonal().add_(1.0)
        torch.mm(e, e, out=e2)

        torch.mul(e2, h2, out=h)
        h.add_(e, alpha=h1)
        h.diagonal().add_(h0)

        if symmetrize_inputs:
            h = _symmetrize(h)

        torch.mm(q, h, out=q_new)
        q, q_new = q_new, q

        torch.mm(y, h, out=temp)
        torch.mm(temp, h, out=y)

        if symmetrize_inputs:
            y = _symmetrize(y)

    return torch.mm(x, q, out=out)


def pe5(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    restart_interval: int = 3,
    coeffs: Sequence[Tuple[float, float, float]] | None = None,
    first_gram_jitter: float = PAPER_FIRST_GRAM_JITTER,
    symmetrize_inputs: bool = False,
) -> PE5Result:
    """Five-step Polar Express with a leaner GPU-oriented inner loop.

    The baseline implementation symmetrized several small-side matrices every
    step using clone-heavy bandwidth passes. That is good for debugging but often
    not worth the cost on the main latency path. Here it is opt-in instead.
    """
    assert a.ndim == 2
    transposed = False
    x = a.clone()
    if x.shape[0] < x.shape[1]:
        x = x.mT.contiguous()
        transposed = True

    centered = centered_coefficients(coeffs or pe5_coefficients(ell0=ell0, steps=5))
    x_buffer = torch.empty_like(x)
    for i in range(0, len(centered), restart_interval):
        block = centered[i : i + restart_interval]
        x_buffer = _apply_block(
            x,
            x_buffer,
            block,
            first_block=(i == 0),
            first_gram_jitter=first_gram_jitter,
            symmetrize_inputs=symmetrize_inputs,
        )
        x, x_buffer = x_buffer, x

    if transposed:
        x = x.mT.contiguous()
    return PE5Result(q=x)
