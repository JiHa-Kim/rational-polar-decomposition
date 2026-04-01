from __future__ import annotations

from collections.abc import Sequence
from math import inf, sqrt

import numpy as np
import torch

from .precond import PolarResult


PAPER_MUON_ELL = 1e-3
PAPER_CUSHION = 0.02407327424182761
PAPER_SAFETY = 1.01
PAPER_FIRST_GRAM_JITTER = 0.0
PAPER_NORM_EPS = 1e-3


def _optimal_quintic(l: float, u: float = 1.0) -> tuple[float, float, float]:
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
) -> list[tuple[float, float, float]]:
    l = float(ell0)
    u = 1.0
    coeffs: list[tuple[float, float, float]] = []
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
    coeffs: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    return [(a + b + c, -(b + 2.0 * c), c) for (a, b, c) in coeffs]


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.mT)


def _apply_block(
    x: torch.Tensor,
    out: torch.Tensor,
    coeffs: Sequence[tuple[float, float, float]],
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
    buf_a = torch.empty_like(y)  # e, then reused as temp
    buf_b = torch.empty_like(y)  # e2, then becomes h in-place
    q_new = torch.empty_like(y)

    for h0, h1, h2 in coeffs:
        # buf_a = I - Y
        torch.neg(y, out=buf_a)
        buf_a.diagonal().add_(1.0)

        # buf_b = (I - Y)^2
        torch.mm(buf_a, buf_a, out=buf_b)

        # h = h0*I + h1*(I-Y) + h2*(I-Y)^2  — computed in-place in buf_b
        buf_b.mul_(h2)
        buf_b.add_(buf_a, alpha=h1)
        buf_b.diagonal().add_(h0)
        # buf_a is now dead; buf_b holds h

        if symmetrize_inputs:
            buf_b = _symmetrize(buf_b)

        torch.mm(q, buf_b, out=q_new)
        q, q_new = q_new, q

        # Y_new = h @ Y @ h  (= Y @ h^2 since h and Y commute)
        torch.mm(y, buf_b, out=buf_a)  # reuse buf_a as temp
        torch.mm(buf_a, buf_b, out=y)

        if symmetrize_inputs:
            y = _symmetrize(y)

    return torch.mm(x, q, out=out)


def pe5(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    restart_interval: int = 3,
    coeffs: Sequence[tuple[float, float, float]] | None = None,
    first_gram_jitter: float = PAPER_FIRST_GRAM_JITTER,
    symmetrize_inputs: bool = False,
) -> PolarResult:
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
    return PolarResult(q=x)
