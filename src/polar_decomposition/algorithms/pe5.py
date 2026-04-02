from __future__ import annotations

from collections.abc import Sequence
from math import inf, sqrt

import numpy as np
import torch

from ..utils.precond import PolarResult


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


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    return a


def _apply_block(
    x: torch.Tensor,
    out: torch.Tensor,
    coeffs: Sequence[tuple[float, float, float]],
    *,
    first_block: bool,
    first_gram_jitter: float,
    symmetrize_inputs: bool,
    gram: torch.Tensor,
    delta: torch.Tensor,
    poly: torch.Tensor,
    q_acc: torch.Tensor,
    q_tmp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.mm(x.mT, x, out=gram)
    if symmetrize_inputs:
        _symmetrize_(gram, delta)
    if first_block and first_gram_jitter:
        gram.diagonal().add_(first_gram_jitter)

    q_acc.zero_()
    q_acc.diagonal().fill_(1.0)

    for h0, h1, h2 in coeffs:
        # delta = I - gram
        torch.neg(gram, out=delta)
        delta.diagonal().add_(1.0)

        # poly = (I - gram)^2
        torch.mm(delta, delta, out=poly)

        # poly = h0*I + h1*(I-gram) + h2*(I-gram)^2
        poly.mul_(h2)
        poly.add_(delta, alpha=h1)
        poly.diagonal().add_(h0)

        if symmetrize_inputs:
            _symmetrize_(poly, q_tmp)

        torch.mm(q_acc, poly, out=q_tmp)
        q_acc, q_tmp = q_tmp, q_acc

        # gram_new = poly @ gram @ poly (= gram @ poly^2 since they commute)
        torch.mm(gram, poly, out=delta)
        torch.mm(delta, poly, out=gram)

        if symmetrize_inputs:
            _symmetrize_(gram, q_tmp)

    return torch.mm(x, q_acc, out=out), q_acc, q_tmp


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
    transposed = a.shape[0] < a.shape[1]
    # The restart ping-pong uses the previous output buffer as the next input.
    # Keep the caller's tensor immutable across repeated invocations.
    x = a.mT.contiguous() if transposed else a.clone()

    centered = centered_coefficients(coeffs or pe5_coefficients(ell0=ell0, steps=5))
    n = x.shape[1]
    gram = torch.empty((n, n), device=x.device, dtype=x.dtype)
    delta = torch.empty_like(gram)
    poly = torch.empty_like(gram)
    q_acc = torch.empty_like(gram)
    q_tmp = torch.empty_like(gram)
    x_buffer = torch.empty_like(x)
    for i in range(0, len(centered), restart_interval):
        block = centered[i : i + restart_interval]
        x_buffer, q_acc, q_tmp = _apply_block(
            x,
            x_buffer,
            block,
            first_block=(i == 0),
            first_gram_jitter=first_gram_jitter,
            symmetrize_inputs=symmetrize_inputs,
            gram=gram,
            delta=delta,
            poly=poly,
            q_acc=q_acc,
            q_tmp=q_tmp,
        )
        x, x_buffer = x_buffer, x

    if transposed:
        x = x.mT.contiguous()
    return PolarResult(q=x)
