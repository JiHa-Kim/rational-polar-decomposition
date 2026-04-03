from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizationInfo:
    method: str
    raw_scale: float
    scale: float
    gram: torch.Tensor | None = None


def _tall_view(a: torch.Tensor) -> torch.Tensor:
    return a if a.shape[0] >= a.shape[1] else a.mT


def _uses_tf32_matmul(x: torch.Tensor) -> bool:
    return bool(
        x.is_cuda
        and x.dtype == torch.float32
        and torch.backends.cuda.matmul.allow_tf32
        and torch.get_float32_matmul_precision() != "highest"
    )


def _gamma(length: int, unit_roundoff: float) -> float:
    prod = float(length) * unit_roundoff
    if prod >= 1.0:
        return math.inf
    return prod / (1.0 - prod)


def _gram_error_envelope(x: torch.Tensor) -> float:
    """Conservative Frobenius-norm envelope for the computed small-side Gram.

    For G = X^T X built from length-m dot products, model the GEMM error as

        |G_hat - G| <= eta * d d^T,   d_i = ||x_i||_2,

    so ||G_hat - G||_F <= eta ||X||_F^2. Under TF32 tensor-core GEMMs the
    dominant extra term is the input rounding before multiply; otherwise this
    reduces to the usual accumulation envelope.
    """
    u_acc = torch.finfo(x.dtype).eps / 2.0
    eta = _gamma(x.shape[0], u_acc)
    if _uses_tf32_matmul(x):
        eta += 2.0 ** -10
    return eta


def _moment_lambda_upper(
    t1: torch.Tensor,
    t2: torch.Tensor,
    n: int,
) -> torch.Tensor:
    radicand = torch.clamp_min((n * t2) - (t1 * t1), 0.0)
    return (t1 + torch.sqrt((n - 1) * radicand)) / n


def spectral_additive_scale(
    a: torch.Tensor,
    *,
    gram: torch.Tensor | None = None,
) -> NormalizationInfo:
    """One-sided spectral upper bound via additive Gram error envelope.

    First upper-bound the computed Gram's spectral radius with the same PSD
    two-moment bound, then add the TF32/FP32 Gram error envelope directly at the
    eigenvalue level:

        lambda_max(G) <= lambda_max(G_hat) + ||G_hat - G||_2
                       <= lambda_hat_ub + eta ||X||_F^2.

    This is usually less conservative than inflating tr(G^2) directly.
    """
    x = _tall_view(a)
    if gram is None:
        gram = x.mT @ x

    gram64 = gram.to(torch.float64)
    t1_hat = torch.trace(gram64)
    gram_fro = torch.linalg.matrix_norm(gram64, ord="fro")
    t2_hat = gram_fro * gram_fro
    raw_lambda = _moment_lambda_upper(t1_hat, t2_hat, x.shape[1])
    raw_scale = float(torch.sqrt(raw_lambda).item())

    t1 = torch.sum(torch.square(x), dtype=torch.float64)
    eta = _gram_error_envelope(x)
    ub_lambda = raw_lambda + (eta * t1)
    scale = float(torch.sqrt(torch.clamp_min(ub_lambda, 0.0)).item())
    return NormalizationInfo(
        method="spectral_additive",
        raw_scale=raw_scale,
        scale=scale,
        gram=gram,
    )


def estimate_normalization(
    a: torch.Tensor,
    *,
    gram: torch.Tensor | None = None,
) -> NormalizationInfo:
    return spectral_additive_scale(a, gram=gram)


def normalize_matrix(
    a: torch.Tensor,
    *,
    eps: float,
    gram: torch.Tensor | None = None,
) -> tuple[torch.Tensor, NormalizationInfo]:
    info = estimate_normalization(a, gram=gram)
    return a / (info.scale + eps), info
