from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizationInfo:
    method: str
    raw_scale: float
    scale: float
    ridge: float = 0.0
    ridge_stat: str = "none"


def _tall_view(a: torch.Tensor) -> torch.Tensor:
    return a if a.shape[0] >= a.shape[1] else a.mT


def fro_scale(a: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(a, ord="fro").item())


def schatten4_scale(
    a: torch.Tensor,
    *,
    ridge_scale: float = 0.0,
    ridge_stat: str = "max",
) -> NormalizationInfo:
    """Estimate ||A||_{S4} via the small-side Gram.

    For a tall view X of A, ||A||_{S4} = ||X^T X||_F^{1/2}. Optional ridging
    adds O(k * eps) times a Gram-diagonal statistic to the Gram diagonal, where
    k is the inner product length of the Gram build. Using the maximum diagonal
    entry was the most conservative low-cost recipe in the TF32 sweeps.
    """
    x = _tall_view(a)
    gram = x.mT @ x
    fro_sq = torch.sum(torch.square(gram))
    raw_scale = float(torch.sqrt(torch.sqrt(fro_sq)).item())

    ridge = 0.0
    trace = None
    if ridge_scale > 0.0:
        diag = gram.diagonal()
        if ridge_stat == "mean":
            diag_stat = torch.mean(diag)
        elif ridge_stat == "max":
            diag_stat = torch.max(diag)
        else:
            raise ValueError(f"unknown schatten4 ridge statistic {ridge_stat}")
        diag_stat = diag_stat.clamp_min(torch.finfo(gram.dtype).tiny)
        ridge = float(
            (ridge_scale * x.shape[0] * torch.finfo(gram.dtype).eps * diag_stat).item()
        )
        trace = torch.sum(diag)

    if ridge == 0.0:
        scale = raw_scale
    else:
        n = gram.shape[0]
        ridged_fro_sq = fro_sq + 2.0 * ridge * trace + n * (ridge**2)
        scale = float(torch.sqrt(torch.sqrt(ridged_fro_sq)).item())
    return NormalizationInfo(
        method="schatten4",
        raw_scale=raw_scale,
        scale=scale,
        ridge=ridge,
        ridge_stat="none" if ridge == 0.0 else ridge_stat,
    )


def estimate_normalization(
    a: torch.Tensor,
    *,
    method: str,
    schatten4_ridge_scale: float = 0.0,
    schatten4_ridge_stat: str = "max",
) -> NormalizationInfo:
    if method == "fro":
        scale = fro_scale(a)
        return NormalizationInfo(method="fro", raw_scale=scale, scale=scale)
    if method == "schatten4":
        return schatten4_scale(
            a,
            ridge_scale=schatten4_ridge_scale,
            ridge_stat=schatten4_ridge_stat,
        )
    raise ValueError(f"unknown normalization method {method}")


def normalize_matrix(
    a: torch.Tensor,
    *,
    method: str,
    eps: float,
    schatten4_ridge_scale: float = 0.0,
    schatten4_ridge_stat: str = "max",
) -> tuple[torch.Tensor, NormalizationInfo]:
    info = estimate_normalization(
        a,
        method=method,
        schatten4_ridge_scale=schatten4_ridge_scale,
        schatten4_ridge_stat=schatten4_ridge_stat,
    )
    return a / (info.scale + eps), info
