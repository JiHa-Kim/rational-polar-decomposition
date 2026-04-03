"""Normalization and SPD preconditioning helpers."""

from .normalization import (
    NormalizationInfo,
    estimate_normalization,
    fro_scale,
    normalize_matrix,
    spectral_additive_scale,
    spectral_bound_scale,
)
from .precond import CholStats, PolarResult, spd_inverse_fast, spd_inverse_safe

__all__ = [
    "CholStats",
    "NormalizationInfo",
    "PolarResult",
    "estimate_normalization",
    "fro_scale",
    "normalize_matrix",
    "spectral_additive_scale",
    "spectral_bound_scale",
    "spd_inverse_fast",
    "spd_inverse_safe",
]
