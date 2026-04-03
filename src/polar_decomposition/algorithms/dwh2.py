from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

PAPER_MUON_ELL = 1e-3
PAPER_NORM_EPS = 1e-3

from ..utils.precond import CholStats, PolarResult, _inverse_from_cholesky


_SMALLSIDE_GRAM_BLOCK_ROWS = 1024


@dataclass(frozen=True)
class DWHStepParams:
    a: float
    b: float
    c: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class DWH2Params:
    ell0: float
    ell1: float
    delta: float
    step0: DWHStepParams
    step1: DWHStepParams


def _dwh_coefficients_fp64(ell: float) -> tuple[float, float, float, float]:
    ell = float(max(min(ell, 1.0), 1e-12))
    gamma = (4.0 * (1.0 - ell * ell) / (ell**4)) ** (1.0 / 3.0)
    root = math.sqrt(1.0 + gamma)
    a = root + 0.5 * math.sqrt(
        8.0 - 4.0 * gamma + 8.0 * (2.0 - ell * ell) / (ell * ell * root)
    )
    b = ((a - 1.0) ** 2) / 4.0
    c = a + b - 1.0
    ell_next = ell * (a + b * ell * ell) / (1.0 + c * ell * ell)
    return float(a), float(b), float(c), float(ell_next)


def dwh_coefficients(ell: float) -> tuple[float, float, float, float]:
    return _dwh_coefficients_fp64(ell)



@lru_cache(maxsize=64)
def get_dwh2_params(ell0: float) -> DWH2Params:
    ell0 = float(ell0)
    a0, b0, c0, ell1 = _dwh_coefficients_fp64(ell0)
    a1, b1, c1, _ = _dwh_coefficients_fp64(ell1)

    step0 = DWHStepParams(
        a=a0,
        b=b0,
        c=c0,
        alpha=b0 / c0,
        beta=a0 - (b0 / c0),
    )
    step1 = DWHStepParams(
        a=a1,
        b=b1,
        c=c1,
        alpha=b1 / c1,
        beta=a1 - (b1 / c1),
    )
    return DWH2Params(
        ell0=ell0,
        ell1=ell1,
        delta=c1 / c0,
        step0=step0,
        step1=step1,
    )


# Import-time precompute for the default Muon paper ell.
DWH2_PAPER_PARAMS = get_dwh2_params(float(PAPER_MUON_ELL))


def emit_precomputed_dwh2_literal(ell0: float = PAPER_MUON_ELL) -> str:
    """Utility: print a hardcoded fp64 literal if you want true offline constants."""
    p = get_dwh2_params(float(ell0))
    return f"""DWH2Params(
    ell0={p.ell0:.17g},
    ell1={p.ell1:.17g},
    delta={p.delta:.17g},
    step0=DWHStepParams(
        a={p.step0.a:.17g},
        b={p.step0.b:.17g},
        c={p.step0.c:.17g},
        alpha={p.step0.alpha:.17g},
        beta={p.step0.beta:.17g},
    ),
    step1=DWHStepParams(
        a={p.step1.a:.17g},
        b={p.step1.b:.17g},
        c={p.step1.c:.17g},
        alpha={p.step1.alpha:.17g},
        beta={p.step1.beta:.17g},
    ),
)"""


def _choose_large_matmul_dtype(
    x: torch.Tensor,
    requested: torch.dtype | None,
) -> torch.dtype:
    if requested is not None:
        return requested
    if not x.is_cuda:
        # CPU fallback: stay in at least fp32.
        if x.dtype in (torch.float32, torch.float64):
            return x.dtype
        return torch.float32
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.dtype
    # Fast default for CUDA float32 inputs.
    return torch.float16


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)
    return a


def _block_gram_tree_(
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int = _SMALLSIDE_GRAM_BLOCK_ROWS,
) -> torch.Tensor:
    """Balanced split-K Gram accumulation in out.dtype.

    This avoids materializing a full fp32 copy of x when the large matrix lives
    in fp16/bf16, while still accumulating the small-side Gram in fp32/fp64.
    """
    partials: list[torch.Tensor | None] = []
    for start in range(0, x.shape[0], block_rows):
        stop = min(start + block_rows, x.shape[0])
        xb = x[start:stop]
        if xb.dtype != out.dtype:
            xb = xb.to(out.dtype)
        carry = xb.mT @ xb
        level = 0
        while True:
            if level == len(partials):
                partials.append(carry)
                break
            prev = partials[level]
            if prev is None:
                partials[level] = carry
                break
            prev.add_(carry)
            partials[level] = None
            carry = prev
            level += 1

    out.zero_()
    for part in partials:
        if part is not None:
            out.add_(part)
    return out


def _left_spd_solve_from_cholesky(
    l: torch.Tensor,
    rhs: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    sol = torch.empty_like(rhs) if out is None else out
    if sol.data_ptr() != rhs.data_ptr():
        sol.copy_(rhs)
    torch.linalg.solve_triangular(l, sol, upper=False, left=True, out=sol)
    torch.linalg.solve_triangular(l.mT, sol, upper=True, left=True, out=sol)
    return sol


def _factor_spd_shift_consistent(
    a: torch.Tensor,
    stats: CholStats,
    *,
    scratch: torch.Tensor,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    max_retries: int = 5,
) -> torch.Tensor:
    """Factor an SPD matrix in place, keeping the applied shift inside `a`.

    This is the key consistency rule for the rational state:
    whatever diagonal shift was actually needed must remain part of the matrix
    that defines the subsequent algebra.
    """
    n = a.shape[0]
    _symmetrize_(a, scratch)

    diag_floored = 0
    if diag_floor_rel > 0.0:
        diag = a.diagonal()
        tiny = torch.finfo(a.dtype).tiny
        floor = torch.mean(diag.abs()).clamp_min(tiny) * diag_floor_rel
        diag_floored = int(torch.count_nonzero(diag < floor).item())
        if diag_floored:
            diag.clamp_min_(floor)

    u = float(torch.finfo(a.dtype).eps)
    jitter = max(scaled_jitter_scale * n * u, 1e-7 if a.dtype == torch.float32 else 0.0)

    if jitter > 0.0:
        a.diagonal().add_(jitter)

    l, info = torch.linalg.cholesky_ex(a, check_errors=False)
    retries = 0
    extra = jitter

    while int(info.item()) != 0 and retries < max_retries:
        extra = max(extra * 10.0, 1e-6 if a.dtype == torch.float32 else 1e-12)
        a.diagonal().add_(extra)
        jitter += extra
        l, info = torch.linalg.cholesky_ex(a, check_errors=False)
        retries += 1

    if int(info.item()) != 0:
        # Last resort. Keep the extra shift in the matrix so downstream state
        # remains consistent with the factorization that was actually used.
        extra = max(extra * 10.0, 1e-2 if a.dtype == torch.float32 else 1e-8)
        a.diagonal().add_(extra)
        jitter += extra
        l = torch.linalg.cholesky(a)
        retries += 1

    stats.update(
        shifted=jitter > 0.0,
        retries=retries,
        jitter=float(jitter),
        diag_floored=diag_floored,
    )
    return l


def _dwh2_core(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    gram_0: torch.Tensor | None = None,
    large_matmul_dtype: torch.dtype | None = None,
) -> PolarResult:
    """Stable 2-step Gram-side DWH with 2 rectangular multiplies.

    Main idea:
      1. Build only the initial small-side Gram G0.
      2. Form H0 = (I + c0 G0)^(-1) in fp32.
      3. Drive the second step through
             S1 = H0 + delta * (I - H0) * M0^2,
         where M0 = alpha0 I + beta0 H0.
         This keeps the second denominator in bounded matrices and avoids
         explicitly building the dangerous large-scaled A1 = I + c1 G1.
      4. Final correction:
             K = alpha1 M0 + beta1 S1^(-1) (M0 H0)
         and apply X -> X K once.

    Assumes the caller already normalized `a` the same way the PE5 / Muon path
    does, so the initial singular values lie in the design interval.
    """
    assert a.ndim == 2

    transposed = a.shape[0] < a.shape[1]
    x = a.mT.contiguous() if transposed else a.contiguous()

    large_dtype = _choose_large_matmul_dtype(x, large_matmul_dtype)
    small_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

    m, n = x.shape
    _ = m  # quiet lint

    stats = CholStats()

    # fp64 parameterization, cached once.
    params = DWH2_PAPER_PARAMS if float(ell0) == float(PAPER_MUON_ELL) else get_dwh2_params(float(ell0))
    s0 = params.step0
    s1 = params.step1
    delta = params.delta

    # Large matrix for the final rectangular apply.
    x_big = x if x.dtype == large_dtype else x.to(large_dtype)

    # Small-side work buffers.
    gram = torch.empty((n, n), device=x.device, dtype=small_dtype)
    scratch = torch.empty((n, n), device=x.device, dtype=small_dtype)
    h0 = torch.empty((n, n), device=x.device, dtype=small_dtype)
    h0_sq = torch.empty((n, n), device=x.device, dtype=small_dtype)
    m0 = torch.empty((n, n), device=x.device, dtype=small_dtype)
    m0_sq = torch.empty((n, n), device=x.device, dtype=small_dtype)
    mh0 = torch.empty((n, n), device=x.device, dtype=small_dtype)
    s1_mat = torch.empty((n, n), device=x.device, dtype=small_dtype)
    rhs = torch.empty((n, n), device=x.device, dtype=small_dtype)
    z = torch.empty((n, n), device=x.device, dtype=small_dtype)
    k = torch.empty((n, n), device=x.device, dtype=small_dtype)

    # 1st rectangular multiply: initial Gram.
    if gram_0 is not None:
        gram.copy_(gram_0.to(device=x.device, dtype=small_dtype))
        _symmetrize_(gram, scratch)
    else:
        _block_gram_tree_(x, gram)

    # A0 = I + c0 G0. Factor in place, keeping whatever shift was actually used.
    gram.mul_(s0.c)
    gram.diagonal().add_(1.0)
    l0 = _factor_spd_shift_consistent(
        gram,
        stats,
        scratch=scratch,
        scaled_jitter_scale=scaled_jitter_scale,
        diag_floor_rel=diag_floor_rel,
    )

    # H0 = A0^{-1}
    _inverse_from_cholesky(l0, out=h0)
    _symmetrize_(h0, scratch)

    # H0^2
    torch.mm(h0, h0, out=h0_sq)
    _symmetrize_(h0_sq, scratch)

    # M0 = alpha0 I + beta0 H0
    m0.copy_(h0)
    m0.mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    _symmetrize_(m0, scratch)

    # M0 H0 = alpha0 H0 + beta0 H0^2
    mh0.copy_(h0_sq)
    mh0.mul_(s0.beta)
    mh0.add_(h0, alpha=s0.alpha)
    _symmetrize_(mh0, scratch)

    # M0^2 = alpha0^2 I + 2 alpha0 beta0 H0 + beta0^2 H0^2
    m0_sq.copy_(h0_sq)
    m0_sq.mul_(s0.beta * s0.beta)
    m0_sq.add_(h0, alpha=2.0 * s0.alpha * s0.beta)
    m0_sq.diagonal().add_(s0.alpha * s0.alpha)
    _symmetrize_(m0_sq, scratch)

    # S1 = H0 + delta * (I - H0) * M0^2
    # We explicitly parametrize the second step through bounded matrices only.
    s1_mat.copy_(h0)
    rhs.copy_(h0)
    rhs.mul_(-1.0)
    rhs.diagonal().add_(1.0)  # rhs = I - H0
    torch.mm(rhs, m0_sq, out=z)
    z.mul_(delta)
    s1_mat.add_(z)
    _symmetrize_(s1_mat, scratch)

    # Factor the bounded second-step SPD denominator.
    l1 = _factor_spd_shift_consistent(
        s1_mat,
        stats,
        scratch=scratch,
        scaled_jitter_scale=scaled_jitter_scale,
        diag_floor_rel=diag_floor_rel,
    )

    # Z = S1^{-1} (M0 H0)
    _left_spd_solve_from_cholesky(l1, mh0, out=z)
    _symmetrize_(z, scratch)

    # Final correction:
    #   K = alpha1 M0 + beta1 S1^{-1} (M0 H0)
    k.copy_(m0)
    k.mul_(s1.alpha)
    k.add_(z, alpha=s1.beta)
    _symmetrize_(k, scratch)

    # 2nd rectangular multiply: final apply.
    y = x_big @ k.to(dtype=x_big.dtype)
    if y.dtype != a.dtype:
        y = y.to(a.dtype)
    if transposed:
        y = y.mT.contiguous()
    return PolarResult(q=y, stats=stats)


def dwh2(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    scaled_jitter_scale: float = 2.0,
    diag_floor_rel: float = 0.0,
    gram_0: torch.Tensor | None = None,
    large_matmul_dtype: torch.dtype | None = None,
) -> PolarResult:
    """Drop-in replacement for the current DWH2 path."""
    return _dwh2_core(
        a,
        ell0=ell0,
        scaled_jitter_scale=scaled_jitter_scale,
        diag_floor_rel=diag_floor_rel,
        gram_0=gram_0,
        large_matmul_dtype=large_matmul_dtype,
    )
