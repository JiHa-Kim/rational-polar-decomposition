from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CholStats:
    calls: int = 0
    shifted_calls: int = 0
    total_retries: int = 0
    max_jitter: float = 0.0

    def update(self, *, shifted: bool, retries: int, jitter: float) -> None:
        self.calls += 1
        self.shifted_calls += int(shifted)
        self.total_retries += int(retries)
        self.max_jitter = max(self.max_jitter, float(jitter))


@dataclass
class PolarResult:
    q: torch.Tensor
    stats: CholStats = field(default_factory=lambda: CholStats())
    theta: float = 1.0
    ell0: float = 1e-3
    retries: int = 0
    alpha_log: list[float] = field(default_factory=list)
    s_c_log: list[float] = field(default_factory=list)


PAPER_MUON_ELL = 1e-3
NORM_EPS = 1e-7
NORM_SAFETY = 1.01
GRAM_BLOCK_ROWS = 1024
_BACKTRACK_ALPHA_LIMIT = 10.0
_BACKTRACK_SHIFTED_ALPHA_SOFT_LIMIT = 2.0
_BACKTRACK_SOLVE_RESID_LIMIT = 2e-2
_BACKTRACK_SKEW_LIMIT = 1e-3
_STABLE_RANK_TRIGGER = 0.95
_ADAPTIVE_RIDGE_SCALE = 1e-6


@dataclass(frozen=True)
class StepParams:
    alpha: float
    beta: float
    c: float


@dataclass(frozen=True)
class DWH2Params:
    step0: StepParams
    step1: StepParams
    delta: float


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
    alpha = b / c
    beta = a - alpha
    return float(alpha), float(beta), float(c), float(ell_next)


def get_dwh2_params(ell0: float) -> DWH2Params:
    alpha0, beta0, c0, ell1 = _dwh_coefficients_fp64(float(ell0))
    alpha1, beta1, c1, _ = _dwh_coefficients_fp64(float(ell1))
    return DWH2Params(
        step0=StepParams(alpha=alpha0, beta=beta0, c=c0),
        step1=StepParams(alpha=alpha1, beta=beta1, c=c1),
        delta=c1 / c0,
    )


@dataclass
class DWH2Workspace:
    n: int
    device: torch.device
    out_dtype: torch.dtype
    block_rows: int

    gram: torch.Tensor
    buf: torch.Tensor
    scratch: torch.Tensor
    h0: torch.Tensor
    k0: torch.Tensor
    m0: torch.Tensor
    tmp: torch.Tensor
    rhs: torch.Tensor
    k_final: torch.Tensor
    linv: torch.Tensor
    sh: torch.Tensor
    invsh: torch.Tensor

    xbuf: torch.Tensor
    L: torch.Tensor
    info: torch.Tensor

    b_mat: torch.Tensor
    resid: torch.Tensor
    k_cast: Optional[torch.Tensor] = None

    @staticmethod
    def allocate(
        n: int,
        device: torch.device,
        out_dtype: torch.dtype,
        block_rows: int = GRAM_BLOCK_ROWS,
    ) -> "DWH2Workspace":
        def mat32() -> torch.Tensor:
            return torch.empty((n, n), device=device, dtype=torch.float32)

        def vec32() -> torch.Tensor:
            return torch.empty((n,), device=device, dtype=torch.float32)

        return DWH2Workspace(
            n=n,
            device=device,
            out_dtype=out_dtype,
            block_rows=block_rows,
            gram=mat32(),
            buf=mat32(),
            scratch=mat32(),
            h0=mat32(),
            k0=mat32(),
            m0=mat32(),
            tmp=mat32(),
            rhs=mat32(),
            k_final=mat32(),
            linv=mat32(),
            sh=vec32(),
            invsh=vec32(),
            xbuf=torch.empty((block_rows, n), device=device, dtype=torch.float32),
            L=mat32(),
            info=torch.empty((), device=device, dtype=torch.int32),
            b_mat=mat32(),
            resid=mat32(),
        )

    def ensure_k_cast(self) -> torch.Tensor:
        k_cast = self.k_cast
        if (
            k_cast is None
            or k_cast.shape != (self.n, self.n)
            or k_cast.device != self.device
            or k_cast.dtype != self.out_dtype
        ):
            k_cast = torch.empty(
                (self.n, self.n),
                device=self.device,
                dtype=self.out_dtype,
            )
            self.k_cast = k_cast
        return k_cast


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> None:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)


def _uses_tf32_matmul() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.backends.cuda.matmul.allow_tf32
        and torch.get_float32_matmul_precision() != "highest"
    )


def _gamma(length: int, unit_roundoff: float) -> float:
    prod = float(length) * unit_roundoff
    if prod >= 1.0:
        return math.inf
    return prod / (1.0 - prod)


def _update_chol_stats(
    stats: Optional[CholStats], *, shifted: bool, retries: int, jitter: float
) -> None:
    if stats is not None:
        stats.update(shifted=shifted, retries=retries, jitter=jitter)


def _mat_skew_rel(a: torch.Tensor, scratch: torch.Tensor) -> float:
    scratch.copy_(a.mT)
    scratch.sub_(a)
    numer = torch.linalg.matrix_norm(scratch)
    denom = torch.linalg.matrix_norm(a).clamp_min(1e-30)
    return float((numer / denom).item())


def _solve_amplification(x: torch.Tensor, b: torch.Tensor) -> float:
    return float((x.abs().max() / b.abs().max().clamp_min(1e-30)).item())


def _solve_residual_rel(
    a: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    resid: torch.Tensor,
) -> float:
    resid.copy_(b)
    resid.addmm_(a, x, alpha=-1.0)
    numer = torch.linalg.matrix_norm(resid)
    denom = torch.linalg.matrix_norm(b).clamp_min(1e-30)
    return float((numer / denom).item())


def _stable_rank(a: torch.Tensor) -> float:
    t1 = float(torch.sum(a.diagonal(), dtype=torch.float64).item())
    t2 = float(torch.sum(a * a, dtype=torch.float64).item())
    if t2 <= 0.0:
        return 0.0
    return (t1 * t1) / t2


def _adaptive_ridge_scale(out_dtype: torch.dtype) -> float:
    if out_dtype == torch.bfloat16:
        return 4.0 * _ADAPTIVE_RIDGE_SCALE
    if out_dtype == torch.float32:
        return 0.5 * _ADAPTIVE_RIDGE_SCALE
    return _ADAPTIVE_RIDGE_SCALE


def _adaptive_ridge_tensor(
    gram: torch.Tensor, n: int, out_dtype: torch.dtype
) -> torch.Tensor:
    trace64 = torch.sum(gram.diagonal(), dtype=torch.float64)
    fro2_64 = torch.sum(gram * gram, dtype=torch.float64)
    n64 = trace64.new_tensor(float(n))
    zero = trace64.new_zeros(())
    stable_rank = torch.where(fro2_64 > 0.0, (trace64 * trace64) / fro2_64, zero)
    ridge_frac = torch.clamp(1.0 - stable_rank / n64, min=0.0)
    ridge = torch.where(
        stable_rank < (_STABLE_RANK_TRIGGER * n64),
        (_adaptive_ridge_scale(out_dtype) * (trace64 / n64) * ridge_frac),
        zero,
    )
    return ridge.to(dtype=gram.dtype)


def _add_adaptive_ridge_tensor_(
    gram: torch.Tensor, n: int, out_dtype: torch.dtype
) -> None:
    gram.diagonal().add_(_adaptive_ridge_tensor(gram, n, out_dtype))


@torch.compiler.disable(recursive=False, reason="data-dependent cholesky retry loop")
def _chol_spd_inplace_ex(
    a: torch.Tensor,
    stats: Optional[CholStats],
    *,
    scratch: torch.Tensor,
    L_out: torch.Tensor,
    info_out: torch.Tensor,
    jitter_scale: float = 2.0,
    min_jitter: float = 1e-4,
    max_retries: int = 6,
) -> torch.Tensor:
    n = a.shape[0]
    _symmetrize_(a, scratch)

    torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
    if int(info_out.item()) == 0:
        _update_chol_stats(stats, shifted=False, retries=0, jitter=0.0)
        return L_out

    u = float(torch.finfo(a.dtype).eps)
    jitter = max(jitter_scale * n * u, min_jitter)
    a.diagonal().add_(jitter)

    torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
    if int(info_out.item()) == 0:
        _update_chol_stats(stats, shifted=True, retries=1, jitter=jitter)
        return L_out

    retries = 2
    step = jitter * 10.0
    while int(info_out.item()) != 0 and retries <= max_retries:
        a.diagonal().add_(step)
        jitter += step
        torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
        if int(info_out.item()) == 0:
            _update_chol_stats(stats, shifted=True, retries=retries, jitter=jitter)
            return L_out
        retries += 1
        step *= 10.0

    a.diagonal().add_(0.1)
    jitter += 0.1
    torch.linalg.cholesky(a, out=L_out)
    _update_chol_stats(stats, shifted=True, retries=retries, jitter=jitter)
    return L_out


def _tri_inv_lower_inplace(
    L: torch.Tensor, out: torch.Tensor, work: torch.Tensor, leaf: int = 512
) -> None:
    n = L.shape[0]
    if n <= leaf:
        out.zero_()
        out.diagonal().fill_(1.0)
        torch.linalg.solve_triangular(L, out, upper=False, left=True, out=out)
        return

    k = n // 2
    A = L[:k, :k]
    C = L[k:, :k]
    D = L[k:, k:]

    outA = out[:k, :k]
    outC = out[k:, :k]
    outD = out[k:, k:]

    workA = work[:k, :k]
    workC = work[k:, :k]
    workD = work[k:, k:]

    _tri_inv_lower_inplace(A, outA, workA, leaf=leaf)
    _tri_inv_lower_inplace(D, outD, workD, leaf=leaf)

    torch.mm(outD, C, out=workC)
    torch.mm(workC, outA, out=outC)
    outC.mul_(-1.0)
    out[:k, k:].zero_()


def _spd_inv_from_cholesky(
    L: torch.Tensor, invA: torch.Tensor, linv: torch.Tensor, work: torch.Tensor
) -> None:
    _tri_inv_lower_inplace(L, linv, work, leaf=512)
    torch.mm(linv.mT, linv, out=invA)


def _ensure_workspace(
    workspace: Optional[DWH2Workspace],
    n: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> DWH2Workspace:
    if (
        workspace is None
        or workspace.n != n
        or workspace.device != device
        or workspace.out_dtype != out_dtype
    ):
        return DWH2Workspace.allocate(
            n,
            device,
            out_dtype,
            block_rows=GRAM_BLOCK_ROWS,
        )
    return workspace


@torch.no_grad()
def normalize_small_gram(
    a: torch.Tensor,
    *,
    eps: float = NORM_EPS,
    safety: float = NORM_SAFETY,
    workspace: Optional[DWH2Workspace] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    transposed = a.shape[0] < a.shape[1]
    x = a.mT if transposed else a
    m, n = x.shape
    device = a.device

    workspace = _ensure_workspace(workspace, n, device, a.dtype)

    gram = workspace.gram
    scratch = workspace.scratch
    xbuf = workspace.xbuf
    tmp = workspace.tmp

    gram.zero_()
    t1 = torch.zeros((), device=device, dtype=torch.float64)

    br = workspace.block_rows
    for s in range(0, m, br):
        r = min(br, m - s)
        xbuf[:r].copy_(x[s : s + r])
        gram.addmm_(xbuf[:r].mT, xbuf[:r])
        t1 += torch.sum(xbuf[:r] * xbuf[:r], dtype=torch.float64)

    _symmetrize_(gram, scratch)

    t1_hat = torch.sum(gram.diagonal(), dtype=torch.float64)
    tmp.copy_(gram)
    tmp.mul_(gram)
    t2_hat = torch.sum(tmp, dtype=torch.float64)

    n_t = torch.tensor(float(n), device=device, dtype=torch.float64)
    rad = torch.clamp_min((n_t * t2_hat) - (t1_hat * t1_hat), 0.0)
    raw_lambda = (t1_hat + torch.sqrt((n_t - 1.0) * rad)) / n_t

    u_acc = float(torch.finfo(torch.float32).eps) / 2.0
    eta = _gamma(m, u_acc)
    if _uses_tf32_matmul():
        eta += 2.0**-10
    eta_t = torch.tensor(float(eta), device=device, dtype=torch.float64)

    ub_lambda = torch.clamp_min(raw_lambda + eta_t * t1, 0.0)

    denom = torch.sqrt(ub_lambda).to(dtype=torch.float32)
    if safety != 1.0:
        denom = denom * float(safety)
    denom = denom + float(eps)

    inv_denom = denom.reciprocal()
    inv_denom_sq = inv_denom * inv_denom
    scale = inv_denom.to(dtype=a.dtype)

    gram.mul_(inv_denom_sq)
    return scale, gram


@torch.no_grad()
def normalize_moment_with_small_gram(
    a: torch.Tensor,
    *,
    eps: float = NORM_EPS,
    safety: float = NORM_SAFETY,
    workspace: Optional[DWH2Workspace] = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, gram = normalize_small_gram(
        a,
        eps=eps,
        safety=safety,
        workspace=workspace,
    )

    if inplace:
        a.mul_(scale)
        a_norm = a
    else:
        a_norm = a * scale
    return a_norm, gram


def _stage1_core(
    gram: torch.Tensor,
    workspace: DWH2Workspace,
    s0: StepParams,
    stats: Optional[CholStats],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buf = workspace.buf
    scratch = workspace.scratch
    h0 = workspace.h0
    k0 = workspace.k0
    m0 = workspace.m0
    rhs = workspace.rhs
    linv = workspace.linv
    L = workspace.L
    info = workspace.info

    buf.copy_(gram).mul_(s0.c)
    buf.diagonal().add_(1.0)
    L = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)

    _spd_inv_from_cholesky(L, h0, linv, rhs)
    _symmetrize_(h0, scratch)

    k0.copy_(h0).mul_(-1.0)
    k0.diagonal().add_(1.0)

    m0.copy_(h0).mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    _symmetrize_(m0, scratch)
    return h0, k0, m0


def _cross_term_core(
    h0: torch.Tensor,
    k0: torch.Tensor,
    workspace: DWH2Workspace,
) -> torch.Tensor:
    scratch = workspace.scratch
    rhs = workspace.rhs

    torch.mm(h0, k0, out=rhs)
    _symmetrize_(rhs, scratch)
    return rhs


def _stage2_core(
    gram: torch.Tensor,
    k0: torch.Tensor,
    rhs_cross: torch.Tensor,
    workspace: DWH2Workspace,
    s0: StepParams,
    delta: float,
    theta: float,
    stats: Optional[CholStats],
) -> torch.Tensor:
    buf = workspace.buf
    scratch = workspace.scratch
    L = workspace.L
    info = workspace.info

    buf.copy_(gram).mul_(delta * s0.c * (s0.alpha * s0.alpha))
    buf.add_(k0, alpha=delta * 2.0 * s0.alpha * s0.beta)
    buf.add_(rhs_cross, alpha=delta * (s0.beta * s0.beta) * theta)
    buf.diagonal().add_(1.0)
    return _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)


def _solve_stage2_apply(
    L: torch.Tensor,
    m0: torch.Tensor,
    workspace: DWH2Workspace,
) -> torch.Tensor:
    rhs = workspace.rhs
    tmp = workspace.tmp
    rhs.copy_(m0.mT)
    torch.linalg.solve_triangular(L, rhs, upper=False, left=True, out=rhs)
    torch.linalg.solve_triangular(L.mT, rhs, upper=True, left=True, out=rhs)
    tmp.copy_(rhs.mT)
    return tmp


def _resolve_apply_mode(
    apply: str,
    workspace: DWH2Workspace,
    *,
    retries: int = 0,
    theta: float = 1.0,
) -> str:
    if apply != "auto":
        return apply
    if workspace.out_dtype == torch.bfloat16 and (retries > 0 or theta < 1.0):
        return "fp32"
    return "fp16"


def _apply_k_to_input(
    a_norm: torch.Tensor,
    k_final: torch.Tensor,
    workspace: DWH2Workspace,
    *,
    apply: str,
    norm_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    transposed = a_norm.shape[0] < a_norm.shape[1]

    if apply == "fp16":
        k_cast = workspace.ensure_k_cast()
        k_cast.copy_(k_final)
        if norm_scale is not None:
            k_cast.mul_(norm_scale.to(device=k_cast.device, dtype=k_cast.dtype))
        if transposed:
            return k_cast @ a_norm
        return a_norm @ k_cast

    if norm_scale is not None:
        workspace.tmp.copy_(k_final)
        workspace.tmp.mul_(
            norm_scale.to(device=workspace.tmp.device, dtype=workspace.tmp.dtype)
        )
        k_apply = workspace.tmp
    else:
        k_apply = k_final

    if transposed:
        return k_apply @ a_norm.float()
    return a_norm.float() @ k_apply


@torch.no_grad()
def _dwh2_core_fast_impl(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert apply in ("fp16", "fp32", "auto")
    del ell0

    n = int(min(a_norm.shape))
    workspace = _ensure_workspace(workspace, n, a_norm.device, a_norm.dtype)
    apply_mode = _resolve_apply_mode(apply, workspace)

    p = get_dwh2_params(PAPER_MUON_ELL) if params is None else params
    s0, s1, delta = p.step0, p.step1, p.delta

    gram = workspace.gram
    gram.copy_(gram_norm)
    _add_adaptive_ridge_tensor_(gram, n, workspace.out_dtype)

    h0, k0, m0 = _stage1_core(gram, workspace, s0, stats=None)
    rhs_cross = _cross_term_core(h0, k0, workspace)
    L = _stage2_core(gram, k0, rhs_cross, workspace, s0, delta, 1.0, stats=None)
    tmp = _solve_stage2_apply(L, m0, workspace)

    k_final = workspace.k_final
    k_final.copy_(m0).mul_(s1.alpha)
    k_final.add_(tmp, alpha=s1.beta)
    _symmetrize_(k_final, workspace.scratch)

    return _apply_k_to_input(
        a_norm,
        k_final,
        workspace,
        apply=apply_mode,
        norm_scale=norm_scale,
    )


@torch.no_grad()
def _dwh2_core_impl(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
    stats: Optional[CholStats] = None,
) -> tuple[torch.Tensor, float, float, int, list[float], list[float]]:
    assert apply in ("fp16", "fp32", "auto")
    n = int(min(a_norm.shape))
    workspace = _ensure_workspace(workspace, n, a_norm.device, a_norm.dtype)

    gram = workspace.gram
    scratch = workspace.scratch
    k_final = workspace.k_final

    current_ell0 = float(ell0)
    theta = 1.0
    retry_count = 0
    max_retries = 3
    alpha_log: list[float] = []
    s_c_log: list[float] = []

    while True:
        p = (
            get_dwh2_params(current_ell0)
            if params is None or retry_count > 0
            else params
        )
        s0, s1, delta = p.step0, p.step1, p.delta

        gram.copy_(gram_norm)
        _add_adaptive_ridge_tensor_(gram, n, workspace.out_dtype)

        h0, k0, m0 = _stage1_core(gram, workspace, s0, stats=stats)
        rhs_cross = _cross_term_core(h0, k0, workspace)
        s_c = _mat_skew_rel(rhs_cross, scratch)

        shifted_before = 0 if stats is None else stats.shifted_calls
        L = _stage2_core(
            gram,
            k0,
            rhs_cross,
            workspace,
            s0,
            delta,
            theta,
            stats=stats,
        )
        stage2_shifted = (
            False if stats is None else (stats.shifted_calls > shifted_before)
        )
        tmp = _solve_stage2_apply(L, m0, workspace)

        alpha = _solve_amplification(tmp, m0)
        solve_resid = _solve_residual_rel(workspace.buf, tmp, m0, workspace.resid)
        alpha_log.append(alpha)
        s_c_log.append(s_c)

        need_retry = (
            alpha > _BACKTRACK_ALPHA_LIMIT
            or s_c > _BACKTRACK_SKEW_LIMIT
            or solve_resid > _BACKTRACK_SOLVE_RESID_LIMIT
            or (stage2_shifted and alpha > _BACKTRACK_SHIFTED_ALPHA_SOFT_LIMIT)
        )

        if need_retry and retry_count < max_retries:
            if theta > 0.125:
                theta *= 0.5
            else:
                theta = 1.0
                current_ell0 *= 10.0
            retry_count += 1
            continue

        k_final.copy_(m0).mul_(s1.alpha)
        k_final.add_(tmp, alpha=s1.beta * theta)
        _symmetrize_(k_final, scratch)
        break

    apply_mode = _resolve_apply_mode(apply, workspace, retries=retry_count, theta=theta)

    res = _apply_k_to_input(
        a_norm,
        k_final,
        workspace,
        apply=apply_mode,
        norm_scale=norm_scale,
    )
    return res, theta, current_ell0, retry_count, alpha_log, s_c_log


@torch.no_grad()
def dwh2_core_q(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dwh2_core_fast_impl(
        a_norm,
        gram_norm,
        ell0=ell0,
        params=params,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
    )


@torch.no_grad()
def dwh2_core(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
) -> PolarResult:
    stats = CholStats()
    y, theta, final_ell0, retries, a_log, s_log = _dwh2_core_impl(
        a_norm,
        gram_norm,
        ell0=ell0,
        params=params,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
        stats=stats,
    )
    return PolarResult(
        q=y,
        stats=stats,
        theta=theta,
        ell0=final_ell0,
        retries=retries,
        alpha_log=a_log,
        s_c_log=s_log,
    )


@torch.no_grad()
def dwh2_end_to_end(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    eps: float = NORM_EPS,
    safety: float = NORM_SAFETY,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    inplace_normalize: bool = False,
) -> PolarResult:
    del inplace_normalize

    n = int(min(a.shape))
    workspace = _ensure_workspace(workspace, n, a.device, a.dtype)

    scale, gram_norm = normalize_small_gram(
        a,
        eps=eps,
        safety=safety,
        workspace=workspace,
    )
    return dwh2_core(
        a,
        gram_norm,
        ell0=ell0,
        workspace=workspace,
        apply=apply,
        norm_scale=scale,
    )
