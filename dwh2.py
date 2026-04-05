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


@dataclass(frozen=True)
class DWH2Config:
    ell0: float = 1e-3
    eps: float = 1e-7
    safety: float = 1.01
    gram_block_rows: int = 1024


# Default configuration instance
DEFAULT_CONFIG = DWH2Config()


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


class DWH2Workspace:
    def __init__(
        self,
        n: int,
        device: torch.device,
        out_dtype: torch.dtype,
        block_rows: int = 1024,
    ):
        self.n = n
        self.device = device
        self.out_dtype = out_dtype
        self.block_rows = block_rows

        # Pre-allocate standard buffers in float32 for stability
        def mat32() -> torch.Tensor:
            return torch.empty((n, n), device=device, dtype=torch.float32)

        def vec32() -> torch.Tensor:
            return torch.empty((n,), device=device, dtype=torch.float32)

        self.gram = mat32()
        self.buf = mat32()
        self.scratch = mat32()
        self.h0 = mat32()
        self.k0 = mat32()
        self.m0 = mat32()
        self.tmp = mat32()
        self.rhs = mat32()
        self.k_final = mat32()
        self.linv = mat32()
        self.sh = vec32()
        self.invsh = vec32()
        self.xbuf = torch.empty((block_rows, n), device=device, dtype=torch.float32)
        self.L = mat32()
        self.info = torch.empty((), device=device, dtype=torch.int32)

        # Optional/Lazy buffers
        self._k_cast: Optional[torch.Tensor] = None

    @staticmethod
    def allocate(
        n: int,
        device: torch.device,
        out_dtype: torch.dtype,
        block_rows: int = 1024,
    ) -> "DWH2Workspace":
        return DWH2Workspace(n, device, out_dtype, block_rows)

    def ensure_k_cast(self) -> torch.Tensor:
        if (
            self._k_cast is None
            or self._k_cast.shape != (self.n, self.n)
            or self._k_cast.device != self.device
            or self._k_cast.dtype != self.out_dtype
        ):
            self._k_cast = torch.empty(
                (self.n, self.n),
                device=self.device,
                dtype=self.out_dtype,
            )
        return self._k_cast


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


def _ensure_workspace(
    workspace: Optional[DWH2Workspace],
    n: int,
    device: torch.device,
    out_dtype: torch.dtype,
    config: DWH2Config = DEFAULT_CONFIG,
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
            block_rows=config.gram_block_rows,
        )
    return workspace


@torch.no_grad()
def normalize_small_gram(
    a: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    workspace: Optional[DWH2Workspace] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    transposed = a.shape[0] < a.shape[1]
    x = a.mT if transposed else a
    m, n = x.shape
    device = a.device

    workspace = _ensure_workspace(workspace, n, device, a.dtype, config)

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
    if config.safety != 1.0:
        denom = denom * float(config.safety)
    denom = denom + float(config.eps)

    inv_denom = denom.reciprocal()
    inv_denom_sq = inv_denom * inv_denom
    scale = inv_denom.to(dtype=a.dtype)

    gram.mul_(inv_denom_sq)
    return scale, gram


@torch.no_grad()
def normalize_moment_with_small_gram(
    a: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    workspace: Optional[DWH2Workspace] = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, gram = normalize_small_gram(
        a,
        config=config,
        workspace=workspace,
    )

    if inplace:
        a.mul_(scale)
        a_norm = a
    else:
        a_norm = a * scale
    return a_norm, gram


def _dwh2_core_impl(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
    stats: Optional[CholStats] = None,
) -> torch.Tensor:
    assert apply in ("fp16", "fp32")
    transposed = a_norm.shape[0] < a_norm.shape[1]
    n = int(min(a_norm.shape))
    workspace = _ensure_workspace(workspace, n, a_norm.device, a_norm.dtype, config)

    p = params if params is not None else get_dwh2_params(float(config.ell0))
    s0, s1, delta = p.step0, p.step1, p.delta

    gram = workspace.gram
    buf = workspace.buf
    scratch = workspace.scratch
    h0 = workspace.h0
    k0 = workspace.k0
    m0 = workspace.m0
    tmp = workspace.tmp
    rhs = workspace.rhs
    k_final = workspace.k_final
    linv = workspace.linv
    sh = workspace.sh
    invsh = workspace.invsh
    L = workspace.L
    info = workspace.info

    gram.copy_(gram_norm)

    buf.copy_(gram).mul_(s0.c)
    buf.diagonal().add_(1.0)
    L = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)

    # Fast LAPACK-based inversion
    torch.cholesky_inverse(L, upper=False, out=h0)
    _symmetrize_(h0, scratch)

    k0.copy_(h0).mul_(-1.0)
    k0.diagonal().add_(1.0)

    m0.copy_(h0).mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    _symmetrize_(m0, scratch)

    sh.copy_(h0.diagonal())
    torch.clamp_(sh, min=1e-30)
    torch.sqrt_(sh)
    invsh.copy_(sh).reciprocal_()

    tmp.copy_(h0).mul_(invsh[:, None]).mul_(invsh[None, :])
    scratch.copy_(k0).mul_(sh[:, None])
    torch.mm(tmp, scratch, out=rhs)
    rhs.mul_(sh[:, None])

    buf.copy_(gram).mul_(delta * s0.c * (s0.alpha * s0.alpha))
    buf.add_(k0, alpha=delta * 2.0 * s0.alpha * s0.beta)
    buf.add_(rhs, alpha=delta * (s0.beta * s0.beta))
    buf.diagonal().add_(1.0)

    L = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)

    rhs.copy_(m0.mT)
    torch.linalg.solve_triangular(L, rhs, upper=False, left=True, out=rhs)
    torch.linalg.solve_triangular(L.mT, rhs, upper=True, left=True, out=rhs)
    tmp.copy_(rhs.mT)

    k_final.copy_(m0).mul_(s1.alpha)
    k_final.add_(tmp, alpha=s1.beta)
    _symmetrize_(k_final, scratch)

    if apply == "fp16":
        k_cast = workspace.ensure_k_cast()
        k_cast.copy_(k_final)
        if norm_scale is not None:
            k_cast.mul_(norm_scale.to(device=k_cast.device, dtype=k_cast.dtype))
        if transposed:
            return k_cast @ a_norm
        return a_norm @ k_cast

    if norm_scale is not None:
        tmp.copy_(k_final)
        tmp.mul_(norm_scale.to(device=tmp.device, dtype=tmp.dtype))
        k_apply = tmp
    else:
        k_apply = k_final
    if transposed:
        return k_apply @ a_norm.float()
    return a_norm.float() @ k_apply


@torch.no_grad()
def dwh2_core_q(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dwh2_core_impl(
        a_norm,
        gram_norm,
        config=config,
        params=params,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
        stats=None,
    )


@torch.no_grad()
def dwh2_core(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
) -> PolarResult:
    stats = CholStats()
    y = _dwh2_core_impl(
        a_norm,
        gram_norm,
        config=config,
        params=params,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
        stats=stats,
    )
    return PolarResult(q=y, stats=stats)


@torch.no_grad()
def dwh2_end_to_end(
    a: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
) -> PolarResult:
    n = int(min(a.shape))
    workspace = _ensure_workspace(workspace, n, a.device, a.dtype, config)

    scale, gram_norm = normalize_small_gram(
        a,
        config=config,
        workspace=workspace,
    )
    return dwh2_core(
        a,
        gram_norm,
        config=config,
        workspace=workspace,
        apply=apply,
        norm_scale=scale,
    )
