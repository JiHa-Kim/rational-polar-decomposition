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


PAPER_MUON_ELL = 1e-3
NORM_EPS = 1e-7
NORM_SAFETY = 1.01
GRAM_BLOCK_ROWS = 1024


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
    cross: torch.Tensor
    rhs: torch.Tensor
    k_final: torch.Tensor
    linv: torch.Tensor
    work: torch.Tensor
    sh: torch.Tensor
    invsh: torch.Tensor

    # fp16/bf16 apply buffer
    k_cast: torch.Tensor

    # normalization streaming buffer
    xbuf: torch.Tensor

    # cholesky_ex outs
    L0: torch.Tensor
    L1: torch.Tensor
    info0: torch.Tensor
    info1: torch.Tensor

    @staticmethod
    def allocate(
        n: int,
        device: torch.device,
        out_dtype: torch.dtype,
        block_rows: int = GRAM_BLOCK_ROWS,
    ) -> "DWH2Workspace":
        def mat32():
            return torch.empty((n, n), device=device, dtype=torch.float32)

        def vec32():
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
            cross=mat32(),
            rhs=mat32(),
            k_final=mat32(),
            linv=mat32(),
            work=mat32(),
            sh=vec32(),
            invsh=vec32(),
            k_cast=torch.empty((n, n), device=device, dtype=out_dtype),
            xbuf=torch.empty((block_rows, n), device=device, dtype=torch.float32),
            L0=mat32(),
            L1=mat32(),
            info0=torch.empty((), device=device, dtype=torch.int32),
            info1=torch.empty((), device=device, dtype=torch.int32),
        )


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


@torch.compiler.disable(recursive=False, reason="data-dependent cholesky retry loop")
def _chol_spd_inplace_ex(
    a: torch.Tensor,
    stats: CholStats,
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

    u = float(torch.finfo(a.dtype).eps)
    jitter = max(jitter_scale * n * u, min_jitter)
    a.diagonal().add_(jitter)

    torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
    if int(info_out.item()) == 0:
        stats.update(shifted=True, retries=0, jitter=jitter)
        return L_out

    retries = 1
    step = jitter * 10.0
    while int(info_out.item()) != 0 and retries < max_retries:
        a.diagonal().add_(step)
        jitter += step
        torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
        retries += 1
        step *= 10.0

    if int(info_out.item()) != 0:
        a.diagonal().add_(0.1)
        jitter += 0.1
        torch.linalg.cholesky(a, out=L_out)
        retries += 1

    stats.update(shifted=True, retries=retries, jitter=jitter)
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


@torch.no_grad()
def normalize_moment_with_small_gram(
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

    if (
        workspace is None
        or workspace.n != n
        or workspace.device != device
        or workspace.out_dtype != a.dtype
    ):
        workspace = DWH2Workspace.allocate(
            n, device, a.dtype, block_rows=GRAM_BLOCK_ROWS
        )

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
    tmp.copy_(gram).mul_(gram)
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

    a_norm = a * inv_denom.to(dtype=a.dtype)
    gram.mul_(inv_denom_sq)
    _symmetrize_(gram, scratch)
    return a_norm, gram


@torch.no_grad()
def dwh2_core(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    params: Optional[DWH2Params] = None,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",  # "fp16" (fast) or "fp32" (quality)
) -> PolarResult:
    assert apply in ("fp16", "fp32")
    transposed = a_norm.shape[0] < a_norm.shape[1]
    x = a_norm.mT.contiguous() if transposed else a_norm.contiguous()
    n = x.shape[1]

    if (
        workspace is None
        or workspace.n != n
        or workspace.device != x.device
        or workspace.out_dtype != x.dtype
    ):
        workspace = DWH2Workspace.allocate(
            n, x.device, x.dtype, block_rows=GRAM_BLOCK_ROWS
        )

    stats = CholStats()
    p = params if params is not None else get_dwh2_params(float(ell0))
    s0, s1, delta = p.step0, p.step1, p.delta

    gram = workspace.gram
    buf = workspace.buf
    scratch = workspace.scratch
    h0 = workspace.h0
    k0 = workspace.k0
    m0 = workspace.m0
    tmp = workspace.tmp
    cross = workspace.cross
    rhs = workspace.rhs
    k_final = workspace.k_final
    linv = workspace.linv
    work = workspace.work
    sh = workspace.sh
    invsh = workspace.invsh
    k_cast = workspace.k_cast

    L0 = workspace.L0
    L1 = workspace.L1
    info0 = workspace.info0
    info1 = workspace.info1

    gram.copy_(gram_norm)

    buf.copy_(gram).mul_(s0.c)
    buf.diagonal().add_(1.0)
    L0 = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L0, info_out=info0)

    _spd_inv_from_cholesky(L0, h0, linv, work)
    _symmetrize_(h0, scratch)

    k0.copy_(h0).mul_(-1.0)
    k0.diagonal().add_(1.0)

    m0.copy_(h0).mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    _symmetrize_(m0, scratch)

    buf.copy_(gram).mul_(delta * s0.c * (s0.alpha * s0.alpha))

    sh.copy_(h0.diagonal())
    torch.clamp_(sh, min=1e-30)
    torch.sqrt_(sh)
    invsh.copy_(sh).reciprocal_()

    scratch.copy_(k0).mul_(sh[:, None])
    tmp.copy_(h0).mul_(invsh[:, None]).mul_(invsh[None, :])

    torch.mm(tmp, scratch, out=cross)
    cross.mul_(sh[:, None])

    buf.add_(k0, alpha=delta * 2.0 * s0.alpha * s0.beta)
    buf.add_(cross, alpha=delta * (s0.beta * s0.beta))
    buf.diagonal().add_(1.0)
    _symmetrize_(buf, scratch)

    L1 = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L1, info_out=info1)

    rhs.copy_(m0.mT)
    torch.linalg.solve_triangular(L1, rhs, upper=False, left=True, out=rhs)
    torch.linalg.solve_triangular(L1.mT, rhs, upper=True, left=True, out=rhs)
    tmp.copy_(rhs.mT)

    k_final.copy_(m0).mul_(s1.alpha)
    k_final.add_(tmp, alpha=s1.beta)
    _symmetrize_(k_final, scratch)

    if apply == "fp16":
        k_cast.copy_(k_final)
        y = x @ k_cast
    else:
        # Quality mode: do final apply in fp32 and return fp32 Q
        y = x.float() @ k_final

    if transposed:
        y = y.mT.contiguous()
    return PolarResult(q=y, stats=stats)


@torch.no_grad()
def dwh2_end_to_end(
    a: torch.Tensor,
    *,
    ell0: float = PAPER_MUON_ELL,
    eps: float = NORM_EPS,
    safety: float = NORM_SAFETY,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
) -> PolarResult:
    n = int(min(a.shape))
    if (
        workspace is None
        or workspace.n != n
        or workspace.device != a.device
        or workspace.out_dtype != a.dtype
    ):
        workspace = DWH2Workspace.allocate(
            n, a.device, a.dtype, block_rows=GRAM_BLOCK_ROWS
        )

    a_norm, gram_norm = normalize_moment_with_small_gram(
        a, eps=eps, safety=safety, workspace=workspace
    )
    return dwh2_core(a_norm, gram_norm, ell0=ell0, workspace=workspace, apply=apply)
