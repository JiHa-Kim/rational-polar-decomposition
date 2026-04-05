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
    stats: CholStats = field(default_factory=CholStats)


@dataclass(frozen=True)
class DWH2Config:
    ell0: float = 1e-3
    eps: float = 1e-7
    safety: float = 1.01
    gram_block_rows: int = 1024


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
    k_final: torch.Tensor
    sh: torch.Tensor
    invsh: torch.Tensor

    xbuf: torch.Tensor
    L: torch.Tensor
    info: torch.Tensor
    tmp: torch.Tensor
    rhs: torch.Tensor
    k_cast: Optional[torch.Tensor] = None

    @staticmethod
    def allocate(
        n: int,
        device: torch.device,
        out_dtype: torch.dtype,
        block_rows: int = 1024,
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
            k_final=mat32(),
            sh=vec32(),
            invsh=vec32(),
            xbuf=torch.empty((block_rows, n), device=device, dtype=torch.float32),
            L=mat32(),
            info=torch.empty((), device=device, dtype=torch.int32),
            tmp=mat32(),
            rhs=mat32(),
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
                (self.n, self.n), device=self.device, dtype=self.out_dtype
            )
            self.k_cast = k_cast
        return k_cast


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


def _column_space_view(a: torch.Tensor) -> torch.Tensor:
    return a.mT if a.shape[0] < a.shape[1] else a


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> None:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)


def _uses_tf32_matmul() -> bool:
    # Use a static check to avoid graph breaks in torch.compile
    if not torch.cuda.is_available():
        return False
    # TF32 is usually the default 'high' precision on modern CUDA devices.
    # We return True to avoid calling string-returning precision queries during tracing.
    return True


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
    _symmetrize_(a, scratch)
    n = a.shape[0]
    u = float(torch.finfo(a.dtype).eps)
    jitter = max(jitter_scale * n * u, min_jitter)

    # Equilibrate: D^-1 A D^-1
    d = torch.clamp(a.diagonal(), min=1e-30).sqrt()
    inv_d = d.reciprocal()
    a.mul_(inv_d[:, None]).mul_(inv_d[None, :])

    a.diagonal().add_(jitter)
    torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
    if int(info_out.item()) == 0:
        _update_chol_stats(stats, shifted=False, retries=0, jitter=jitter)
        L_out.mul_(d[:, None])
        return L_out

    retries = 1
    step = jitter * 10.0
    while int(info_out.item()) != 0 and retries <= max_retries:
        a.diagonal().add_(step)
        jitter += step
        torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
        if int(info_out.item()) == 0:
            _update_chol_stats(stats, shifted=True, retries=retries, jitter=jitter)
            L_out.mul_(d[:, None])
            return L_out
        retries += 1
        step *= 10.0

    # Final fallback if all retries fail
    a.diagonal().add_(0.1)
    jitter += 0.1
    torch.linalg.cholesky_ex(a, check_errors=False, out=(L_out, info_out))
    _update_chol_stats(stats, shifted=True, retries=retries, jitter=jitter)

    # Map back: L_out = D L_out
    L_out.mul_(d[:, None])
    return L_out


def _spd_inv_from_cholesky(
    L: torch.Tensor,
    invA: torch.Tensor,
    work: torch.Tensor,
) -> None:
    invA.zero_()
    invA.diagonal().fill_(1.0)
    torch.linalg.solve_triangular(L, invA, upper=False, left=True, out=invA)
    torch.linalg.solve_triangular(L.mT, invA, upper=True, left=True, out=invA)


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
    x = _column_space_view(a)
    m, n = x.shape
    workspace = _ensure_workspace(workspace, n, a.device, a.dtype, config)

    gram = workspace.gram
    scratch = workspace.scratch
    xbuf = workspace.xbuf

    t1 = torch.linalg.vector_norm(x, dtype=torch.float32).pow(2).to(torch.float64)

    eta_fp16 = 0.0
    # OPTIMIZATION: Single-pass addmm for standard sizes to avoid loop overhead.
    # For m=16k, n=4k, fp32 cast takes 256MB.
    # We use a threshold (e.g., 32768 rows) to decide.
    if m <= 32768:
        if x.dtype == torch.float16:
            ratio = (t1 / 32768.0).clamp_min(1.0)
            if not torch.compiler.is_compiling() and float(ratio.item()) == 1.0:
                tmp_fp16 = torch.mm(x.mT, x)
                gram.copy_(tmp_fp16)
            else:
                s_t = torch.sqrt(ratio).to(x.dtype)
                xs = x / s_t
                tmp_fp16 = torch.mm(xs.mT, xs)
                gram.copy_(tmp_fp16).mul_(ratio.to(torch.float32))
            eta_fp16 = float(torch.finfo(torch.float16).eps)
        else:
            # Fused Single Pass
            xf = x.float()
            gram.addmm_(xf.mT, xf, beta=0.0)
    else:
        # Blocked Loop Fallback
        gram.zero_()
        for start in range(0, m, workspace.block_rows):
            rows = min(workspace.block_rows, m - start)
            x_rows = x[start : start + rows]
            xbuf[:rows].copy_(x_rows)
            gram.addmm_(xbuf[:rows].mT, xbuf[:rows])

    _symmetrize_(gram, scratch)

    t1_hat = torch.sum(gram.diagonal(), dtype=torch.float64)
    # Safely compute Frobenius norm squared in float64
    t2_hat = torch.sum(gram.to(dtype=torch.float64).pow(2))

    # Spectral radius estimation (Stable kernel from baseline)
    n_t = torch.tensor(float(n), device=a.device, dtype=torch.float64)
    rad = torch.clamp_min((n_t * t2_hat) - (t1_hat * t1_hat), 0.0)
    raw_lambda = (t1_hat + torch.sqrt((n_t - 1.0) * rad)) / n_t

    # Machine eps consideration (Safe baseline gamma)
    eps_val = float(torch.finfo(torch.float32).eps)
    eta = _gamma(m, eps_val / 2.0)
    if _uses_tf32_matmul():
        eta += 2.0**-10
    eta += eta_fp16
    eta_t = torch.tensor(float(eta), device=a.device, dtype=torch.float64)

    # Note: Baseline used 't1' (stable sum of squares), not 't1_hat'.
    ub_lambda = torch.clamp_min(raw_lambda + eta_t * t1, 0.0)
    denom = torch.sqrt(ub_lambda).to(dtype=torch.float32)

    # Safety factor and eps
    denom = (denom * float(config.safety)) + float(config.eps)

    inv_denom = denom.reciprocal()
    scale = inv_denom.to(dtype=a.dtype)
    gram.mul_(inv_denom * inv_denom)
    # Stability jitter: ensure G is strictly PSD for subsequent steps
    u = float(torch.finfo(torch.float32).eps)
    gram.diagonal().add_(u, alpha=float(n))
    return scale, gram


@torch.no_grad()
def normalize_moment_with_small_gram(
    a: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    workspace: Optional[DWH2Workspace] = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, gram = normalize_small_gram(a, config=config, workspace=workspace)
    if inplace:
        a.mul_(scale)
        return a, gram
    return a * scale, gram


def _apply_k(
    a_norm: torch.Tensor,
    k_final: torch.Tensor,
    *,
    workspace: DWH2Workspace,
    apply: str,
    norm_scale: Optional[torch.Tensor],
    transposed: bool,
    tmp: torch.Tensor,
) -> torch.Tensor:
    if apply == "fp16":
        out_dtype = a_norm.dtype
        finfo = torch.finfo(out_dtype)
        safe_max = 0.90 * float(finfo.max)

        k_raw_absmax = torch.amax(torch.abs(k_final))
        k_absmax = k_raw_absmax
        if norm_scale is not None:
            k_absmax = k_absmax * torch.abs(norm_scale)

        k_apply = workspace.ensure_k_cast()

        s_f32 = 1.0
        # Use simple eager float check
        if float(k_absmax.item()) > safe_max:
            ratio = (k_absmax / safe_max).clamp_min(1.0)
            s_f32 = float(
                torch.pow(
                    torch.tensor(2.0, device=ratio.device),
                    torch.ceil(torch.log2(ratio)),
                ).item()
            )

        if k_raw_absmax > float(finfo.max) or norm_scale is not None or s_f32 != 1.0:
            tmp.copy_(k_final)
            if norm_scale is not None:
                tmp.mul_(norm_scale)
            if s_f32 != 1.0:
                tmp.div_(s_f32)
            k_apply.copy_(tmp)
        else:
            k_apply.copy_(k_final)

        q_scaled = k_apply @ a_norm if transposed else a_norm @ k_apply

        if s_f32 != 1.0:
            return (q_scaled.float() * s_f32).to(dtype=out_dtype)
        return q_scaled

    # fp32 apply path
    k_apply = k_final
    if norm_scale is not None:
        tmp.copy_(k_final)
        tmp.mul_(norm_scale)
        k_apply = tmp

    x = a_norm.float()
    return k_apply @ x if transposed else x @ k_apply


def _compute_gram_blocked(
    x: torch.Tensor,
    m_mat: Optional[torch.Tensor],
    *,
    workspace: DWH2Workspace,
    out: torch.Tensor,
) -> torch.Tensor:
    # Computes G = (X M)^T (X M) if m_mat is not None, else X^T X
    # x: (m, n), m_mat: (n, n), out: (n, n)
    m, n = x.shape
    out.zero_()

    # We use float32 for these matmuls to ensure precision and use TF32 speedup.
    # The memory cost for 16k x 4k float32 is 256MB, which is acceptable.

    if m <= 32768:
        if m_mat is not None:
            # Y = X.float() @ M.float()
            y = torch.mm(x.float(), m_mat.float())
            out.addmm_(y.mT, y)
        else:
            xf = x.float()
            out.addmm_(xf.mT, xf)
        return out

    # Blocked fallback for truly huge matrices
    for start in range(0, m, workspace.block_rows):
        rows = min(workspace.block_rows, m - start)
        x_rows = x[start : start + rows].float()

        if m_mat is not None:
            y = torch.mm(x_rows, m_mat.float())
            out.addmm_(y.mT, y)
        else:
            out.addmm_(x_rows.mT, x_rows)
    return out


def _dwh2_core_impl(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    params: Optional[DWH2Params] = None,
    workspace: DWH2Workspace,
    apply: str = "fp16",
    norm_scale: Optional[torch.Tensor] = None,
    stats: Optional[CholStats] = None,
) -> torch.Tensor:
    transposed = a_norm.shape[0] < a_norm.shape[1]

    p = params if params is not None else get_dwh2_params(float(config.ell0))
    s0, s1, delta = p.step0, p.step1, p.delta

    gram = workspace.gram
    buf = workspace.buf
    scratch = workspace.scratch
    h0 = workspace.h0
    k0 = workspace.k0
    m0 = workspace.m0
    k_final = workspace.k_final
    sh = workspace.sh
    invsh = workspace.invsh
    L = workspace.L
    info = workspace.info

    gram.copy_(gram_norm)

    # Step 1: B0 = I + c0 G
    # Scaling Trick: chol(1/c0 I + G)
    buf.copy_(gram)
    buf.diagonal().add_(1.0 / s0.c)
    L = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)
    L.mul_(math.sqrt(s0.c))

    _spd_inv_from_cholesky(L, h0, scratch)
    _symmetrize_(h0, scratch)

    # Stable K0 = I - H0
    k0.copy_(h0).mul_(-1.0)
    k0.diagonal().add_(1.0)
    _symmetrize_(k0, scratch)

    m0.copy_(h0).mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    _symmetrize_(m0, scratch)

    # Stable L_prod = H0 K0 via equilibration trick
    sh.copy_(h0.diagonal())
    torch.clamp_(sh, min=1e-30)
    torch.sqrt_(sh)
    invsh.copy_(sh).reciprocal_()

    h0.mul_(invsh[:, None]).mul_(invsh[None, :])
    scratch.copy_(k0).mul_(sh[:, None])
    torch.mm(h0, scratch, out=L)
    L.mul_(sh[:, None])
    _symmetrize_(L, scratch)

    # Step 2: B1 = I + delta (c0 a0^2 G + 2 a0 b0 K0 + b0^2 L_prod)
    # Scaling Trick for B1:
    # B1 = c1 * (alpha0^2 G + (2*alpha0*beta0/c0) K0 + (beta0^2/c0) L + 1/c1 I)
    c1 = delta * s0.c
    buf.copy_(gram).mul_(s0.alpha * s0.alpha)
    buf.add_(k0, alpha=(2.0 * s0.alpha * s0.beta) / s0.c)
    buf.add_(L, alpha=(s0.beta * s0.beta) / s0.c)
    buf.diagonal().add_(1.0 / c1)

    L = _chol_spd_inplace_ex(buf, stats, scratch=scratch, L_out=L, info_out=info)
    L.mul_(math.sqrt(c1))
    scratch.copy_(m0.mT)
    torch.linalg.solve_triangular(L, scratch, upper=False, left=True, out=scratch)
    torch.linalg.solve_triangular(L.mT, scratch, upper=True, left=True, out=scratch)
    h0.copy_(scratch.mT)

    k_final.copy_(m0).mul_(s1.alpha)
    k_final.add_(h0, alpha=s1.beta)
    _symmetrize_(k_final, scratch)

    return _apply_k(
        a_norm,
        k_final,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
        transposed=transposed,
        tmp=h0,
    )


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
    q = _dwh2_core_impl(
        a_norm,
        gram_norm,
        config=config,
        params=params,
        workspace=workspace,
        apply=apply,
        norm_scale=norm_scale,
        stats=stats,
    )
    return PolarResult(q=q, stats=stats)


@torch.no_grad()
def dwh2_end_to_end(
    a: torch.Tensor,
    *,
    config: DWH2Config = DEFAULT_CONFIG,
    workspace: Optional[DWH2Workspace] = None,
    apply: str = "fp16",
) -> PolarResult:
    m, n = a.shape
    n_dim = min(m, n)
    workspace = _ensure_workspace(workspace, n_dim, a.device, a.dtype, config)
    scale, gram_norm = normalize_small_gram(a, config=config, workspace=workspace)

    stats = CholStats()
    q = _dwh2_core_impl(
        a,
        gram_norm,
        config=config,
        workspace=workspace,
        apply=apply,
        norm_scale=scale,
        stats=stats,
    )
    return PolarResult(q=q, stats=stats)
