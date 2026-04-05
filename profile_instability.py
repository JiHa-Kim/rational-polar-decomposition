from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dwh2  # noqa: E402
from bench_profile import CaseGenerator, MetricsSuite  # noqa: E402


@dataclass
class TensorSummary:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    nan_count: int
    inf_count: int
    abs_max: float
    abs_mean: float
    fro: float
    diag_min: float | None = None
    diag_max: float | None = None
    sym_rel_fro: float | None = None
    trace: float | None = None
    stable_rank: float | None = None
    min_eig: float | None = None
    max_eig: float | None = None


def _to_f32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()


def _safe_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    return float(x.item())


def _matrix_or_vector_fro(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return torch.linalg.matrix_norm(x)
    return torch.linalg.vector_norm(x)


def summarize_tensor(name: str, x: torch.Tensor, eigs: bool = False) -> TensorSummary:
    xf = _to_f32(x)
    finite_mask = torch.isfinite(xf)
    finite = bool(finite_mask.all().item())
    nan_count = int(torch.isnan(xf).sum().item())
    inf_count = int(torch.isinf(xf).sum().item())

    if finite:
        abs_x = xf.abs()
        abs_max = _safe_float(abs_x.max())
        abs_mean = _safe_float(abs_x.mean())
        fro = _safe_float(_matrix_or_vector_fro(xf))
    else:
        finite_vals = xf[finite_mask]
        if finite_vals.numel() == 0:
            abs_max = float("nan")
            abs_mean = float("nan")
            fro = float("nan")
        else:
            abs_vals = finite_vals.abs()
            abs_max = _safe_float(abs_vals.max())
            abs_mean = _safe_float(abs_vals.mean())
            fro = _safe_float(torch.linalg.vector_norm(finite_vals))

    diag_min = diag_max = sym_rel_fro = trace = stable_rank = None
    min_eig = max_eig = None
    if xf.ndim == 2 and xf.shape[0] == xf.shape[1]:
        d = xf.diagonal()
        if d.numel() > 0:
            diag_min = _safe_float(d.min())
            diag_max = _safe_float(d.max())
            trace = _safe_float(torch.sum(d, dtype=torch.float64))
        if finite:
            denom = torch.linalg.matrix_norm(xf).clamp_min(1e-30)
            sym_rel_fro = _safe_float(torch.linalg.matrix_norm(xf - xf.mT) / denom)
            t1 = torch.sum(d, dtype=torch.float64)
            t2 = torch.sum(xf * xf, dtype=torch.float64)
            if t2.item() > 0:
                stable_rank = _safe_float((t1 * t1) / t2)
            if eigs:
                try:
                    evals = torch.linalg.eigvalsh(0.5 * (xf + xf.mT).double())
                    min_eig = _safe_float(evals.min())
                    max_eig = _safe_float(evals.max())
                except Exception:
                    pass

    return TensorSummary(
        name=name,
        shape=tuple(int(s) for s in x.shape),
        dtype=str(x.dtype),
        finite=finite,
        nan_count=nan_count,
        inf_count=inf_count,
        abs_max=abs_max,
        abs_mean=abs_mean,
        fro=fro,
        diag_min=diag_min,
        diag_max=diag_max,
        sym_rel_fro=sym_rel_fro,
        trace=trace,
        stable_rank=stable_rank,
        min_eig=min_eig,
        max_eig=max_eig,
    )


def print_summary(s: TensorSummary) -> None:
    parts = [
        f"{s.name:>18}",
        f"finite={str(s.finite):<5}",
        f"nan={s.nan_count:<8}",
        f"inf={s.inf_count:<8}",
        f"abs_max={s.abs_max:.3e}",
        f"fro={s.fro:.3e}",
    ]
    if s.diag_min is not None:
        parts.append(f"diag=[{s.diag_min:.3e},{s.diag_max:.3e}]")
    if s.sym_rel_fro is not None:
        parts.append(f"symerr={s.sym_rel_fro:.3e}")
    if s.stable_rank is not None:
        parts.append(f"stable_rank={s.stable_rank:.3e}")
    if s.min_eig is not None:
        parts.append(f"eig=[{s.min_eig:.3e},{s.max_eig:.3e}]")
    print(" | ".join(parts))


@torch.no_grad()
def exact_spectrum(a: torch.Tensor) -> dict[str, float | int]:
    x = a.mT if a.shape[0] < a.shape[1] else a
    xf = _to_f32(x)
    gram = 0.5 * ((xf.mT @ xf) + (xf.mT @ xf).mT)
    evals = torch.linalg.eigvalsh(gram.double())
    evals = torch.clamp_min(evals, 0.0)
    lam_max = _safe_float(evals[-1])
    tol = max(1e-12 * lam_max, 1e-30)
    nonzero = evals[evals > tol]
    return {
        "lambda_min": _safe_float(evals[0]),
        "lambda_max": lam_max,
        "stable_rank": _safe_float(
            (evals.sum() ** 2) / evals.square().sum().clamp_min(1e-30)
        ),
        "rank_eff_tol_1e12": int(nonzero.numel()),
        "cond_nonzero": _safe_float(nonzero[-1] / nonzero[0].clamp_min(1e-30))
        if nonzero.numel() >= 2
        else (1.0 if nonzero.numel() == 1 else float("inf")),
    }


@torch.no_grad()
def chol_probe(a_in: torch.Tensor, ws: dwh2.DWH2Workspace) -> dict[str, Any]:
    a = ws.buf
    a.copy_(a_in)
    dwh2._symmetrize_(a, ws.scratch)
    torch.linalg.cholesky_ex(a, check_errors=False, out=(ws.L, ws.info))
    info0 = int(ws.info.item())
    out: dict[str, Any] = {
        "info_before_shift": info0,
        "diag_min": _safe_float(a.diagonal().min()),
        "diag_max": _safe_float(a.diagonal().max()),
    }
    if info0 == 0:
        return out
    stats = dwh2.CholStats()
    dwh2._chol_spd_inplace_ex(
        a, stats, scratch=ws.scratch, L_out=ws.L, info_out=ws.info
    )
    out.update(
        {
            "info_after_shift": int(ws.info.item()),
            "shifted_calls": stats.shifted_calls,
            "total_retries": stats.total_retries,
            "max_jitter": stats.max_jitter,
        }
    )
    return out


def run_core_impl(
    a_norm: torch.Tensor,
    gram_norm: torch.Tensor,
    ws: dwh2.DWH2Workspace,
    *,
    params: dwh2.DWH2Params,
    apply: str,
    norm_scale: Optional[torch.Tensor] = None,
) -> dwh2.PolarResult:
    return dwh2.dwh2_core(
        a_norm,
        gram_norm,
        params=params,
        workspace=ws,
        apply=apply,
        norm_scale=norm_scale,
    )


@torch.no_grad()
def _build_rank_projector(
    gram: torch.Tensor, tol_rel: float
) -> tuple[torch.Tensor, int, dict[str, float]]:
    G = 0.5 * (gram.float() + gram.float().mT)
    evals, evecs = torch.linalg.eigh(G.double())
    lam_max = max(_safe_float(evals[-1]), 0.0)
    tau = max(tol_rel * lam_max, 1e-30)
    mask = evals > tau
    rank = int(mask.sum().item())
    if rank > 0:
        V = evecs[:, mask].float()
        proj = V @ V.mT
    else:
        proj = torch.zeros_like(G)
    return (
        proj,
        rank,
        {"tau": tau, "lambda_max": lam_max, "lambda_min": _safe_float(evals[0])},
    )


@torch.no_grad()
def _extra_q_diagnostics(
    a_norm: torch.Tensor,
    q: torch.Tensor,
    gram_norm: torch.Tensor,
    projector: Optional[torch.Tensor],
    projector_rank: int,
) -> dict[str, float | bool]:
    transposed = a_norm.shape[0] < a_norm.shape[1]
    X, Q = (
        (a_norm.mT.float(), q.mT.float()) if transposed else (a_norm.float(), q.float())
    )

    S = 0.5 * ((Q.mT @ Q) + (Q.mT @ Q).mT)
    G = gram_norm.float()
    H = Q.mT @ X
    P = 0.5 * (H + H.mT)
    eps = 1e-30

    out: dict[str, float | bool] = {
        "q_finite": bool(torch.isfinite(Q).all().item()),
        "s_finite": bool(torch.isfinite(S).all().item()),
        "p_finite": bool(torch.isfinite(P).all().item()),
    }
    if not (out["q_finite"] and out["s_finite"] and out["p_finite"]):
        out.update(
            {
                "proj_defect_rel": float("nan"),
                "support_resid_rel": float("nan"),
                "proj_gap_rel": float("nan"),
            }
        )
        return out

    s_fro = torch.linalg.matrix_norm(S).clamp_min_(eps)
    g_fro = torch.linalg.matrix_norm(G).clamp_min_(eps)

    out["proj_defect_rel"] = _safe_float(torch.linalg.matrix_norm(S @ S - S) / s_fro)
    out["support_resid_rel"] = _safe_float(
        torch.linalg.matrix_norm(
            (torch.eye(S.shape[0], device=S.device, dtype=S.dtype) - S) @ G
        )
        / g_fro
    )

    if projector is not None:
        proj = projector.to(device=S.device, dtype=S.dtype)
        proj_den = max(math.sqrt(float(projector_rank)), 1.0)
        out["proj_gap_rel"] = _safe_float(torch.linalg.matrix_norm(S - proj) / proj_den)
    else:
        out["proj_gap_rel"] = float("nan")

    neg_eigs = torch.linalg.eigvalsh(P.double())
    neg_mass = torch.clamp_min(-neg_eigs, 0.0).sum()
    abs_mass = neg_eigs.abs().sum().clamp_min(1e-30)
    out["p_negative_mass_rel"] = _safe_float(neg_mass / abs_mass)
    return out


@torch.no_grad()
def stage_profile(
    a: torch.Tensor,
    ws: dwh2.DWH2Workspace,
    *,
    ell0: float,
    eigs: bool,
    apply_compare: bool,
    projector_tol_rel: float,
) -> tuple[list[TensorSummary], dict[str, Any]]:
    summaries: list[TensorSummary] = []
    extras: dict[str, Any] = {}
    params = dwh2.get_dwh2_params(ell0)
    s0, s1, delta = params.step0, params.step1, params.delta

    a_work = a.clone()
    a_norm, gram = dwh2.normalize_moment_with_small_gram(
        a_work, workspace=ws, inplace=True
    )
    summaries.append(summarize_tensor("a_input", a, eigs=False))
    summaries.append(summarize_tensor("a_norm", a_norm, eigs=False))
    summaries.append(summarize_tensor("gram_norm", gram, eigs=eigs))

    extras["a_has_finite"] = bool(torch.isfinite(a).all().item())
    extras["a_norm_has_finite"] = bool(torch.isfinite(a_norm).all().item())
    extras["gram_has_finite"] = bool(torch.isfinite(gram).all().item())

    try:
        proj, proj_rank, proj_meta = _build_rank_projector(gram, projector_tol_rel)
        extras["projector_rank"] = proj_rank
        extras["projector_tau"] = proj_meta["tau"]
        extras["projector_lambda_max"] = proj_meta["lambda_max"]
        extras["projector_lambda_min"] = proj_meta["lambda_min"]
    except Exception as e:
        proj = None
        proj_rank = 0
        extras["projector_error"] = str(e)

    try:
        extras["gram_stable_rank"] = _safe_float(
            (torch.sum(gram.diagonal(), dtype=torch.float64) ** 2)
            / torch.sum(gram * gram, dtype=torch.float64).clamp_min(1e-30)
        )
    except Exception as e:
        extras["gram_stable_rank_error"] = str(e)

    if a.shape[1] <= 1024:
        try:
            extras["exact_spectrum"] = exact_spectrum(a.float())
        except Exception as e:
            extras["exact_spectrum_error"] = str(e)

    gram32 = ws.gram
    buf = ws.buf
    scratch = ws.scratch
    h0 = ws.h0
    k0 = ws.k0
    m0 = ws.m0
    tmp = ws.tmp
    rhs = ws.rhs
    k_final = ws.k_final
    linv = ws.linv
    sh = ws.sh
    invsh = ws.invsh
    L = ws.L
    info = ws.info

    gram32.copy_(gram)

    buf.copy_(gram32).mul_(s0.c)
    buf.diagonal().add_(1.0)
    extras["chol0_probe"] = chol_probe(buf, ws)
    L = dwh2._chol_spd_inplace_ex(
        buf, dwh2.CholStats(), scratch=scratch, L_out=L, info_out=info
    )
    summaries.append(summarize_tensor("step0_buf", buf, eigs=eigs))
    summaries.append(summarize_tensor("step0_L", L, eigs=False))

    dwh2._spd_inv_from_cholesky(L, h0, linv, rhs)
    dwh2._symmetrize_(h0, scratch)
    summaries.append(summarize_tensor("h0", h0, eigs=eigs))

    k0.copy_(h0).mul_(-1.0)
    k0.diagonal().add_(1.0)
    summaries.append(summarize_tensor("k0", k0, eigs=eigs))

    m0.copy_(h0).mul_(s0.beta)
    m0.diagonal().add_(s0.alpha)
    dwh2._symmetrize_(m0, scratch)
    summaries.append(summarize_tensor("m0", m0, eigs=eigs))

    sh.copy_(h0.diagonal())
    torch.clamp_(sh, min=1e-30)
    torch.sqrt_(sh)
    invsh.copy_(sh).reciprocal_()
    summaries.append(summarize_tensor("sqrt_diag_h0", sh, eigs=False))
    summaries.append(summarize_tensor("inv_sqrt_diag_h0", invsh, eigs=False))

    tmp.copy_(h0).mul_(invsh[:, None]).mul_(invsh[None, :])
    scratch.copy_(k0).mul_(sh[:, None])
    torch.mm(tmp, scratch, out=rhs)
    rhs.mul_(sh[:, None])
    summaries.append(summarize_tensor("rhs_step1", rhs, eigs=eigs))

    buf.copy_(gram32).mul_(delta * s0.c * (s0.alpha * s0.alpha))
    buf.add_(k0, alpha=delta * 2.0 * s0.alpha * s0.beta)
    buf.add_(rhs, alpha=delta * (s0.beta * s0.beta))
    buf.diagonal().add_(1.0)
    extras["chol1_probe"] = chol_probe(buf, ws)
    L = dwh2._chol_spd_inplace_ex(
        buf, dwh2.CholStats(), scratch=scratch, L_out=L, info_out=info
    )
    summaries.append(summarize_tensor("step1_buf", buf, eigs=eigs))
    summaries.append(summarize_tensor("step1_L", L, eigs=False))

    rhs.copy_(m0.mT)
    torch.linalg.solve_triangular(L, rhs, upper=False, left=True, out=rhs)
    torch.linalg.solve_triangular(L.mT, rhs, upper=True, left=True, out=rhs)
    tmp.copy_(rhs.mT)
    summaries.append(summarize_tensor("step1_solve", tmp, eigs=eigs))

    k_final.copy_(m0).mul_(s1.alpha)
    k_final.add_(tmp, alpha=s1.beta)
    dwh2._symmetrize_(k_final, scratch)
    summaries.append(summarize_tensor("k_final", k_final, eigs=eigs))

    res32 = run_core_impl(a_norm, gram, ws, params=params, apply="fp32")
    q32 = res32.q
    summaries.append(summarize_tensor("q_apply_fp32", q32, eigs=False))
    extras["fp32_chol_stats"] = asdict(res32.stats)

    q16 = None
    if apply_compare:
        res16 = run_core_impl(a_norm, gram, ws, params=params, apply="fp16")
        q16 = res16.q
        summaries.append(summarize_tensor("q_apply_fp16", q16, eigs=False))
        extras["fp16_chol_stats"] = asdict(res16.stats)
        diff = q16.float() - q32.float()
        extras["q16_vs_q32_rel_fro"] = _safe_float(
            torch.linalg.matrix_norm(diff)
            / torch.linalg.matrix_norm(q32.float()).clamp_min(1e-30)
        )

    for label, q in [("fp32", q32)] + ([("fp16", q16)] if q16 is not None else []):
        if q is None:
            continue
        core = MetricsSuite.all_stats(a_norm, q, gram, ws)
        for key, value in core.items():
            extras[f"{label}_{key}"] = value
        extra = _extra_q_diagnostics(a_norm, q, gram, proj, proj_rank)
        for key, value in extra.items():
            extras[f"{label}_{key}"] = value

    return summaries, extras


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--case", default="rank_1_heavy")
    ap.add_argument("--shape", default="16384x4096")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ell0", type=float, default=dwh2.DEFAULT_CONFIG.ell0)
    ap.add_argument(
        "--gram-block-rows", type=int, default=dwh2.DEFAULT_CONFIG.gram_block_rows
    )
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--eigs", action="store_true")
    ap.add_argument("--apply-compare", action="store_true")
    ap.add_argument("--projector-tol-rel", type=float, default=1e-5)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    m, n = [int(x) for x in args.shape.lower().split("x")]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.set_float32_matmul_precision("high" if not args.no_tf32 else "highest")

    a = CaseGenerator.make_case(args.case, m, n, device, args.seed).to(dtype)
    ws = dwh2.DWH2Workspace.allocate(
        min(m, n), device, dtype, block_rows=args.gram_block_rows
    )
    summaries, extras = stage_profile(
        a,
        ws,
        ell0=args.ell0,
        eigs=args.eigs,
        apply_compare=args.apply_compare,
        projector_tol_rel=args.projector_tol_rel,
    )

    print(
        f"Case={args.case} shape={args.shape} dtype={args.dtype} device={device} "
        f"ell0={args.ell0:g} tf32={not args.no_tf32}"
    )
    for s in summaries:
        print_summary(s)

    print("\nDiagnostics:")
    print(json.dumps(extras, indent=2, sort_keys=True))

    if args.json:
        payload = {
            "args": vars(args),
            "summaries": [asdict(s) for s in summaries],
            "extras": extras,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
