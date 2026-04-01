from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .dwh2 import DWH2Result, dwh2
from .pe5 import PAPER_MUON_ELL, PAPER_NORM_EPS, PE5Result, pe5, pe5_coefficients
from .precond import CholStats


@dataclass(frozen=True)
class Case:
    name: str
    a: torch.Tensor


@dataclass(frozen=True)
class Record:
    case: str
    method: str
    is_stress: bool
    trials: int
    runtime_ms_median: float
    runtime_ms_min: float
    ortho_fro: float
    q_fro_error: Optional[float]
    objective_ratio: Optional[float]
    objective_proj: Optional[float]
    chol_calls: int
    chol_shifted_calls: int
    chol_total_retries: int
    chol_max_jitter: float
    diag_floored: int


def set_fast_matmul(tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def _randn(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=dtype, generator=g)


def make_case(
    name: str, rows: int, cols: int, *, device: torch.device, seed: int
) -> Case:
    m, n = rows, cols
    if name == "gaussian":
        a = _randn((m, n), device=device, seed=seed)
    elif name == "lognormal_cols":
        x = _randn((m, n), device=device, seed=seed)
        scales = torch.exp(1.5 * _randn((n,), device=device, seed=seed + 1))
        scales = scales / scales.median().clamp_min(1e-8)
        a = x * scales[None, :]
    elif name == "ar1_cols":
        rho = 0.995
        x = _randn((m, n), device=device, seed=seed)
        a = torch.empty_like(x)
        a[:, 0] = x[:, 0]
        coeff = math.sqrt(max(1.0 - rho * rho, 0.0))
        for j in range(1, n):
            a[:, j] = rho * a[:, j - 1] + coeff * x[:, j]
    elif name == "duplicate_cols":
        k = max(64, n // 16)
        base = _randn((m, k), device=device, seed=seed)
        reps = math.ceil(n / k)
        tiled = base.repeat(1, reps)[:, :n]
        noise = 1e-3 * _randn((m, n), device=device, seed=seed + 1)
        a = tiled + noise
    elif name == "lowrank_noise":
        r = min(64, n // 8)
        u = _randn((m, r), device=device, seed=seed)
        v = _randn((r, n), device=device, seed=seed + 1)
        noise = 1e-3 * _randn((m, n), device=device, seed=seed + 2)
        a = u @ v + noise
    elif name == "ill_conditioned":
        # Systematically decaying singular values via fast column scaling and mixing
        x = _randn((m, n), device=device, seed=seed)
        v = _randn((n, n), device=device, seed=seed + 1)
        v = torch.linalg.qr(v)[0]  # qr on much smaller nxn
        s = torch.logspace(0, -6, steps=n, device=device, dtype=x.dtype)
        a = (x * s[None, :]) @ v
    elif name == "heavy_tail_t":
        # Student-t distribution with 2 degress of freedom (heavy tails/outliers)
        z = _randn((m, n), device=device, seed=seed)
        chi2 = _randn((m, n), device=device, seed=seed + 1)**2 + _randn((m, n), device=device, seed=seed + 2)**2
        a = z / torch.sqrt(chi2 / 2.0).clamp_min(1e-4)
    elif name == "sparse_like":
        # 95% sparsity pseudo-sparse matrix
        base = _randn((m, n), device=device, seed=seed)
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1)
        mask = torch.rand((m, n), device=device, generator=g) > 0.95
        a = base * mask.float()
    elif name == "orthogonal_noisy":
        # Nearly orthogonal columns (tall skiny identity-like)
        n_min = min(m, n)
        a = torch.zeros((m, n), device=device, dtype=torch.float32)
        a[:n_min, :n_min] = torch.eye(n_min, device=device, dtype=torch.float32)
        noise = 1e-4 * _randn((m, n), device=device, seed=seed + 1)
        a = a + noise
    elif name == "rank_1_heavy":
        # Extreme low-rank + some noise
        u = _randn((m, 1), device=device, seed=seed)
        v = _randn((1, n), device=device, seed=seed + 1)
        noise = 1e-6 * _randn((m, n), device=device, seed=seed + 2)
        a = u @ v + noise
    elif name == "adversarial_condition":
        # Exact condition number bound via SVD on small random matrix + mixing
        x = _randn((m, n), device=device, seed=seed)
        v = torch.linalg.qr(_randn((n, n), device=device, seed=seed + 1))[0]
        s = torch.linspace(1.0, 1e-7, steps=n, device=device, dtype=x.dtype)
        a = (x * s[None, :]) @ v
    else:
        raise ValueError(f"unknown case {name}")
    return Case(name=name, a=a)


def normalize_fro(a: torch.Tensor) -> torch.Tensor:
    return a / (torch.linalg.matrix_norm(a, ord="fro") + PAPER_NORM_EPS)


def polar_reference(a: torch.Tensor) -> Tuple[torch.Tensor, float]:
    transposed = False
    x = a
    if x.shape[0] < x.shape[1]:
        x = x.mT.contiguous()
        transposed = True
    x64 = x.to(torch.float64)
    g = x64.mT @ x64
    g = 0.5 * (g + g.mT)
    evals, evecs = torch.linalg.eigh(g)
    cutoff = float(torch.finfo(torch.float64).eps) * max(float(evals.max().item()), 1.0)
    evals = evals.clamp_min(cutoff)
    inv_sqrt = (evecs * torch.rsqrt(evals)[None, :]) @ evecs.mT
    q64 = x64 @ inv_sqrt
    if transposed:
        q64 = q64.mT.contiguous()
    opt_obj = float(torch.sum(q64 * a.to(torch.float64)).item())
    return q64.to(a.dtype), opt_obj


def measure(
    fn: Callable[[], object], trials: int, warmup: int
) -> Tuple[object, List[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    times: List[float] = []
    if torch.cuda.is_available():
        probe = out.q if hasattr(out, "q") else out
        if isinstance(probe, torch.Tensor) and probe.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(trials):
                torch.cuda.synchronize()
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
            return out, times
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def summarize(
    *,
    case: Case,
    q: torch.Tensor,
    ref_q: Optional[torch.Tensor],
    ref_obj: Optional[float],
    times: Sequence[float],
    method: str,
    trials: int,
    stats: CholStats,
    is_stress: bool,
) -> Record:
    n = q.shape[1]
    eye = torch.eye(n, device=q.device, dtype=q.dtype)
    gram = q.mT @ q
    ortho_fro = float(
        torch.linalg.matrix_norm(gram - eye, ord="fro").item() / math.sqrt(n)
    )
    q_fro_error = None
    objective_ratio = None
    objective_proj = None
    if ref_q is not None and ref_obj is not None:
        q_fro_error = float(
            torch.linalg.matrix_norm(q - ref_q, ord="fro").item() / math.sqrt(n)
        )
        objective = float(torch.sum(q * case.a).item())
        objective_ratio = float(objective / ref_obj)

        # Projected objective: Q_proj = Q (Q^T Q)^{-1/2} = U V^T from SVD(Q)
        try:
            q_dp = q.to(torch.float64)
            u, _, vh = torch.linalg.svd(q_dp, full_matrices=False)
            q_proj = u @ vh
            objective_proj_val = float(torch.sum(q_proj * case.a.to(torch.float64)).item())
            objective_proj = float(objective_proj_val / ref_obj)
        except Exception:
            objective_proj = None

    return Record(
        case=case.name,
        method=method,
        is_stress=is_stress,
        trials=trials,
        runtime_ms_median=float(torch.tensor(times).median().item()),
        runtime_ms_min=float(min(times)),
        ortho_fro=ortho_fro,
        q_fro_error=q_fro_error,
        objective_ratio=objective_ratio,
        objective_proj=objective_proj,
        chol_calls=stats.calls,
        chol_shifted_calls=stats.shifted_calls,
        chol_total_retries=stats.total_retries,
        chol_max_jitter=stats.max_jitter,
        diag_floored=stats.diag_floored,
    )


def line_writer(path: str | None):
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", buffering=1, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal realistic benchmark: DWH2 vs PE5 on tall matrices."
    )
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile the algorithmic kernels with torch.compile")
    parser.add_argument("--ell0", type=float, default=PAPER_MUON_ELL)
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip float64 reference polar computation.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "gaussian",
            "lognormal_cols",
            "ar1_cols",
            "duplicate_cols",
            "lowrank_noise",
            "ill_conditioned",
            "heavy_tail_t",
            "sparse_like",
            "orthogonal_noisy",
            "rank_1_heavy",
            "adversarial_condition",
        ],
    )
    parser.add_argument("--methods", nargs="+", default=["dwh2", "pe5"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("runs") / time.strftime("%Y%m%d_%H%M%S") / "results.jsonl"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_fast_matmul(args.tf32)
    out_f = line_writer(args.output)

    try:
        with torch.no_grad():
            coeffs = pe5_coefficients(ell0=args.ell0, steps=5)

            dwh2_fn = dwh2
            pe5_fn = pe5
            if args.compile:
                # Compile main kernels, using fullgraph=False as Cholesky error checks cause harmless Python syncs
                dwh2_fn = torch.compile(dwh2, mode="max-autotune") # type: ignore
                pe5_fn = torch.compile(pe5, mode="max-autotune") # type: ignore

            STRESS_CASES = {"duplicate_cols", "lowrank_noise"}
            STRESS_ELL0 = 1e-6

            for i, case_name in enumerate(args.cases):
                is_stress = case_name in STRESS_CASES
                case_ell0 = STRESS_ELL0 if is_stress else args.ell0

                case = make_case(
                    case_name,
                    args.rows,
                    args.cols,
                    device=device,
                    seed=args.seed + 1000 * i,
                )
                a = normalize_fro(case.a).contiguous()
                case = Case(name=case.name, a=a)

                ref_q = None
                ref_obj = None
                if not args.no_reference:
                    ref_q, ref_obj = polar_reference(case.a)

                # Pre-calculate stress coefficients if needed
                case_coeffs = coeffs
                if is_stress and case_ell0 != args.ell0:
                    case_coeffs = pe5_coefficients(ell0=case_ell0, steps=5)

                methods: Dict[str, Callable[[], object]] = {
                    "dwh2": lambda a=case.a, ell=case_ell0: dwh2_fn(a, ell0=ell, tf32=args.tf32),
                    "pe5": lambda a=case.a, cs=case_coeffs, ell=case_ell0: pe5_fn(
                        a, ell0=ell, coeffs=cs
                    ),
                }

                for method_name in args.methods:
                    out, times = measure(methods[method_name], args.trials, args.warmup)
                    if isinstance(out, DWH2Result):
                        q = out.q
                        stats = out.stats
                    elif isinstance(out, PE5Result):
                        q = out.q
                        stats = CholStats()
                    else:
                        raise TypeError(f"unexpected output type: {type(out)}")

                    record = summarize(
                        case=case,
                        q=q,
                        ref_q=ref_q,
                        ref_obj=ref_obj,
                        times=times,
                        method=method_name,
                        trials=args.trials,
                        stats=stats,
                        is_stress=is_stress,
                    )
                    row = json.dumps(asdict(record), sort_keys=True)
                    if not args.quiet:
                        print(row, flush=True)
                    if out_f is not None:
                        out_f.write(row + "\n")
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
