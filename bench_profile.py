from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import math
import os
import statistics
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable

import torch

import dwh2

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


logger = logging.getLogger("bench_profile")


def setup_logging(log_file: str, quiet: bool) -> None:
    logger.setLevel(logging.DEBUG)

    # File handler (all levels)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)

    # Console handler (INFO and above, clean format)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


def log_status(msg: str):
    if tqdm is not None:
        tqdm.write(msg)
    else:
        logger.info(msg)


def set_fast_matmul(tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def measure(
    fn: Callable[[], object], trials: int, warmup: int
) -> tuple[object, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()

    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return out, times


def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> None:
    scratch.copy_(a.mT)
    a.add_(scratch).mul_(0.5)


@torch.no_grad()
def ortho_stats(q: torch.Tensor, workspace: dwh2.DWH2Workspace) -> tuple[float, float]:
    """Returns (||Q^T Q - I||_F / sqrt(n), max_abs(Q^T Q - I)) with fp32 accumulation."""
    transposed = q.shape[0] < q.shape[1]
    Q = q.mT if transposed else q
    m, n = Q.shape

    gram = workspace.buf
    xbuf = workspace.xbuf
    scratch = workspace.scratch
    tmp = workspace.tmp

    gram.zero_()
    br = int(workspace.block_rows)
    for s in range(0, m, br):
        r = min(br, m - s)
        xbuf[:r].copy_(Q[s : s + r])
        gram.addmm_(xbuf[:r].mT, xbuf[:r])

    _symmetrize_(gram, scratch)
    gram.diagonal().sub_(1.0)

    tmp.copy_(gram)
    tmp.mul_(gram)
    fro = torch.sum(tmp, dtype=torch.float64).sqrt() / math.sqrt(float(n))
    max_abs = gram.abs().max()
    return float(fro.item()), float(max_abs.item())


@torch.no_grad()
def polar_p_stats(
    a_norm: torch.Tensor,
    q: torch.Tensor,
    gram_norm: torch.Tensor,
    workspace: dwh2.DWH2Workspace,
) -> tuple[float, float]:
    """Returns (skew_rel_fro(P), rel_fro(P@P - G)) where G is the small-side Gram of a_norm."""
    transposed = a_norm.shape[0] < a_norm.shape[1]
    X = a_norm.mT if transposed else a_norm
    Q = q.mT if transposed else q
    m, _n = X.shape

    P = workspace.buf
    xbuf = workspace.xbuf
    scratch = workspace.scratch
    tmp = workspace.tmp

    P.zero_()
    br = int(workspace.block_rows)

    qbuf = getattr(workspace, "_metric_qbuf", None)
    if qbuf is None or qbuf.shape != xbuf.shape or qbuf.dtype != torch.float32:
        qbuf = torch.empty_like(xbuf)
        setattr(workspace, "_metric_qbuf", qbuf)

    for s in range(0, m, br):
        r = min(br, m - s)
        xbuf[:r].copy_(X[s : s + r])
        qbuf[:r].copy_(Q[s : s + r])
        P.addmm_(qbuf[:r].mT, xbuf[:r])

    scratch.copy_(P.mT)
    scratch.neg_()
    scratch.add_(P)
    tmp.copy_(scratch)
    tmp.mul_(scratch)
    skew_fro = torch.sum(tmp, dtype=torch.float64).sqrt()

    tmp.copy_(P)
    tmp.mul_(P)
    p_fro = torch.sum(tmp, dtype=torch.float64).sqrt().clamp_min_(1e-30)
    skew_rel = float((skew_fro / p_fro).item())

    _symmetrize_(P, scratch)

    torch.mm(P, P, out=tmp)
    tmp.sub_(gram_norm)
    tmp.mul_(tmp)
    err_fro = torch.sum(tmp, dtype=torch.float64).sqrt()

    tmp.copy_(gram_norm)
    tmp.mul_(gram_norm)
    g_fro = torch.sum(tmp, dtype=torch.float64).sqrt().clamp_min_(1e-30)
    rel = float((err_fro / g_fro).item())

    return skew_rel, rel


@dataclass(frozen=True)
class Record:
    method: str
    case: str
    shape: str
    dtype: str
    tf32: bool
    compile: bool
    trials: int
    warmup: int
    median_ms: float
    min_ms: float

    ortho_fro: float
    ortho_max_abs: float
    p_skew_rel_fro: float
    p2_gram_rel_fro: float

    chol_calls: int
    chol_shifted_calls: int
    chol_total_retries: int
    chol_max_jitter: float


def import_gns(gns_path: str):
    root = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(root, "third_party", "gram-newton-schulz"),
        os.path.join(root, "third_party", "quack"),
    ]
    if gns_path:
        paths.insert(0, os.path.abspath(gns_path))
    for p in paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        return importlib.import_module("gram_newton_schulz")
    except ModuleNotFoundError:
        return importlib.import_module("newton_schulz.gram_newton_schulz")


def make_gns_core_runner(gns_mod, *, use_kernels: bool, reset_iters: list[int]):
    try:
        try:
            coefs_mod = importlib.import_module("gram_newton_schulz.coefficients")
        except ModuleNotFoundError:
            coefs_mod = importlib.import_module("newton_schulz.coefficients")
        coefs = getattr(coefs_mod, "POLAR_EXPRESS_COEFFICIENTS")
    except Exception:
        coefs = getattr(gns_mod, "POLAR_EXPRESS_COEFFICIENTS", None)

    GramNewtonSchulz = getattr(gns_mod, "GramNewtonSchulz")
    gns_obj = GramNewtonSchulz(
        ns_use_kernels=bool(use_kernels),
        ns_coefficients=coefs,
        gram_newton_schulz_reset_iterations=list(reset_iters),
    )

    def core(x_norm_fp16: torch.Tensor) -> torch.Tensor:
        X = x_norm_fp16
        if X.dtype != torch.float16:
            X = X.half()

        should_transpose = X.size(-2) > X.size(-1)
        if should_transpose:
            X = X.mT.contiguous()

        Xb = X.unsqueeze(0)
        ar = getattr(gns_obj, "aspect_ratio_to_use_gram_newton_schulz", 1)
        use_gram = max(Xb.shape[-2:]) > ar * min(Xb.shape[-2:])

        if hasattr(gns_obj, "_gram_newton_schulz") and hasattr(
            gns_obj, "_standard_newton_schulz"
        ):
            Yb = (
                gns_obj._gram_newton_schulz(Xb)
                if use_gram
                else gns_obj._standard_newton_schulz(Xb)
            )
            Y = Yb.squeeze(0)
        else:
            Y = gns_obj(X)

        if should_transpose:
            Y = Y.mT.contiguous()
        return Y

    return core


def cast_dtype(a: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp16":
        return a.half()
    if dtype == "bf16":
        return a.bfloat16()
    if dtype == "fp32":
        return a.float()
    raise ValueError(dtype)


def _randn(
    shape, *, device: torch.device, seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=dtype, generator=g)


def make_case(
    name: str, m: int, n: int, *, device: torch.device, seed: int
) -> torch.Tensor:
    if name == "gaussian":
        return _randn((m, n), device=device, seed=seed)
    if name == "lognormal_cols":
        x = _randn((m, n), device=device, seed=seed)
        scales = torch.exp(1.5 * _randn((n,), device=device, seed=seed + 1))
        scales = scales / scales.median().clamp_min(1e-8)
        return x * scales[None, :]
    if name == "ar1_cols":
        rho = 0.995
        x = _randn((m, n), device=device, seed=seed)
        coeff = math.sqrt(max(1.0 - rho * rho, 0.0))
        powers = torch.pow(
            torch.tensor(rho, device=device, dtype=x.dtype),
            torch.arange(n, device=device, dtype=x.dtype),
        )
        scale = torch.full((n,), coeff, device=device, dtype=x.dtype)
        scale[0] = 1.0
        scale[1:] /= powers[1:]
        a = torch.cumsum(x * scale[None, :], dim=1)
        a.mul_(powers[None, :])
        return a
    if name == "duplicate_cols":
        k = max(64, n // 16)
        base = _randn((m, k), device=device, seed=seed)
        reps = (n + k - 1) // k
        tiled = base.repeat(1, reps)[:, :n]
        noise = 1e-3 * _randn((m, n), device=device, seed=seed + 1)
        return tiled + noise
    if name == "lowrank_noise":
        r = min(64, n // 8)
        u = _randn((m, r), device=device, seed=seed)
        v = _randn((r, n), device=device, seed=seed + 1)
        noise = 1e-3 * _randn((m, n), device=device, seed=seed + 2)
        return u @ v + noise
    if name == "ill_conditioned":
        x = _randn((m, n), device=device, seed=seed)
        v = torch.linalg.qr(_randn((n, n), device=device, seed=seed + 1))[0]
        s = torch.logspace(0, -6, steps=n, device=device, dtype=x.dtype)
        return (x * s[None, :]) @ v
    if name == "heavy_tail_t":
        z = _randn((m, n), device=device, seed=seed)
        chi2 = _randn((m, n), device=device, seed=seed + 1)
        chi2.square_()
        tail = _randn((m, n), device=device, seed=seed + 2)
        chi2.addcmul_(tail, tail)
        chi2.mul_(0.5).clamp_min_(1e-4).sqrt_()
        return z / chi2
    if name == "sparse_like":
        base = _randn((m, n), device=device, seed=seed)
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1)
        mask = torch.rand((m, n), device=device, generator=g) > 0.95
        return base * mask.float()
    if name == "orthogonal_noisy":
        a = 1e-4 * _randn((m, n), device=device, seed=seed + 1)
        k = min(m, n)
        a[:k, :k].diagonal().add_(1.0)
        return a
    if name == "rank_1_heavy":
        u = _randn((m, 1), device=device, seed=seed)
        v = _randn((1, n), device=device, seed=seed + 1)
        noise = 1e-6 * _randn((m, n), device=device, seed=seed + 2)
        return u @ v + noise
    if name == "adversarial_condition":
        x = _randn((m, n), device=device, seed=seed)
        v = torch.linalg.qr(_randn((n, n), device=device, seed=seed + 1))[0]
        s = torch.linspace(1.0, 1e-7, steps=n, device=device, dtype=x.dtype)
        return (x * s[None, :]) @ v
    raise ValueError(name)


def _make_bar(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def _bar_update(bar, *, shape: str, case: str, seed: int, method: str, med: float, mn: float) -> None:
    if bar is None:
        logger.info(f"[{method}] {shape} {case} seed={seed} med={med:.2f}ms")
        return
    bar.set_postfix_str(f"{method} {shape} {case} med={med:.1f}ms")
    bar.update(1)


def _bar_update_metrics(bar, *, shape: str, case: str, seed: int, method: str, ortho: float, p2g: float) -> None:
    if bar is None:
        logger.info(f"[{method}] {shape} {case} seed={seed} ortho={ortho:.2e}")
        return
    bar.set_postfix_str(f"{method} {shape} {case} ortho={ortho:.1e}")
    bar.update(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)

    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"]
    )

    ap.add_argument(
        "--cases",
        type=str,
        default="gaussian,lognormal_cols,ar1_cols,duplicate_cols,lowrank_noise,ill_conditioned,heavy_tail_t,"
        "sparse_like,orthogonal_noisy,rank_1_heavy,adversarial_condition",
    )
    ap.add_argument(
        "--shapes",
        type=str,
        default="16384x4096,8192x4096,4096x4096,4096x8192,32768x2048,65536x1024",
    )

    ap.add_argument("--gns-path", type=str, default="")
    ap.add_argument("--gns-use-kernels", action="store_true")
    ap.add_argument("--gns-reset-iters", type=str, default="2")

    ap.add_argument("--ell0", type=float, default=dwh2.PAPER_MUON_ELL)
    ap.add_argument("--norm-eps", type=float, default=dwh2.NORM_EPS)
    ap.add_argument("--norm-safety", type=float, default=dwh2.NORM_SAFETY)

    ap.add_argument("--output", type=str, default="results.jsonl")
    ap.add_argument("--log", type=str, default="bench_profile.log")
    ap.add_argument("--quiet", action="store_true", help="Minimize console output.")
    ap.add_argument(
        "--no-metrics",
        action="store_true",
        help="Only measure runtime; skip quality metrics.",
    )
    ap.add_argument(
        "--one-pass",
        action="store_true",
        help="If set, compute speed+metrics per config. Default is 2-pass (all timing, then all metrics).",
    )

    ap.add_argument(
        "--ws-cache-max",
        type=int,
        default=1,
        help="Max cached DWH2 workspaces. 1 is safest on small GPUs.",
    )
    ap.add_argument(
        "--empty-cache-threshold",
        type=float,
        default=0.90,
        help="If reserved/total exceeds this, run torch.cuda.empty_cache().",
    )
    ap.add_argument(
        "--no-empty-cache",
        action="store_true",
        help="Disable torch.cuda.empty_cache() pressure relief.",
    )

    args = ap.parse_args()
    setup_logging(args.log, args.quiet)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark is tuned for CUDA GPUs.")
    if device.index is not None:
        torch.cuda.set_device(device.index)

    set_fast_matmul(not args.no_tf32)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    shapes: list[tuple[int, int, str]] = []
    for s in [x.strip() for x in args.shapes.split(",") if x.strip()]:
        m, n = s.lower().split("x")
        shapes.append((int(m), int(n), s))
    reset_iters = [int(x) for x in args.gns_reset_iters.split(",") if x.strip()]

    two_pass = (not args.one_pass) and (not args.no_metrics)
    timing_path = args.output + ".timing.jsonl"

    logger.info(
        f"[start] cfgs={len(shapes) * len(cases) * len(seeds)} seeds={len(seeds)} warmup={args.warmup} trials={args.trials}"
    )

    gns_mod = import_gns(args.gns_path)
    gns_core = make_gns_core_runner(
        gns_mod, use_kernels=bool(args.gns_use_kernels), reset_iters=reset_iters
    )

    empty_cache_enabled = not bool(args.no_empty_cache)

    def maybe_empty_cache(force: bool = False) -> None:
        if not empty_cache_enabled or not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return
        try:
            _, total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved()
            if total > 0 and float(reserved) / float(total) >= float(args.empty_cache_threshold):
                torch.cuda.empty_cache()
        except Exception:
            return

    ws_cache: OrderedDict[tuple[int, str, int | None], dwh2.DWH2Workspace] = OrderedDict()

    def get_ws(n_small: int, dtype_str: str) -> dwh2.DWH2Workspace:
        key = (
            int(n_small),
            str(dtype_str),
            int(device.index) if device.index is not None else None,
        )
        ws = ws_cache.get(key)
        if ws is not None:
            ws_cache.move_to_end(key)
            return ws

        out_dtype = (
            torch.float16
            if dtype_str == "fp16"
            else (torch.bfloat16 if dtype_str == "bf16" else torch.float32)
        )
        ws = dwh2.DWH2Workspace.allocate(
            n_small, device, out_dtype, block_rows=dwh2.GRAM_BLOCK_ROWS
        )
        ws_cache[key] = ws
        ws_cache.move_to_end(key)

        while len(ws_cache) > int(max(0, args.ws_cache_max)):
            _, old_ws = ws_cache.popitem(last=False)
            del old_ws
            gc.collect()
            maybe_empty_cache(force=True)

        return ws

    out_f = open(args.output, "w", encoding="utf-8")
    timing_f = open(timing_path, "w", encoding="utf-8") if two_pass else None

    def write_line(fh, rec: Record) -> None:
        line = json.dumps(asdict(rec), sort_keys=True)
        fh.write(line + "\n")
        fh.flush()
        logger.debug(f"Wrote result: {rec.method} {rec.case} {rec.shape}")

    timing_bar = _make_bar(len(shapes) * len(cases) * len(seeds) * 2, "timing")
    metrics_bar = None

    try:
        timing: dict[tuple[str, str, str, int], tuple[float, float]] = {}

        with torch.inference_mode():
            for m, n, shape_str in shapes:
                for case_name in cases:
                    for seed in seeds:
                        maybe_empty_cache(force=False)

                        a = cast_dtype(
                            make_case(case_name, m, n, device=device, seed=seed),
                            args.dtype,
                        )
                        ws = get_ws(min(a.shape), args.dtype)

                        a_norm, gram_norm = dwh2.normalize_moment_with_small_gram(
                            a,
                            eps=float(args.norm_eps),
                            safety=float(args.norm_safety),
                            workspace=ws,
                        )
                        del a

                        def run_dwh2_core(
                            a_norm_t: torch.Tensor = a_norm,
                            gram_norm_t: torch.Tensor = gram_norm,
                            ws_t: dwh2.DWH2Workspace = ws,
                        ) -> dwh2.PolarResult:
                            return dwh2.dwh2_core(
                                a_norm_t,
                                gram_norm_t,
                                ell0=float(args.ell0),
                                workspace=ws_t,
                            )

                        def run_gns_core(a_norm_t: torch.Tensor = a_norm) -> dwh2.PolarResult:
                            y = gns_core(a_norm_t.half())
                            return dwh2.PolarResult(q=y)

                        f_dwh2 = run_dwh2_core
                        f_gns = run_gns_core
                        if args.compile:
                            f_dwh2 = torch.compile(f_dwh2, mode="max-autotune")  # type: ignore
                            f_gns = torch.compile(f_gns, mode="max-autotune")  # type: ignore

                        for method, fn in [("dwh2", f_dwh2), ("gns", f_gns)]:
                            out, times = measure(fn, int(args.trials), int(args.warmup))
                            key = (method, case_name, shape_str, int(seed))
                            med = float(statistics.median(times))
                            mn = float(min(times))
                            timing[key] = (med, mn)
                            _bar_update(
                                timing_bar,
                                shape=shape_str,
                                case=case_name,
                                seed=seed,
                                method=method,
                                med=med,
                                mn=mn,
                            )

                            if args.one_pass:
                                q = out.q
                                st = out.stats
                                torch.cuda.synchronize()
                                ortho_f, ortho_max = ortho_stats(q, ws)
                                p_skew, p2g = polar_p_stats(a_norm, q, gram_norm, ws)
                                torch.cuda.synchronize()
                                rec = Record(
                                    method=method,
                                    case=case_name,
                                    shape=shape_str,
                                    dtype=args.dtype,
                                    tf32=bool(not args.no_tf32),
                                    compile=bool(args.compile),
                                    trials=int(args.trials),
                                    warmup=int(args.warmup),
                                    median_ms=med,
                                    min_ms=mn,
                                    ortho_fro=ortho_f,
                                    ortho_max_abs=ortho_max,
                                    p_skew_rel_fro=p_skew,
                                    p2_gram_rel_fro=p2g,
                                    chol_calls=int(st.calls),
                                    chol_shifted_calls=int(st.shifted_calls),
                                    chol_total_retries=int(st.total_retries),
                                    chol_max_jitter=float(st.max_jitter),
                                )
                                write_line(out_f, rec)
                            elif args.no_metrics:
                                st = out.stats
                                rec = Record(
                                    method=method,
                                    case=case_name,
                                    shape=shape_str,
                                    dtype=args.dtype,
                                    tf32=bool(not args.no_tf32),
                                    compile=bool(args.compile),
                                    trials=int(args.trials),
                                    warmup=int(args.warmup),
                                    median_ms=med,
                                    min_ms=mn,
                                    ortho_fro=float("nan"),
                                    ortho_max_abs=float("nan"),
                                    p_skew_rel_fro=float("nan"),
                                    p2_gram_rel_fro=float("nan"),
                                    chol_calls=int(st.calls),
                                    chol_shifted_calls=int(st.shifted_calls),
                                    chol_total_retries=int(st.total_retries),
                                    chol_max_jitter=float(st.max_jitter),
                                )
                                write_line(out_f, rec)
                            elif timing_f is not None:
                                st = out.stats if hasattr(out, "stats") else dwh2.CholStats()
                                rec = Record(
                                    method=method,
                                    case=case_name,
                                    shape=shape_str,
                                    dtype=args.dtype,
                                    tf32=bool(not args.no_tf32),
                                    compile=bool(args.compile),
                                    trials=int(args.trials),
                                    warmup=int(args.warmup),
                                    median_ms=med,
                                    min_ms=mn,
                                    ortho_fro=float("nan"),
                                    ortho_max_abs=float("nan"),
                                    p_skew_rel_fro=float("nan"),
                                    p2_gram_rel_fro=float("nan"),
                                    chol_calls=int(st.calls),
                                    chol_shifted_calls=int(st.shifted_calls),
                                    chol_total_retries=int(st.total_retries),
                                    chol_max_jitter=float(st.max_jitter),
                                )
                                write_line(timing_f, rec)

                        del a_norm
                        del gram_norm
                        gc.collect()
                        maybe_empty_cache(force=False)

            if args.one_pass or args.no_metrics:
                return

            if timing_bar is not None:
                timing_bar.close()

            metrics_bar = _make_bar(len(shapes) * len(cases) * len(seeds) * 2, "metrics")
            for m, n, shape_str in shapes:
                for case_name in cases:
                    for seed in seeds:
                        maybe_empty_cache(force=False)

                        a = cast_dtype(
                            make_case(case_name, m, n, device=device, seed=seed),
                            args.dtype,
                        )
                        ws = get_ws(min(a.shape), args.dtype)
                        a_norm, gram_norm = dwh2.normalize_moment_with_small_gram(
                            a,
                            eps=float(args.norm_eps),
                            safety=float(args.norm_safety),
                            workspace=ws,
                        )
                        del a

                        def run_dwh2_once(
                            a_norm_t: torch.Tensor = a_norm,
                            gram_norm_t: torch.Tensor = gram_norm,
                            ws_t: dwh2.DWH2Workspace = ws,
                        ) -> dwh2.PolarResult:
                            return dwh2.dwh2_core(
                                a_norm_t,
                                gram_norm_t,
                                ell0=float(args.ell0),
                                workspace=ws_t,
                            )

                        def run_gns_once(a_norm_t: torch.Tensor = a_norm) -> dwh2.PolarResult:
                            y = gns_core(a_norm_t.half())
                            return dwh2.PolarResult(q=y)

                        for method, fn in [("dwh2", run_dwh2_once), ("gns", run_gns_once)]:
                            out = fn()
                            q = out.q
                            st = out.stats

                            torch.cuda.synchronize()
                            ortho_f, ortho_max = ortho_stats(q, ws)
                            p_skew, p2g = polar_p_stats(a_norm, q, gram_norm, ws)
                            torch.cuda.synchronize()

                            tkey = (method, case_name, shape_str, int(seed))
                            med, mn = timing[tkey]
                            _bar_update_metrics(
                                metrics_bar,
                                shape=shape_str,
                                case=case_name,
                                seed=seed,
                                method=method,
                                ortho=ortho_f,
                                p2g=p2g,
                            )

                            rec = Record(
                                method=method,
                                case=case_name,
                                shape=shape_str,
                                dtype=args.dtype,
                                tf32=bool(not args.no_tf32),
                                compile=bool(args.compile),
                                trials=int(args.trials),
                                warmup=int(args.warmup),
                                median_ms=float(med),
                                min_ms=float(mn),
                                ortho_fro=ortho_f,
                                ortho_max_abs=ortho_max,
                                p_skew_rel_fro=p_skew,
                                p2_gram_rel_fro=p2g,
                                chol_calls=int(st.calls),
                                chol_shifted_calls=int(st.shifted_calls),
                                chol_total_retries=int(st.total_retries),
                                chol_max_jitter=float(st.max_jitter),
                            )
                            write_line(out_f, rec)

                        del a_norm
                        del gram_norm
                        gc.collect()
                        maybe_empty_cache(force=False)
    finally:
        if timing_bar is not None:
            timing_bar.close()
        if metrics_bar is not None:
            metrics_bar.close()
        if timing_f is not None:
            timing_f.close()
        out_f.close()


if __name__ == "__main__":
    main()
