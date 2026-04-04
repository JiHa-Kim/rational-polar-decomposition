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
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch

import dwh2

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("bench_profile")


def setup_logging(quiet: bool) -> None:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


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
    chol_calls: int = 0
    chol_shifted_calls: int = 0
    chol_total_retries: int = 0
    chol_max_jitter: float = 0.0


class CaseGenerator:
    """Handles matrix generation for benchmarking."""

    @staticmethod
    def _randn(
        shape, device: torch.device, seed: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype, generator=g)

    @classmethod
    def make_case(
        cls, name: str, m: int, n: int, device: torch.device, seed: int
    ) -> torch.Tensor:
        if name == "gaussian":
            return cls._randn((m, n), device, seed)
        if name == "lognormal_cols":
            x = cls._randn((m, n), device, seed)
            scales = torch.exp(1.5 * cls._randn((n,), device, seed + 1))
            scales = scales / scales.median().clamp_min(1e-8)
            return x * scales[None, :]
        if name == "ar1_cols":
            rho = 0.995
            x = cls._randn((m, n), device, seed)
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
            base = cls._randn((m, k), device, seed)
            reps = (n + k - 1) // k
            tiled = base.repeat(1, reps)[:, :n]
            noise = 1e-3 * cls._randn((m, n), device, seed + 1)
            return tiled + noise
        if name == "lowrank_noise":
            r = min(64, n // 8)
            u = cls._randn((m, r), device, seed)
            v = cls._randn((r, n), device, seed + 1)
            noise = 1e-3 * cls._randn((m, n), device, seed + 2)
            return u @ v + noise
        if name == "ill_conditioned":
            x = cls._randn((m, n), device, seed)
            v = torch.linalg.qr(cls._randn((n, n), device, seed + 1))[0]
            s = torch.logspace(0, -6, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v
        if name == "heavy_tail_t":
            z = cls._randn((m, n), device, seed)
            chi2 = cls._randn((m, n), device, seed + 1).square_()
            tail = cls._randn((m, n), device, seed + 2)
            chi2.addcmul_(tail, tail).mul_(0.5).clamp_min_(1e-4).sqrt_()
            return z / chi2
        if name == "sparse_like":
            base = cls._randn((m, n), device, seed)
            g = torch.Generator(device=device).manual_seed(seed + 1)
            mask = torch.rand((m, n), device=device, generator=g) > 0.95
            return base * mask.float()
        if name == "orthogonal_noisy":
            a = 1e-4 * cls._randn((m, n), device, seed + 1)
            k = min(m, n)
            a[:k, :k].diagonal().add_(1.0)
            return a
        if name == "rank_1_heavy":
            u = cls._randn((m, 1), device, seed)
            v = cls._randn((1, n), device, seed + 1)
            noise = 1e-6 * cls._randn((m, n), device, seed + 2)
            return u @ v + noise
        if name == "adversarial_condition":
            x = cls._randn((m, n), device, seed)
            v = torch.linalg.qr(cls._randn((n, n), device, seed + 1))[0]
            s = torch.linspace(1.0, 1e-7, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v
        raise ValueError(f"Unknown case: {name}")


class MetricsSuite:
    """Calculates orthogonality and polar P metrics."""

    @staticmethod
    def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> None:
        scratch.copy_(a.mT)
        a.add_(scratch).mul_(0.5)

    @classmethod
    def ortho_stats(
        cls, q: torch.Tensor, workspace: dwh2.DWH2Workspace
    ) -> tuple[float, float]:
        transposed = q.shape[0] < q.shape[1]
        Q = q.mT if transposed else q
        m, n = Q.shape
        gram, xbuf, scratch, tmp = (
            workspace.buf,
            workspace.xbuf,
            workspace.scratch,
            workspace.tmp,
        )
        gram.zero_()
        br = int(workspace.block_rows)
        for s in range(0, m, br):
            r = min(br, m - s)
            xbuf[:r].copy_(Q[s : s + r])
            gram.addmm_(xbuf[:r].mT, xbuf[:r])
        cls._symmetrize_(gram, scratch)
        gram.diagonal().sub_(1.0)
        tmp.copy_(gram).mul_(gram)
        fro = torch.sum(tmp, dtype=torch.float64).sqrt() / math.sqrt(float(n))
        return float(fro.item()), float(gram.abs().max().item())

    @classmethod
    def polar_p_stats(
        cls,
        a_norm: torch.Tensor,
        q: torch.Tensor,
        gram_norm: torch.Tensor,
        workspace: dwh2.DWH2Workspace,
    ) -> tuple[float, float]:
        transposed = a_norm.shape[0] < a_norm.shape[1]
        X, Q = (a_norm.mT, q.mT) if transposed else (a_norm, q)
        m, _n = X.shape
        P, xbuf, scratch, tmp = (
            workspace.buf,
            workspace.xbuf,
            workspace.scratch,
            workspace.tmp,
        )
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
        scratch.copy_(P.mT).neg_().add_(P)
        tmp.copy_(scratch).mul_(scratch)
        skew_fro = torch.sum(tmp, dtype=torch.float64).sqrt()
        tmp.copy_(P).mul_(P)
        p_fro = torch.sum(tmp, dtype=torch.float64).sqrt().clamp_min_(1e-30)
        skew_rel = float((skew_fro / p_fro).item())
        cls._symmetrize_(P, scratch)
        torch.mm(P, P, out=tmp)
        tmp.sub_(gram_norm).mul_(tmp)
        err_fro = torch.sum(tmp, dtype=torch.float64).sqrt()
        tmp.copy_(gram_norm).mul_(gram_norm)
        g_fro = torch.sum(tmp, dtype=torch.float64).sqrt().clamp_min_(1e-30)
        return skew_rel, float((err_fro / g_fro).item())


class BenchmarkRunner:
    """Core logic for measuring execution time and collecting metrics."""

    def __init__(self, device: torch.device, args):
        self.device = device
        self.args = args
        self.ws_cache: OrderedDict[tuple[int, str, int | None], dwh2.DWH2Workspace] = (
            OrderedDict()
        )

    def set_fast_matmul(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = not self.args.no_tf32
            torch.backends.cudnn.allow_tf32 = not self.args.no_tf32
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(
            "high" if not self.args.no_tf32 else "highest"
        )

    def measure(
        self, fn: Callable[[], object], trials: int, warmup: int
    ) -> tuple[object, list[float]]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times: list[float] = []
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(trials):
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        return out, times

    def get_workspace(self, n: int, dtype_str: str) -> dwh2.DWH2Workspace:
        key = (n, dtype_str, self.device.index)
        if key in self.ws_cache:
            self.ws_cache.move_to_end(key)
            return self.ws_cache[key]
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
            dtype_str
        ]
        ws = dwh2.DWH2Workspace.allocate(
            n, self.device, dtype, block_rows=dwh2.GRAM_BLOCK_ROWS
        )
        self.ws_cache[key] = ws
        while len(self.ws_cache) > max(0, self.args.ws_cache_max):
            _, old_ws = self.ws_cache.popitem(last=False)
            del old_ws
            gc.collect()
            torch.cuda.empty_cache()
        return ws


class RegressionSuite:
    """Integrates regression checking compactly."""

    @staticmethod
    def check(baseline_path, current_path, time_thresh=0.10, metric_thresh=0.05):
        def load(p):
            res = defaultdict(list)
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        r = json.loads(ln)
                        k = (
                            r["method"],
                            r["case"],
                            r["shape"],
                            r["dtype"],
                            r["tf32"],
                            r["compile"],
                        )
                        res[k].append(r)
            return res

        from collections import defaultdict

        base, curr = load(baseline_path), load(current_path)
        all_keys = sorted(set(base.keys()) | set(curr.keys()))
        regressions, improvements = [], []

        print(
            f"\n{'Method':<6} {'Case':<20} {'Shape':<12} {'Metric':<15} {'Base':<10} {'Curr':<10} {'Diff %':<10}"
        )
        print("-" * 88)

        for k in all_keys:
            if k not in base or k not in curr:
                continue
            for m in ["median_ms", "ortho_fro", "p2_gram_rel_fro"]:
                b_val = statistics.mean([r[m] for r in base[k]])
                c_val = statistics.mean([r[m] for r in curr[k]])
                diff = (
                    (c_val - b_val) / b_val
                    if b_val != 0
                    else (0 if c_val == 0 else float("inf"))
                )
                thresh = time_thresh if m == "median_ms" else metric_thresh
                if diff > thresh:
                    regressions.append((k, m, diff))
                    print(
                        f"{k[0]:<6} {k[1]:<20} {k[2]:<12} {m:<15} {b_val:<10.4f} {c_val:<10.4f} {diff * 100:>+7.1f}% [REGRESSION]"
                    )
                elif diff < -thresh:
                    improvements.append((k, m, diff))
                    print(
                        f"{k[0]:<6} {k[1]:<20} {k[2]:<12} {m:<15} {b_val:<10.4f} {c_val:<10.4f} {diff * 100:>+7.1f}% [IMPROVED]"
                    )

        if regressions:
            print(f"\nDetected {len(regressions)} regressions.")
        elif not improvements:
            print("\nNo significant regressions found.")
        if improvements:
            print(f"Detected {len(improvements)} improvements.")


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


def make_gns_runner(gns_mod, kernel: bool, reset: list[int]):
    try:
        try:
            mod = importlib.import_module("gram_newton_schulz.coefficients")
        except ImportError, ModuleNotFoundError:
            mod = importlib.import_module("newton_schulz.coefficients")
        coefs = getattr(mod, "POLAR_EXPRESS_COEFFICIENTS")
    except ImportError, ModuleNotFoundError, AttributeError:
        coefs = getattr(gns_mod, "POLAR_EXPRESS_COEFFICIENTS", None)
    gns = getattr(gns_mod, "GramNewtonSchulz")(
        ns_use_kernels=kernel,
        ns_coefficients=coefs,
        gram_newton_schulz_reset_iterations=reset,
    )

    def core(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.half()
        t = x.size(-2) > x.size(-1)
        if t:
            x = x.mT.contiguous()
        xb = x.unsqueeze(0)
        ar = getattr(gns, "aspect_ratio_to_use_gram_newton_schulz", 1)
        use_gram = max(xb.shape[-2:]) > ar * min(xb.shape[-2:])
        if hasattr(gns, "_gram_newton_schulz") and hasattr(
            gns, "_standard_newton_schulz"
        ):
            yb = (
                gns._gram_newton_schulz(xb)
                if use_gram
                else gns._standard_newton_schulz(xb)
            )
            y = yb.squeeze(0)
        else:
            y = gns(x)
        return y.mT.contiguous() if t else y

    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--compile", action="store_true", default=False)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["reduce-overhead", "max-autotune"],
    )
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument(
        "--cases",
        default="gaussian,lognormal_cols,ar1_cols,duplicate_cols,lowrank_noise,ill_conditioned,heavy_tail_t,sparse_like,orthogonal_noisy,rank_1_heavy,adversarial_condition",
    )
    ap.add_argument("--shapes", default="16384x4096,8192x4096,4096x4096")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--baseline", help="Baseline JSONL for regression check")
    ap.add_argument(
        "--hard",
        action="store_true",
        help="Run only hard cases (ill_conditioned, lowrank_noise)",
    )
    ap.add_argument("--one-pass", action="store_true")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--ws-cache-max", type=int, default=1)
    ap.add_argument("--gns-path", default="")
    ap.add_argument("--gns-use-kernels", action="store_true")
    ap.add_argument("--gns-reset-iters", default="2")
    ap.add_argument("--no-gns", action="store_true", help="Skip GNS benchmark run")
    ap.add_argument(
        "--load-gns",
        type=str,
        default="stable/gns_baseline.jsonl",
        help="Load GNS results from file if skipped",
    )
    args = ap.parse_args()
    if args.hard:
        args.cases = "ill_conditioned,lowrank_noise"
        args.shapes = "16384x4096"

    setup_logging(args.quiet)
    torch._dynamo.config.capture_scalar_outputs = True
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=r".*torch\._prims_common\.check.*"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    runner = BenchmarkRunner(device, args)
    runner.set_fast_matmul()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    shapes = [
        (int(s.lower().split("x")[0]), int(s.lower().split("x")[1]), s)
        for s in args.shapes.split(",")
        if s.strip()
    ]

    gns_core = None
    if not args.no_gns:
        gns_mod = import_gns(args.gns_path)
        gns_core = make_gns_runner(
            gns_mod,
            args.gns_use_kernels,
            [int(x) for x in args.gns_reset_iters.split(",") if x.strip()],
        )

    dwh2_core = dwh2.dwh2_core

    if args.compile:
        logger.info(f"[compile] Jitting ({args.compile_mode})...")
        dwh2_core = torch.compile(dwh2_core, mode=args.compile_mode, fullgraph=False)
        if gns_core:
            gns_core = torch.compile(gns_core, mode=args.compile_mode, fullgraph=False)

    params = dwh2.get_dwh2_params(dwh2.PAPER_MUON_ELL)
    out_f = open(args.output, "w", encoding="utf-8")

    def run_one(m, n, case, seed, method, fn_base, ws):
        a = CaseGenerator.make_case(case, m, n, device, seed).to(
            {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
                args.dtype
            ]
        )
        a_norm, g_norm = dwh2.normalize_moment_with_small_gram(
            a, workspace=ws, inplace=True
        )
        if method == "dwh2":

            def fn():
                return fn_base(a_norm, g_norm, params=params, workspace=ws)
        else:

            def fn():
                return dwh2.PolarResult(q=fn_base(a_norm))

        out, times = runner.measure(fn, args.trials, args.warmup)
        return out, a_norm, g_norm, statistics.median(times), min(times)

    cb = len(shapes) * len(cases) * len(seeds) * 2
    bar = tqdm(total=cb, desc="bench", disable=args.quiet) if tqdm else None

    results = {}
    try:
        with torch.inference_mode():
            for m, n, s_str in shapes:
                ws = runner.get_workspace(min(m, n), args.dtype)
                for case in cases:
                    for seed in seeds:
                        methods = [("dwh2", dwh2_core)]
                        if not args.no_gns:
                            methods.append(("gns", gns_core))

                        for name, core in methods:
                            out, a_norm, g_norm, med, mn = run_one(
                                m, n, case, seed, name, core, ws
                            )
                            res = {
                                "method": name,
                                "case": case,
                                "shape": s_str,
                                "dtype": args.dtype,
                                "tf32": not args.no_tf32,
                                "compile": args.compile,
                                "trials": args.trials,
                                "warmup": args.warmup,
                                "median_ms": med,
                                "min_ms": mn,
                            }
                            if not args.no_metrics:
                                st = getattr(out, "stats", dwh2.CholStats())
                                ortho, o_max = MetricsSuite.ortho_stats(out.q, ws)
                                skew, p2g = MetricsSuite.polar_p_stats(
                                    a_norm, out.q, g_norm, ws
                                )
                                res.update(
                                    {
                                        "ortho_fro": ortho,
                                        "ortho_max_abs": o_max,
                                        "p_skew_rel_fro": skew,
                                        "p2_gram_rel_fro": p2g,
                                        "chol_calls": st.calls,
                                        "chol_shifted_calls": st.shifted_calls,
                                        "chol_total_retries": st.total_retries,
                                        "chol_max_jitter": st.max_jitter,
                                    }
                                )
                            out_f.write(json.dumps(res, sort_keys=True) + "\n")
                            out_f.flush()
                            if bar:
                                bar.update(1)
                            results[(name, case, s_str, seed)] = res
    finally:
        if bar:
            bar.close()

        # Load stable GNS if skipped but baseline exists
        if args.no_gns and args.load_gns and os.path.exists(args.load_gns):
            logger.info(f"Merging GNS results from {args.load_gns}...")
            with open(args.load_gns, "r", encoding="utf-8") as gf:
                for line in gf:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    # Simple filter to match current session parameters
                    if (
                        r.get("shape") in [s[2] for s in shapes]
                        and r.get("case") in cases
                    ):
                        out_f.write(json.dumps(r, sort_keys=True) + "\n")

        out_f.close()
        if args.baseline:
            RegressionSuite.check(args.baseline, args.output)


if __name__ == "__main__":
    main()
