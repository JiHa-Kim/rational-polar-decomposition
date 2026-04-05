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
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable

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
    median_ms: float = float("nan")
    min_ms: float = float("nan")
    ortho_proj: float = float("nan")
    ortho_supp: float = float("nan")
    p_skew_rel_fro: float = float("nan")
    p2_gram_rel_fro: float = float("nan")
    rec_resid: float = float("nan")
    stable_rank: float = float("nan")
    chol_calls: int = 0
    chol_shifted_calls: int = 0
    chol_total_retries: int = 0
    chol_max_jitter: float = 0.0


class CaseGenerator:
    @staticmethod
    def _randn(shape, device, seed: int, dtype):
        import torch

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype, generator=g)

    @classmethod
    def make_case(cls, name: str, m: int, n: int, device, seed: int):
        import torch

        if name == "gaussian":
            return cls._randn((m, n), device, seed, torch.float32)
        if name == "lognormal_cols":
            x = cls._randn((m, n), device, seed, torch.float32)
            scales = torch.exp(1.5 * cls._randn((n,), device, seed + 1, torch.float32))
            scales = scales / scales.median().clamp_min(1e-8)
            return x * scales[None, :]
        if name == "ar1_cols":
            rho = 0.995
            x = cls._randn((m, n), device, seed, torch.float32)
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
            base = cls._randn((m, k), device, seed, torch.float32)
            reps = (n + k - 1) // k
            tiled = base.repeat(1, reps)[:, :n]
            noise = 1e-3 * cls._randn((m, n), device, seed + 1, torch.float32)
            return tiled + noise
        if name == "lowrank_noise":
            r = min(64, n // 8)
            u = cls._randn((m, r), device, seed, torch.float32)
            v = cls._randn((r, n), device, seed + 1, torch.float32)
            noise = 1e-3 * cls._randn((m, n), device, seed + 2, torch.float32)
            return u @ v + noise
        if name == "ill_conditioned":
            x = cls._randn((m, n), device, seed, torch.float32)
            v = torch.linalg.qr(cls._randn((n, n), device, seed + 1, torch.float32))[0]
            s = torch.logspace(0, -6, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v
        if name == "heavy_tail_t":
            z = cls._randn((m, n), device, seed, torch.float32)
            chi2 = cls._randn((m, n), device, seed + 1, torch.float32).square_()
            tail = cls._randn((m, n), device, seed + 2, torch.float32)
            chi2.addcmul_(tail, tail).mul_(0.5).clamp_min_(1e-4).sqrt_()
            return z / chi2
        if name == "sparse_like":
            base = cls._randn((m, n), device, seed, torch.float32)
            import torch

            g = torch.Generator(device=device)
            g.manual_seed(seed + 1)
            mask = torch.rand((m, n), device=device, generator=g) > 0.95
            return base * mask.float()
        if name == "orthogonal_noisy":
            a = 1e-4 * cls._randn((m, n), device, seed + 1, torch.float32)
            k = min(m, n)
            a[:k, :k].diagonal().add_(1.0)
            return a
        if name == "rank_1_heavy":
            u = cls._randn((m, 1), device, seed, torch.float32)
            v = cls._randn((1, n), device, seed + 1, torch.float32)
            noise = 1e-6 * cls._randn((m, n), device, seed + 2, torch.float32)
            return u @ v + noise
        if name == "adversarial_condition":
            x = cls._randn((m, n), device, seed, torch.float32)
            v = torch.linalg.qr(cls._randn((n, n), device, seed + 1, torch.float32))[0]
            s = torch.linspace(1.0, 1e-7, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v
        raise ValueError(f"Unknown case: {name}")


class MetricsSuite:
    @staticmethod
    def _symmetrize_(a, scratch) -> None:
        scratch.copy_(a.mT)
        a.add_(scratch).mul_(0.5)

    @classmethod
    def all_stats(
        cls,
        a_norm,
        q,
        gram_norm,
        workspace,
    ) -> dict[str, float]:
        import torch

        # Handle shapes
        transposed = a_norm.shape[0] < a_norm.shape[1]
        X, Q = (a_norm.mT, q.mT) if transposed else (a_norm, q)
        m, n = X.shape

        eps = 1e-30
        S = workspace.gram  # repurposed as S = Q^T Q
        H = workspace.buf  # repurposed as H = Q^T X
        xbuf = workspace.xbuf
        scratch = workspace.scratch
        tmp = workspace.tmp

        S.zero_()
        H.zero_()
        br = int(workspace.block_rows)

        # Prepare qbuf
        qbuf = getattr(workspace, "_metric_qbuf", None)
        if qbuf is None or qbuf.shape != xbuf.shape or qbuf.dtype != torch.float32:
            qbuf = torch.empty_like(xbuf)
            setattr(workspace, "_metric_qbuf", qbuf)

        # Batch form S and H to minimize passes
        for s in range(0, m, br):
            r = min(br, m - s)
            xbuf[:r].copy_(X[s : s + r])
            qbuf[:r].copy_(Q[s : s + r])
            S.addmm_(qbuf[:r].mT, qbuf[:r])
            H.addmm_(qbuf[:r].mT, xbuf[:r])

        cls._symmetrize_(S, scratch)

        # 1. Projector defect (e_proj)
        # e_proj = |S^2 - S|_F / (|S|_F + eps)
        torch.mm(S, S, out=tmp)
        tmp.sub_(S)
        e_proj_num = torch.sum(tmp * tmp, dtype=torch.float64).sqrt()
        e_proj_den = torch.sum(S * S, dtype=torch.float64).sqrt().clamp_min_(eps)
        e_proj = float((e_proj_num / e_proj_den).item())

        # 2. Support consistency (e_supp)
        # e_supp = |(I - S)G|_F / (|G|_F + eps)
        tmp.copy_(S).neg_().diagonal().add_(1.0)  # (I - S)
        torch.mm(tmp, gram_norm, out=scratch)
        e_supp_num = torch.sum(scratch * scratch, dtype=torch.float64).sqrt()
        e_supp_den = (
            torch.sum(gram_norm * gram_norm, dtype=torch.float64).sqrt().clamp_min_(eps)
        )
        e_supp = float((e_supp_num / e_supp_den).item())

        # 3. Skewness and P^2 error (existing p_skew and p2_gram)
        # P = sym(H)
        P = workspace.rhs  # repurposed
        P.copy_(H.mT).add_(H).mul_(0.5)

        skew = scratch.copy_(H).sub_(P)  # H - P
        e_skew_num = torch.sum(skew * skew, dtype=torch.float64).sqrt()
        e_skew_den = torch.sum(P * P, dtype=torch.float64).sqrt().clamp_min_(eps)
        e_skew = float((e_skew_num / e_skew_den).item())

        torch.mm(P, P, out=tmp)
        tmp.sub_(gram_norm)
        e_p2_num = torch.sum(tmp * tmp, dtype=torch.float64).sqrt()
        e_p2_den = e_supp_den  # |G|_F
        e_p2 = float((e_p2_num / e_p2_den).item())

        # 4. Reconstruction Residual (e_rec)
        # |X - QP|_F^2 = tr(G) - 2 tr(P^T H) + tr(P^T S P)
        trG = float(torch.diagonal(gram_norm).sum(dtype=torch.float64).item())
        trPTH = float(torch.sum(P * H, dtype=torch.float64).item())
        torch.mm(S, P, out=tmp)
        trPTSP = float(torch.sum(P * tmp, dtype=torch.float64).item())

        e_rec_sq = max(trG - 2.0 * trPTH + trPTSP, 0.0)
        e_rec = float(math.sqrt(e_rec_sq) / (math.sqrt(max(trG, 0.0)) + eps))

        # 5. Stable Rank (r_stable)
        r_stable = float((trG * trG / (e_supp_den * e_supp_den)).item())

        return {
            "ortho_proj": e_proj,
            "ortho_supp": e_supp,
            "p_skew_rel_fro": e_skew,
            "p2_gram_rel_fro": e_p2,
            "rec_resid": e_rec,
            "stable_rank": r_stable,
        }


class BenchmarkRunner:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.ws_cache: OrderedDict = OrderedDict()

    def set_fast_matmul(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = not self.args.no_tf32
            torch.backends.cudnn.allow_tf32 = not self.args.no_tf32
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(
            "high" if not self.args.no_tf32 else "highest"
        )

    def clear_transient_memory(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        gc.collect()

    def measure(
        self, fn: Callable[[], object], trials: int, warmup: int
    ) -> tuple[object, list[float]]:
        import torch

        out = None
        for _ in range(warmup):
            out = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        times: list[float] = []
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(trials):
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

    def get_workspace(self, n: int, dtype_str: str):
        import torch
        import dwh2

        key = (n, dtype_str, self.device.index, int(self.args.gram_block_rows))
        ws = self.ws_cache.get(key)
        if ws is not None:
            self.ws_cache.move_to_end(key)
            return ws
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
            dtype_str
        ]
        ws = dwh2.DWH2Workspace.allocate(
            n,
            self.device,
            dtype,
            block_rows=int(self.args.gram_block_rows),
        )
        self.ws_cache[key] = ws
        self.ws_cache.move_to_end(key)
        while len(self.ws_cache) > max(0, self.args.ws_cache_max):
            _, old_ws = self.ws_cache.popitem(last=False)
            del old_ws
            self.clear_transient_memory()
        return ws


class RegressionSuite:
    @staticmethod
    def _color_diff(val: float, is_speed: bool = True) -> str:
        if abs(val) < 1e-6:
            return f"{val * 100:>+7.1f}%"
        color = "\033[92m" if val < 0 else "\033[91m"
        reset = "\033[0m"
        return f"{color}{val * 100:>+7.1f}%{reset}"

    @staticmethod
    def check(
        baseline_path, current_path, time_thresh=0.05, metric_thresh=0.01
    ) -> bool:
        from collections import defaultdict

        def load(p):
            res = defaultdict(list)
            if not os.path.exists(p):
                return res
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        r = json.loads(ln)
                        k = (
                            r["method"],
                            r["case"],
                            r["shape"],
                            r["dtype"],
                        )
                        res[k].append(r)
            return res

        base, curr = load(baseline_path), load(current_path)
        all_keys = sorted(set(base.keys()) | set(curr.keys()))

        print(
            f"\n{'Method':<6} {'Case':<16} {'Shape':<12} {'Median (ms)':<20} {'Proj Def':<20} {'Supp Def':<20} {'Rec (F)':<20}"
        )
        print("-" * 120)

        all_passed = True
        for k in all_keys:
            if k not in base or k not in curr:
                continue

            b_med = statistics.median([r.get("median_ms", 0.0) for r in base[k]])
            c_med = statistics.median([r.get("median_ms", 0.0) for r in curr[k]])
            time_diff = (c_med - b_med) / b_med if b_med > 1e-6 else 0.0

            b_proj = statistics.mean([r.get("ortho_proj", 0.0) for r in base[k]])
            c_proj = statistics.mean([r.get("ortho_proj", 0.0) for r in curr[k]])
            proj_diff = (
                (c_proj - b_proj) / b_proj if b_proj > 1e-18 else (c_proj - b_proj)
            )

            b_supp = statistics.mean([r.get("ortho_supp", 0.0) for r in base[k]])
            c_supp = statistics.mean([r.get("ortho_supp", 0.0) for r in curr[k]])
            supp_diff = (
                (c_supp - b_supp) / b_supp if b_supp > 1e-18 else (c_supp - b_supp)
            )

            b_rec = statistics.mean([r.get("rec_resid", 0.0) for r in base[k]])
            c_rec = statistics.mean([r.get("rec_resid", 0.0) for r in curr[k]])
            rec_diff = (c_rec - b_rec) / b_rec if b_rec > 1e-18 else (c_rec - b_rec)

            def fmt_abs_delta(abs_val, delta_val, is_speed=True):
                delta_str = RegressionSuite._color_diff(delta_val, is_speed)
                if is_speed:
                    return f"{abs_val:>8.2f} ({delta_str:>7})"
                else:
                    return f"{abs_val:>8.2e} ({delta_str:>7})"

            s_col = fmt_abs_delta(c_med, time_diff, True)
            o_col = fmt_abs_delta(c_proj, proj_diff, False)
            p_col = fmt_abs_delta(c_supp, supp_diff, False)
            r_col = fmt_abs_delta(c_rec, rec_diff, False)

            if (not math.isnan(proj_diff) and proj_diff > metric_thresh) or (
                not math.isnan(supp_diff) and supp_diff > metric_thresh
            ):
                all_passed = False

            print(
                f"{k[0]:<6} {k[1]:<16} {k[2]:<12} {s_col:<20} {o_col:<20} {p_col:<20} {r_col:<20}"
            )
        return all_passed

    @staticmethod
    def summarize(path, args) -> bool:
        baseline = args.baseline
        if not baseline:
            from collections import defaultdict

            res = defaultdict(list)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        if ln.strip():
                            r = json.loads(ln)
                            k = (r["method"], r["case"], r["shape"], r["dtype"])
                            res[k].append(r)
            except Exception:
                return True

            print(
                f"\n{'Method':<6} {'Case':<20} {'Shape':<12} {'Med (ms)':<10} {'Proj':<12} {'Supp':<12} {'Rec':<12} {'Rank':<8}"
            )
            print("-" * 100)
            for k in sorted(res.keys()):
                m_ms = statistics.median([r["median_ms"] for r in res[k]])
                o_proj = statistics.mean([r.get("ortho_proj", 0.0) for r in res[k]])
                o_supp = statistics.mean([r.get("ortho_supp", 0.0) for r in res[k]])
                p_rec = statistics.mean([r.get("rec_resid", 0.0) for r in res[k]])
                rank = statistics.mean([r.get("stable_rank", 0.0) for r in res[k]])
                print(
                    f"{k[0]:<6} {k[1]:<20} {k[2]:<12} {m_ms:<10.4f} {o_proj:<12.2e} {o_supp:<12.2e} {p_rec:<12.2e} {rank:<8.1f}"
                )
            return True

        import subprocess

        is_rev = False
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", baseline],
                capture_output=True,
                check=True,
            )
            is_rev = True
        except Exception:
            pass

        current_json = "current_tmp.jsonl"
        baseline_json = f"baseline_{baseline if is_rev else 'file'}.jsonl"

        def build_cmd(output_file):
            cmd = [
                sys.executable,
                "bench_profile.py",
                "--output",
                output_file,
                "--quiet",
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--trials",
                str(args.trials),
                "--warmup",
                str(args.warmup),
                "--seeds",
                args.seeds,
                "--cases",
                args.cases,
                "--shapes",
                args.shapes,
                "--gram-block-rows",
                str(args.gram_block_rows),
            ]
            if not args.compile:
                cmd.append("--no-compile")
            if args.compile:
                cmd.append("--compile")
            if args.no_tf32:
                cmd.append("--no-tf32")
            if args.no_gns:
                cmd.append("--no-gns")
            if args.hard:
                cmd.append("--hard")
            if args.one_pass:
                cmd.append("--one-pass")
            if args.no_metrics:
                cmd.append("--no-metrics")
            return cmd

        try:
            print(">>> Benchmarking current code...")
            subprocess.run(build_cmd(current_json), check=True)

            if is_rev:
                print(f">>> Benchmarking baseline revision: {baseline}")
                with open("dwh2.py", "r") as f:
                    saved_code = f.read()
                try:
                    subprocess.run(
                        ["git", "checkout", "-f", baseline, "--", "dwh2.py"], check=True
                    )
                    subprocess.run(build_cmd(baseline_json), check=True)
                finally:
                    with open("dwh2.py", "w") as f:
                        f.write(saved_code)
            else:
                if baseline.endswith(".jsonl"):
                    baseline_json = baseline
                else:
                    print("Error: Baseline must be a git revision or a .jsonl file")
                    return False

            return RegressionSuite.check(baseline_json, current_json)

        finally:
            if os.path.exists(current_json):
                os.remove(current_json)
            if is_rev and os.path.exists(baseline_json) and baseline_json != baseline:
                os.remove(baseline_json)


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
        try:
            return importlib.import_module("newton_schulz.gram_newton_schulz")
        except ModuleNotFoundError:
            return None


def make_gns_runner(gns_mod, kernel: bool, reset: list[int]):
    import torch

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


def main(argv: list[str] | None = None) -> bool:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    ap.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        choices=["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument(
        "--cases",
        default="gaussian,lognormal_cols,ar1_cols,duplicate_cols,lowrank_noise,ill_conditioned,heavy_tail_t,sparse_like,orthogonal_noisy,rank_1_heavy,adversarial_condition",
    )
    ap.add_argument("--shapes", default="16384x4096,8192x4096,4096x4096")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--baseline", help="Baseline revision or JSONL")
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--one-pass", action="store_true")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--no-time",
        action="store_true",
        help="Skip timing trials and median/min ms report",
    )
    ap.add_argument("--ws-cache-max", type=int, default=1)
    ap.add_argument("--gram-block-rows", type=int, default=1024)
    ap.add_argument("--gns-path", default="")
    ap.add_argument("--gns-use-kernels", action="store_true")
    ap.add_argument("--gns-reset-iters", default="2")
    ap.add_argument("--no-gns", action="store_true")
    ap.add_argument("--load-gns", type=str, default="stable/gns_baseline.jsonl")

    args = ap.parse_args(argv)

    if args.baseline and not args.quiet:
        passed = RegressionSuite.summarize(args.output, args)
        sys.exit(0 if passed else 1)

    import torch
    import dwh2

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
        if gns_mod:
            gns_core = make_gns_runner(
                gns_mod,
                args.gns_use_kernels,
                [int(x) for x in args.gns_reset_iters.split(",") if x.strip()],
            )

    dwh2_mod = importlib.reload(dwh2)
    dwh2_speed_core = dwh2_mod.dwh2_core_q
    dwh2_quality_core = dwh2_mod.dwh2_core
    if args.compile:
        logger.info(f"[compile] Jitting tensor-only fast path ({args.compile_mode})...")
        dwh2_speed_core = torch.compile(
            dwh2_speed_core, mode=args.compile_mode, fullgraph=False
        )
        if gns_core is not None:
            gns_core = torch.compile(gns_core, mode=args.compile_mode, fullgraph=False)

    if hasattr(dwh2_mod, "DEFAULT_CONFIG"):
        ell0 = dwh2_mod.DEFAULT_CONFIG.ell0
    else:
        ell0 = getattr(dwh2_mod, "PAPER_MUON_ELL", 1e-3)
    params = dwh2_mod.get_dwh2_params(ell0)

    count_methods = 1 if args.no_gns else 2
    total = len(shapes) * len(cases) * len(seeds) * count_methods
    bar = tqdm(total=total, desc="bench", disable=args.quiet) if tqdm else None

    with open(args.output, "w", encoding="utf-8") as out_f:
        try:
            with torch.inference_mode():
                dtype_map = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                }
                for m, n, s_str in shapes:
                    runner.clear_transient_memory()
                    ws = runner.get_workspace(min(m, n), args.dtype)
                    for case in cases:
                        runner.clear_transient_memory()
                        for seed in seeds:
                            methods = [("dwh2", dwh2_speed_core, dwh2_quality_core)]
                            if not args.no_gns and gns_core is not None:
                                methods.append(("gns", gns_core, None))

                            a_master = CaseGenerator.make_case(
                                case, m, n, device, seed
                            ).to(dtype_map[args.dtype])

                            for name, speed_core, quality_core in methods:
                                a = a_master.clone()
                                a_norm, g_norm = (
                                    dwh2_mod.normalize_moment_with_small_gram(
                                        a, workspace=ws, inplace=True
                                    )
                                )

                                def fn_full(an=a_norm, gn=g_norm):
                                    if name == "dwh2":
                                        return quality_core(
                                            an, gn, params=params, workspace=ws
                                        )
                                    return dwh2_mod.PolarResult(q=speed_core(an))

                                def fn_speed(an=a_norm, gn=g_norm):
                                    if name == "dwh2":
                                        return speed_core(
                                            an, gn, params=params, workspace=ws
                                        )
                                    return speed_core(an)

                                if args.one_pass:
                                    if args.no_time:
                                        out = fn_full()
                                        times = []
                                        med = mn = float("nan")
                                    else:
                                        out, times = runner.measure(
                                            fn_full, args.trials, args.warmup
                                        )
                                        med = statistics.median(times)
                                        mn = min(times)

                                    st = getattr(out, "stats", dwh2_mod.CholStats())
                                    if args.no_metrics:
                                        rec = Record(
                                            method=name,
                                            case=case,
                                            shape=s_str,
                                            dtype=args.dtype,
                                            tf32=not args.no_tf32,
                                            compile=args.compile,
                                            trials=args.trials,
                                            warmup=args.warmup,
                                            median_ms=med,
                                            min_ms=mn,
                                            chol_calls=st.calls,
                                            chol_shifted_calls=st.shifted_calls,
                                            chol_total_retries=st.total_retries,
                                            chol_max_jitter=float(st.max_jitter),
                                        )
                                    else:
                                        stats = MetricsSuite.all_stats(
                                            a_norm, out.q, g_norm, ws
                                        )
                                        rec = Record(
                                            method=name,
                                            case=case,
                                            shape=s_str,
                                            dtype=args.dtype,
                                            tf32=not args.no_tf32,
                                            compile=args.compile,
                                            trials=args.trials,
                                            warmup=args.warmup,
                                            median_ms=med,
                                            min_ms=mn,
                                            ortho_proj=stats["ortho_proj"],
                                            ortho_supp=stats["ortho_supp"],
                                            p_skew_rel_fro=stats["p_skew_rel_fro"],
                                            p2_gram_rel_fro=stats["p2_gram_rel_fro"],
                                            rec_resid=stats["rec_resid"],
                                            stable_rank=stats["stable_rank"],
                                            chol_calls=st.calls,
                                            chol_shifted_calls=st.shifted_calls,
                                            chol_total_retries=st.total_retries,
                                            chol_max_jitter=float(st.max_jitter),
                                        )
                                else:
                                    if args.no_time:
                                        med = mn = float("nan")
                                    else:
                                        _, times = runner.measure(
                                            fn_speed, args.trials, args.warmup
                                        )
                                        med = statistics.median(times)
                                        mn = min(times)

                                    if args.no_metrics:
                                        rec = Record(
                                            method=name,
                                            case=case,
                                            shape=s_str,
                                            dtype=args.dtype,
                                            tf32=not args.no_tf32,
                                            compile=args.compile,
                                            trials=args.trials,
                                            warmup=args.warmup,
                                            median_ms=med,
                                            min_ms=mn,
                                        )
                                    else:
                                        out_qual, _ = runner.measure(fn_full, 1, 0)
                                        st = getattr(
                                            out_qual, "stats", dwh2_mod.CholStats()
                                        )
                                        stats = MetricsSuite.all_stats(
                                            a_norm, out_qual.q, g_norm, ws
                                        )
                                        rec = Record(
                                            method=name,
                                            case=case,
                                            shape=s_str,
                                            dtype=args.dtype,
                                            tf32=not args.no_tf32,
                                            compile=args.compile,
                                            trials=args.trials,
                                            warmup=args.warmup,
                                            median_ms=med,
                                            min_ms=mn,
                                            ortho_proj=stats["ortho_proj"],
                                            ortho_supp=stats["ortho_supp"],
                                            p_skew_rel_fro=stats["p_skew_rel_fro"],
                                            p2_gram_rel_fro=stats["p2_gram_rel_fro"],
                                            rec_resid=stats["rec_resid"],
                                            stable_rank=stats["stable_rank"],
                                            chol_calls=st.calls,
                                            chol_shifted_calls=st.shifted_calls,
                                            chol_total_retries=st.total_retries,
                                            chol_max_jitter=float(st.max_jitter),
                                        )
                                        del out_qual

                                out_f.write(
                                    json.dumps(asdict(rec), sort_keys=True) + "\n"
                                )
                                out_f.flush()
                                if bar:
                                    bar.update(1)

                                del a, a_norm, g_norm

                            del a_master
                        runner.clear_transient_memory()
        finally:
            if bar:
                bar.close()

    if args.no_gns and args.load_gns and os.path.exists(args.load_gns):
        logger.info(f"[merge] GNS results available at {args.load_gns}")


def main_hard() -> None:
    import subprocess

    def run_phase(name, extra_args):
        print(f">>> {name}")
        cmd = [
            sys.executable,
            "bench_profile.py",
            "--no-gns",
            "--hard",
            "--seeds",
            "0",
        ] + extra_args

        if "--baseline" in sys.argv:
            idx = sys.argv.index("--baseline")
            if idx + 1 < len(sys.argv):
                cmd.extend(["--baseline", sys.argv[idx + 1]])

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"!!! {name} failed or detected regression.")
            sys.exit(res.returncode)

    run_phase(
        "Phase 1: Fast no-compile regression check",
        [
            "--trials",
            "5",
            "--warmup",
            "1",
            "--no-compile",
            "--output",
            "hard_no_compile.jsonl",
        ],
    )
    run_phase(
        "Phase 2: Full compiled performance comparison",
        [
            "--trials",
            "10",
            "--warmup",
            "2",
            "--compile",
            "--output",
            "hard_compile.jsonl",
        ],
    )


def main_baseline() -> None:
    import subprocess

    cmd = [
        sys.executable,
        "bench_profile.py",
        "--no-gns",
        "--hard",
        "--trials",
        "10",
        "--warmup",
        "2",
        "--seeds",
        "0",
        "--output",
        "baseline.jsonl",
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)


def main_promote() -> None:
    import shutil

    if os.path.exists("results.jsonl"):
        shutil.copy("results.jsonl", "baseline.jsonl")
        print("Promoted results.jsonl to baseline.jsonl")
    else:
        print("Error: results.jsonl not found")


if __name__ == "__main__":
    main()
