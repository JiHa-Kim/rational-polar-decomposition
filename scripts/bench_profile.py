from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import gc
import importlib
import json
import logging
import math
import subprocess
import statistics
import time
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from bench_common import (
    CaseGenerator,
    DEFAULT_CASES,
    DEFAULT_SHAPES,
    DTYPE_MAP,
    MetricsSuite,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("bench_profile")


def setup_logging(quiet: bool) -> None:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING if quiet else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)


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
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times: list[float] = []
            for _ in range(trials):
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
            return out, times

        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return out, times

    def get_workspace(self, n: int, dtype_str: str):
        import dwh2

        key = (n, dtype_str, self.device.index, int(self.args.gram_block_rows))
        ws = self.ws_cache.get(key)
        if ws is not None:
            self.ws_cache.move_to_end(key)
            return ws

        ws = dwh2.DWH2Workspace.allocate(
            n,
            self.device,
            DTYPE_MAP[dtype_str],
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
    def _group_records(path: str) -> dict[tuple[str, str, str, str], list[dict]]:
        grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
        if not os.path.exists(path):
            return grouped
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    record["method"],
                    record["case"],
                    record["shape"],
                    record["dtype"],
                )
                grouped[key].append(record)
        return grouped

    @staticmethod
    def _color_diff(val: float) -> str:
        if abs(val) < 1e-6:
            return f"{val * 100:>+7.1f}%"
        color = "\033[92m" if val < 0 else "\033[91m"
        return f"{color}{val * 100:>+7.1f}%\033[0m"

    @staticmethod
    def check(
        baseline_path, current_path, time_thresh=0.05, metric_thresh=0.01
    ) -> bool:
        del time_thresh
        base = RegressionSuite._group_records(baseline_path)
        curr = RegressionSuite._group_records(current_path)
        all_keys = sorted(set(base) | set(curr))

        print(
            f"\n{'Method':<6} {'Case':<16} {'Shape':<12} {'Median (ms)':<20} {'Proj Def':<20} {'Supp Def':<20} {'Rec (F)':<20}"
        )
        print("-" * 120)

        def mean_metric(records: list[dict], key: str) -> float:
            return statistics.mean(r.get(key, 0.0) for r in records)

        def median_metric(records: list[dict], key: str) -> float:
            return statistics.median(r.get(key, 0.0) for r in records)

        def fmt_abs_delta(abs_val: float, delta_val: float, *, is_speed: bool) -> str:
            delta_str = RegressionSuite._color_diff(delta_val)
            if is_speed:
                return f"{abs_val:>8.2f} ({delta_str:>7})"
            return f"{abs_val:>8.2e} ({delta_str:>7})"

        all_passed = True
        for key in all_keys:
            if key not in base or key not in curr:
                continue

            b_med = median_metric(base[key], "median_ms")
            c_med = median_metric(curr[key], "median_ms")
            time_diff = (c_med - b_med) / b_med if b_med > 1e-6 else 0.0

            b_proj = mean_metric(base[key], "ortho_proj")
            c_proj = mean_metric(curr[key], "ortho_proj")
            proj_diff = (
                (c_proj - b_proj) / b_proj if b_proj > 1e-18 else (c_proj - b_proj)
            )

            b_supp = mean_metric(base[key], "ortho_supp")
            c_supp = mean_metric(curr[key], "ortho_supp")
            supp_diff = (
                (c_supp - b_supp) / b_supp if b_supp > 1e-18 else (c_supp - b_supp)
            )

            b_rec = mean_metric(base[key], "rec_resid")
            c_rec = mean_metric(curr[key], "rec_resid")
            rec_diff = (c_rec - b_rec) / b_rec if b_rec > 1e-18 else (c_rec - b_rec)

            if (not math.isnan(proj_diff) and proj_diff > metric_thresh) or (
                not math.isnan(supp_diff) and supp_diff > metric_thresh
            ):
                all_passed = False

            print(
                f"{key[0]:<6} {key[1]:<16} {key[2]:<12} "
                f"{fmt_abs_delta(c_med, time_diff, is_speed=True):<20} "
                f"{fmt_abs_delta(c_proj, proj_diff, is_speed=False):<20} "
                f"{fmt_abs_delta(c_supp, supp_diff, is_speed=False):<20} "
                f"{fmt_abs_delta(c_rec, rec_diff, is_speed=False):<20}"
            )
        return all_passed

    @staticmethod
    def summarize(path, args) -> bool:
        if not args.baseline:
            grouped = RegressionSuite._group_records(path)
            print(
                f"\n{'Method':<6} {'Case':<20} {'Shape':<12} {'Med (ms)':<10} {'Proj':<12} {'Supp':<12} {'Rec':<12} {'Rank':<8}"
            )
            print("-" * 100)
            for key in sorted(grouped):
                records = grouped[key]
                print(
                    f"{key[0]:<6} {key[1]:<20} {key[2]:<12} "
                    f"{statistics.median(r['median_ms'] for r in records):<10.4f} "
                    f"{statistics.mean(r.get('ortho_proj', 0.0) for r in records):<12.2e} "
                    f"{statistics.mean(r.get('ortho_supp', 0.0) for r in records):<12.2e} "
                    f"{statistics.mean(r.get('rec_resid', 0.0) for r in records):<12.2e} "
                    f"{statistics.mean(r.get('stable_rank', 0.0) for r in records):<8.1f}"
                )
            return True

        baseline = args.baseline
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

        def build_cmd(output_file: str) -> list[str]:
            cmd = [
                sys.executable,
                __file__,
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
                with open("dwh2.py", "r", encoding="utf-8") as handle:
                    saved_code = handle.read()
                try:
                    subprocess.run(
                        ["git", "checkout", "-f", baseline, "--", "dwh2.py"],
                        check=True,
                    )
                    subprocess.run(build_cmd(baseline_json), check=True)
                finally:
                    with open("dwh2.py", "w", encoding="utf-8") as handle:
                        handle.write(saved_code)
            elif baseline.endswith(".jsonl"):
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
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

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
        transposed = x.size(-2) > x.size(-1)
        if transposed:
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
        return y.mT.contiguous() if transposed else y

    return core


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in parse_csv(raw)]


def parse_shapes(raw: str) -> list[tuple[int, int, str]]:
    shapes: list[tuple[int, int, str]] = []
    for spec in parse_csv(raw):
        m_str, n_str = spec.lower().split("x")
        shapes.append((int(m_str), int(n_str), spec))
    return shapes


def method_specs(args, dwh2_mod, gns_core):
    methods = []
    if not args.no_dwh2:
        methods.append(("dwh2", dwh2_mod.dwh2_core_q, dwh2_mod.dwh2_core))
    if not args.no_gns and gns_core is not None:
        methods.append(("gns", gns_core, None))
    return methods


def record_base(
    args, name: str, case: str, shape: str, median_ms: float, min_ms: float
) -> dict:
    return {
        "method": name,
        "case": case,
        "shape": shape,
        "dtype": args.dtype,
        "tf32": not args.no_tf32,
        "compile": args.compile,
        "trials": args.trials,
        "warmup": args.warmup,
        "median_ms": median_ms,
        "min_ms": min_ms,
    }


def make_record(
    args,
    name: str,
    case: str,
    shape: str,
    median_ms: float,
    min_ms: float,
    *,
    stats=None,
    chol_stats=None,
) -> Record:
    payload = record_base(args, name, case, shape, median_ms, min_ms)
    if stats is not None:
        payload.update(stats)
    if chol_stats is not None:
        payload.update(
            chol_calls=chol_stats.calls,
            chol_shifted_calls=chol_stats.shifted_calls,
            chol_total_retries=chol_stats.total_retries,
            chol_max_jitter=float(chol_stats.max_jitter),
        )
    return Record(**payload)


def median_and_min(times: list[float], no_time: bool) -> tuple[float, float]:
    if no_time:
        return float("nan"), float("nan")
    return statistics.median(times), min(times)


def benchmark_method(
    *,
    args,
    runner: BenchmarkRunner,
    dwh2_mod,
    params,
    ws,
    name: str,
    speed_core,
    quality_core,
    case: str,
    shape: str,
    a_master,
):
    a = a_master.clone()
    a_norm, g_norm = dwh2_mod.normalize_moment_with_small_gram(
        a, workspace=ws, inplace=True
    )

    def fn_full(an=a_norm, gn=g_norm):
        if name == "dwh2":
            return quality_core(an, gn, params=params, workspace=ws)
        return dwh2_mod.PolarResult(q=speed_core(an))

    def fn_speed(an=a_norm, gn=g_norm):
        if name == "dwh2":
            return speed_core(an, gn, params=params, workspace=ws)
        return speed_core(an)

    if args.one_pass:
        if args.no_time:
            out = fn_full()
            med = mn = float("nan")
        else:
            out, times = runner.measure(fn_full, args.trials, args.warmup)
            med, mn = median_and_min(times, False)

        chol_stats = getattr(out, "stats", dwh2_mod.CholStats())
        metrics = (
            None
            if args.no_metrics
            else MetricsSuite.all_stats(a_norm, out.q, g_norm, ws)
        )
        record = make_record(
            args,
            name,
            case,
            shape,
            med,
            mn,
            stats=metrics,
            chol_stats=chol_stats,
        )
    else:
        if args.no_time:
            med = mn = float("nan")
        else:
            _, times = runner.measure(fn_speed, args.trials, args.warmup)
            med, mn = median_and_min(times, False)

        if args.no_metrics:
            record = make_record(args, name, case, shape, med, mn)
        else:
            out, _ = runner.measure(fn_full, 1, 0)
            record = make_record(
                args,
                name,
                case,
                shape,
                med,
                mn,
                stats=MetricsSuite.all_stats(a_norm, out.q, g_norm, ws),
                chol_stats=getattr(out, "stats", dwh2_mod.CholStats()),
            )

    del a, a_norm, g_norm
    return record


def write_records(args) -> None:
    import torch
    import dwh2

    if args.hard:
        args.cases = "ill_conditioned,lowrank_noise"
        args.shapes = "16384x4096"

    setup_logging(args.quiet)
    torch._dynamo.config.capture_scalar_outputs = True
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*torch\._prims_common\.check.*",
    )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    runner = BenchmarkRunner(device, args)
    runner.set_fast_matmul()

    seeds = parse_int_csv(args.seeds)
    cases = parse_csv(args.cases)
    shapes = parse_shapes(args.shapes)

    gns_core = None
    if not args.no_gns:
        gns_mod = import_gns(args.gns_path)
        if gns_mod is not None:
            gns_core = make_gns_runner(
                gns_mod, args.gns_use_kernels, parse_int_csv(args.gns_reset_iters)
            )

    dwh2_mod = importlib.reload(dwh2)
    dwh2_speed_core = dwh2_mod.dwh2_core_q
    if args.compile:
        logger.info(f"[compile] Jitting tensor-only fast path ({args.compile_mode})...")
        dwh2_speed_core = torch.compile(
            dwh2_speed_core,
            mode=args.compile_mode,
            fullgraph=False,
        )
        if gns_core is not None:
            gns_core = torch.compile(gns_core, mode=args.compile_mode, fullgraph=False)

    ell0 = getattr(
        getattr(dwh2_mod, "DEFAULT_CONFIG", None),
        "ell0",
        getattr(dwh2_mod, "PAPER_MUON_ELL", 1e-3),
    )
    params = dwh2_mod.get_dwh2_params(ell0)

    methods = method_specs(args, dwh2_mod, gns_core)
    total = len(shapes) * len(cases) * len(seeds) * len(methods)
    bar = tqdm(total=total, desc="bench", disable=args.quiet) if tqdm else None

    with open(args.output, "w", encoding="utf-8") as out_f:
        try:
            with torch.inference_mode():
                for m, n, shape_str in shapes:
                    runner.clear_transient_memory()
                    ws = runner.get_workspace(min(m, n), args.dtype)
                    for case in cases:
                        runner.clear_transient_memory()
                        for seed in seeds:
                            a_master = CaseGenerator.make_case(
                                case, m, n, device, seed
                            ).to(DTYPE_MAP[args.dtype])
                            for name, speed_core, quality_core in methods:
                                record = benchmark_method(
                                    args=args,
                                    runner=runner,
                                    dwh2_mod=dwh2_mod,
                                    params=params,
                                    ws=ws,
                                    name=name,
                                    speed_core=speed_core,
                                    quality_core=quality_core,
                                    case=case,
                                    shape=shape_str,
                                    a_master=a_master,
                                )
                                out_f.write(
                                    json.dumps(asdict(record), sort_keys=True) + "\n"
                                )
                                out_f.flush()
                                if bar is not None:
                                    bar.update(1)
                            del a_master
                        runner.clear_transient_memory()
        finally:
            if bar is not None:
                bar.close()

    if args.no_gns and args.load_gns and os.path.exists(args.load_gns):
        logger.info(f"[merge] GNS results available at {args.load_gns}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=True)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        choices=["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--dtype", default="fp16", choices=tuple(DTYPE_MAP))
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--baseline", help="Baseline revision or JSONL")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--one-pass", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Skip timing trials and median/min ms report",
    )
    parser.add_argument("--ws-cache-max", type=int, default=1)
    parser.add_argument("--gram-block-rows", type=int, default=1024)
    parser.add_argument("--gns-path", default="")
    parser.add_argument("--gns-use-kernels", action="store_true")
    parser.add_argument("--gns-reset-iters", default="2")
    parser.add_argument("--no-gns", action="store_true", help="Skip GNS benchmarks")
    parser.add_argument("--no-dwh2", action="store_true", help="Skip DWH2 benchmarks")
    parser.add_argument("--load-gns", type=str, default="stable/gns_baseline.jsonl")
    return parser


def main(argv: list[str] | None = None) -> bool:
    args = build_arg_parser().parse_args(argv)
    if args.baseline and not args.quiet:
        passed = RegressionSuite.summarize(args.output, args)
        sys.exit(0 if passed else 1)

    write_records(args)
    return True


def _run_subprocess(extra_args: Iterable[str]) -> None:
    cmd = [sys.executable, __file__, *extra_args]
    subprocess.run(cmd, check=True)


def main_hard() -> None:
    def run_phase(name: str, extra_args: list[str]) -> None:
        print(f">>> {name}")
        cmd = [
            sys.executable,
            __file__,
            "--no-gns",
            "--hard",
            "--seeds",
            "0",
            *extra_args,
        ]
        if "--baseline" in sys.argv:
            idx = sys.argv.index("--baseline")
            if idx + 1 < len(sys.argv):
                cmd.extend(["--baseline", sys.argv[idx + 1]])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!!! {name} failed or detected regression.")
            sys.exit(result.returncode)

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
    _run_subprocess(
        [
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
            *sys.argv[1:],
        ]
    )


def main_promote() -> None:
    import shutil

    if os.path.exists("results.jsonl"):
        shutil.copy("results.jsonl", "baseline.jsonl")
        print("Promoted results.jsonl to baseline.jsonl")
    else:
        print("Error: results.jsonl not found")


if __name__ == "__main__":
    main()
