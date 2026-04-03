from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from .bench import make_case, measure, set_fast_matmul
from ..algorithms.dwh2 import DWH2_MODES, dwh2
from ..utils.normalization import normalize_matrix
from ..algorithms.pe5 import PAPER_MUON_ELL, PAPER_NORM_EPS, pe5, pe5_coefficients


def compile_fn(
    fn: Callable[[torch.Tensor], object], mode: str
) -> Callable[[torch.Tensor], object]:
    if mode == "eager":
        return fn
    if mode in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        return torch.compile(fn, mode=None if mode == "default" else mode)  # type: ignore[arg-type]
    raise ValueError(f"unknown mode {mode}")


@dataclass
class OpStats:
    name: str
    total_time_us: float = 0.0
    count: int = 0

    @property
    def total_time_ms(self) -> float:
        return self.total_time_us / 1000.0

    @property
    def mean_time_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time_ms / self.count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile optimized polar kernels on GPU."
    )
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--method", choices=["dwh2", "pe5"], default="pe5")
    parser.add_argument(
        "--case",
        type=str,
        default="gaussian",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="eager",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--trace-dir", type=str, default="profiles")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--robust-dwh2", action="store_true")
    parser.add_argument("--ell0", type=float, default=PAPER_MUON_ELL)
    parser.add_argument(
        "--dwh2-mode",
        choices=list(DWH2_MODES),
        default="rectangular",
    )
    parser.add_argument("--symmetrize-pe5", action="store_true")
    parser.add_argument(
        "--normalizer",
        choices=["fro", "spectral_bound", "spectral_additive"],
        default="spectral_additive",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    set_fast_matmul(args.tf32)

    case = make_case(args.case, args.rows, args.cols, device=device, seed=args.seed)
    a, _ = normalize_matrix(
        case.a,
        method=args.normalizer,
        eps=PAPER_NORM_EPS,
    )
    a = a.contiguous()

    coeffs = pe5_coefficients(ell0=PAPER_MUON_ELL, steps=5)

    if args.method == "dwh2":

        def base(x):
            return dwh2(
                x,
                ell0=args.ell0,
                mode=args.dwh2_mode,
                robust=args.robust_dwh2,
            )
    else:

        def base(x):
            return pe5(
                x,
                ell0=PAPER_MUON_ELL,
                coeffs=coeffs,
                symmetrize_inputs=args.symmetrize_pe5,
            )

    fn = compile_fn(base, args.mode)

    with torch.inference_mode():
        out, times = measure(lambda: fn(a), args.iters, args.warmup)
        median_ms = float(statistics.median(times))
        min_ms = float(min(times))
        print(
            json.dumps(
                {
                    "method": args.method,
                    "case": args.case,
                    "mode": args.mode,
                    "rows": args.rows,
                    "cols": args.cols,
                    "median_ms": median_ms,
                    "min_ms": min_ms,
                    "output_shape": tuple(out.q.shape),
                },
                sort_keys=True,
            )
        )

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_name = f"{args.method}_{args.case}_{args.mode}_{args.rows}x{args.cols}"
        trace_path = trace_dir / f"{trace_name}.json"

        # Use 5 iterations for profiling to get a stable average
        profile_iters = 5
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(profile_iters):
                fn(a)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        prof.export_chrome_trace(str(trace_path))
        print(f"chrome_trace={trace_path}")

        # Standard profiler table
        sort_key = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        print("\n=== Standard Profiler Table ===")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

        # Detailed breakdown
        stats: dict[str, OpStats] = {}

        def get_stat(name):
            if name not in stats:
                stats[name] = OpStats(name)
            return stats[name]

        for event in prof.key_averages(group_by_input_shape=True):
            name = event.key
            # Use 'device_time_total' for high-level aten ops to capture kernel work
            # and 'self_device_time_total' for leaf-level kernels/nodes.
            is_aten = name.startswith("aten::")
            time_us = (
                getattr(event, "device_time_total", 0)
                if is_aten
                else getattr(event, "self_device_time_total", 0)
            )
            if time_us == 0 and device.type == "cpu":
                time_us = event.cpu_time_total if is_aten else event.self_cpu_time_total

            count = event.count

            # Avoid double-counting high-level dispatcher ops and their aten::mm children
            if name in {"aten::matmul"}:
                continue

            # Filtering to relevant ops
            if not (
                is_aten
                or "affine_diag" in name.lower()
                or "scale_symmetrize" in name.lower()
                or "cholesky" in name.lower()
            ):
                continue

            if name.startswith("aten::mm"):
                shapes = event.input_shapes
                if shapes and len(shapes) >= 2:
                    s1, s2 = shapes[0], shapes[1]
                    m, k, n = s1[0], s1[1], s2[1]
                    bucket = f"GEMM {m}x{k}x{n}"
                else:
                    bucket = "GEMM (other)"
                s = get_stat(bucket)
                s.total_time_us += time_us
                s.count += count
            elif "cholesky" in name.lower():
                s = get_stat("Cholesky (small-side)")
                s.total_time_us += time_us
                s.count += count
            elif "solve_triangular" in name.lower():
                s = get_stat("Triangular Solve (small-side)")
                s.total_time_us += time_us
                s.count += count
            elif "affine_diag" in name.lower():
                s = get_stat("Triton: Affine Diagonal")
                s.total_time_us += time_us
                s.count += count
            elif "scale_symmetrize" in name.lower():
                s = get_stat("Triton: Scale/Symmetrize")
                s.total_time_us += time_us
                s.count += count
            elif any(
                x in name.lower()
                for x in ["copy", "fill", "zero", "neg", "add_", "mul_", "div_"]
            ):
                s = get_stat("Memory / Element-wise")
                s.total_time_us += time_us
                s.count += count
            elif "synchronize" in name.lower():
                continue
            else:
                if time_us > 0:
                    s = get_stat("Other overhead")
                    s.total_time_us += time_us
                    s.count += count

        total_ms = sum(s.total_time_ms for s in stats.values())
        print(f"\n=== Detailed Per-Op Breakdown ({args.method.upper()}) ===")
        print(
            f"{'Operation':<35} | {'Aggregate (ms)':>15} | {'Count':>7} | {'Per-op (ms)':>12} | {'Share (%)':>10}"
        )
        print("-" * 90)
        # Sort by total time, but keep GEMMs together ideally (sorting handles this well enough naturally)
        for s in sorted(stats.values(), key=lambda x: x.total_time_us, reverse=True):
            if s.total_time_us == 0:
                continue
            share = (s.total_time_ms / total_ms) * 100
            # Normalize aggregate time to one 'fn(a)' call
            agg_per_run = s.total_time_ms / profile_iters
            # Normalize count to one 'fn(a)' call
            count_per_run = s.count / profile_iters
            count_str = f"{int(count_per_run)}" if count_per_run.is_integer() else f"{count_per_run:.1f}"
            print(
                f"{s.name:<35} | {agg_per_run:>15.4f} | {count_str:>7} | {s.mean_time_ms:>12.4f} | {share:>10.2f}%"
            )


if __name__ == "__main__":
    main()
