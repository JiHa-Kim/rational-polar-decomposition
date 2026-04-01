from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from .bench import make_case, measure, normalize_fro, set_fast_matmul
from .dwh2 import dwh2
from .pe5 import PAPER_MUON_ELL, pe5, pe5_coefficients


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
        default="max-autotune",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--trace-dir", type=str, default="profiles")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--robust-dwh2", action="store_true")
    parser.add_argument("--symmetrize-pe5", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_fast_matmul(args.tf32)

    case = make_case(args.case, args.rows, args.cols, device=device, seed=args.seed)
    a = normalize_fro(case.a).contiguous()

    coeffs = pe5_coefficients(ell0=PAPER_MUON_ELL, steps=5)

    if args.method == "dwh2":

        def base(x):
            return dwh2(x, ell0=PAPER_MUON_ELL, tf32=args.tf32, robust=args.robust_dwh2)
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

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(3):
                fn(a)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        prof.export_chrome_trace(str(trace_path))
        print(f"chrome_trace={trace_path}")

        sort_key = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        print(prof.key_averages().table(sort_by=sort_key, row_limit=30))


if __name__ == "__main__":
    main()
