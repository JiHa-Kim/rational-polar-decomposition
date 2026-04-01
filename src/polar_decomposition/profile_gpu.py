from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.profiler import ProfilerActivity, profile

from .dwh2 import dwh2
from .pe5 import PAPER_MUON_ELL, PAPER_NORM_EPS, pe5, pe5_coefficients


def set_fast_matmul(tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def _randn(shape: Sequence[int], *, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=torch.float32, generator=g)


def make_case(
    name: str, rows: int, cols: int, *, device: torch.device, seed: int
) -> torch.Tensor:
    m, n = rows, cols
    if name == "gaussian":
        a = _randn((m, n), device=device, seed=seed)
    elif name == "lognormal_cols":
        x = _randn((m, n), device=device, seed=seed)
        scales = torch.exp(1.5 * _randn((n,), device=device, seed=seed + 1))
        scales = scales / scales.median().clamp_min(1e-8)
        a = x * scales[None, :]
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
    else:
        raise ValueError(f"unknown case {name}")
    return a / (torch.linalg.matrix_norm(a, ord="fro") + PAPER_NORM_EPS)


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


def measure_ms(
    fn: Callable[[], object], warmup: int, iters: int
) -> tuple[object, float, float]:
    out = None
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        times_t = torch.tensor(times)
        return out, float(times_t.median().item()), float(times_t.min().item())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times_t = torch.tensor(times)
    return out, float(times_t.median().item()), float(times_t.min().item())


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
        choices=["gaussian", "lognormal_cols", "duplicate_cols", "lowrank_noise"],
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
    a = make_case(args.case, args.rows, args.cols, device=device, seed=args.seed)

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
        out, median_ms, min_ms = measure_ms(lambda: fn(a), args.warmup, args.iters)
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

        prof_ctx = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        with prof_ctx as prof:
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
