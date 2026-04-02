"""Command-line entrypoints for benchmarks, sweeps, and profiling."""

from .bench import main as bench_main
from .norm_sweep import main as norm_sweep_main
from .profile_gpu import main as profile_gpu_main

__all__ = ["bench_main", "norm_sweep_main", "profile_gpu_main"]
