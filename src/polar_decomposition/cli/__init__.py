"""Command-line entrypoints for benchmarks and profiling."""

from .bench import main as bench_main
from .profile_gpu import main as profile_gpu_main

__all__ = ["bench_main", "profile_gpu_main"]
