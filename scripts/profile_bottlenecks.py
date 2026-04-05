import torch
from dwh2 import (
    dwh2_end_to_end,
    normalize_small_gram,
    _ensure_workspace,
    _dwh2_core_impl,
)


def profile_dwh2(m=16384, n=4096, dtype=torch.float16, device="cuda"):
    a = torch.randn(m, n, device=device, dtype=dtype)
    workspace = _ensure_workspace(None, n, device, dtype)

    # Warmup (also triggers compilation)
    print("Warmup and Compilation...")
    for _ in range(3):
        dwh2_end_to_end(a, workspace=workspace)

    torch.cuda.synchronize()

    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)

    # Section Events
    e_norm_start = torch.cuda.Event(enable_timing=True)
    e_norm_end = torch.cuda.Event(enable_timing=True)

    e_core_start = torch.cuda.Event(enable_timing=True)
    e_core_end = torch.cuda.Event(enable_timing=True)

    # Begin Profile
    start_total.record()

    # 1. Normalization
    e_norm_start.record()
    scale, gram_norm = normalize_small_gram(a, workspace=workspace)
    e_norm_end.record()

    # 2. Core (Step 1 + Step 2 + Final Apply)
    e_core_start.record()
    _dwh2_core_impl(a, gram_norm, workspace=workspace, norm_scale=scale)
    e_core_end.record()

    end_total.record()
    torch.cuda.synchronize()

    # Results
    t_total = start_total.elapsed_time(end_total)
    t_norm = e_norm_start.elapsed_time(e_norm_end)
    t_core = e_core_start.elapsed_time(e_core_end)

    print(f"Profiling DWH2 on {m}x{n} ({dtype})")
    print(f"{'Section':<20} | {'Time (ms)':<10} | {'% Total':<10}")
    print("-" * 45)
    print(f"{'Normalization':<20} | {t_norm:>10.2f} | {100 * t_norm / t_total:>9.1f}%")
    print(f"{'Core Pipeline':<20} | {t_core:>10.2f} | {100 * t_core / t_total:>9.1f}%")
    print("-" * 45)
    print(f"{'Total':<20} | {t_total:>10.2f} | 100.0%")


if __name__ == "__main__":
    profile_dwh2()
