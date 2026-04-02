import argparse
from dataclasses import dataclass
import torch
from torch.profiler import profile, ProfilerActivity
from .bench import make_case, set_fast_matmul
from .dwh2 import dwh2
from .normalization import normalize_matrix
from .pe5 import PAPER_MUON_ELL, PAPER_NORM_EPS, pe5, pe5_coefficients

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--method", choices=["dwh2", "pe5"], default="dwh2")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--tf32", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    set_fast_matmul(args.tf32)

    case = make_case("gaussian", args.rows, args.cols, device=device, seed=0)
    a, _ = normalize_matrix(case.a, method="fro", eps=PAPER_NORM_EPS)
    a = a.contiguous()
    coeffs = pe5_coefficients(ell0=PAPER_MUON_ELL, steps=5)

    if args.method == "dwh2":
        def fn(x): return dwh2(x, ell0=PAPER_MUON_ELL)
    else:
        def fn(x): return pe5(x, ell0=PAPER_MUON_ELL, coeffs=coeffs)

    for _ in range(2): fn(a)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(args.iters):
            fn(a)
            torch.cuda.synchronize()

    stats: dict[str, OpStats] = {}
    def get_stat(name):
        if name not in stats: stats[name] = OpStats(name)
        return stats[name]

    for event in prof.key_averages(group_by_input_shape=True):
        name = event.key
        # For aten:: ops, the self_device_time_total is the sum of kernels it launched.
        # Mangled kernel names are also present in key_averages and would cause double counting.
        # Thus we only count aten:: ops and our custom Triton kernels.
        if not (name.startswith("aten::") or "affine_diag" in name.lower() or "scale_symmetrize" in name.lower() or "cholesky" in name.lower()):
            continue
            
        # Check if we have device time (CUDA/MPS etc.)
        time_us = getattr(event, "self_device_time_total", 0)
        if time_us == 0:
            # Some ops might only report cpu time if they are purely CPU or if profiler missed it
            # But we are mostly interested in the GPU hot path.
            # If it's a top-levelaten op that we care about (like mm) and it has 0 device time, 
            # it might be on the wrong device or missing.
            pass
            
        count = event.count
        
        if name.startswith("aten::mm") or name.startswith("aten::matmul"):
            shapes = event.input_shapes
            is_rect = False
            if shapes:
                for s in shapes:
                    if args.rows in s: 
                        is_rect = True
                        break
            bucket = "Rectangular GEMM" if is_rect else "Small-side GEMM"
            s = get_stat(bucket)
            s.total_time_us += time_us
            s.count += count
        elif "cholesky" in name.lower():
            s = get_stat("Cholesky")
            s.total_time_us += time_us
            s.count += count
        elif "solve_triangular" in name.lower():
            s = get_stat("Triangular Solve")
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
        elif any(x in name.lower() for x in ["copy", "fill", "zero", "neg", "add_", "mul_", "div_"]):
            s = get_stat("Memory / Element-wise")
            s.total_time_us += time_us
            s.count += count
        elif "synchronize" in name.lower():
            continue # Ignore sync
        else:
            if time_us > 0:
                s = get_stat("Other GPU kernels")
                s.total_time_us += time_us
                s.count += count
                # Debug print for large unknown kernels
                if time_us > 10000: # >10ms per iter (if iters=1)
                     # print(f"DEBUG: Unknown large kernel: {name}, time_us={time_us}")
                     pass

    total_ms = sum(s.total_time_ms for s in stats.values())
    print(f"\nDetailed Profile for {args.method.upper()} ({args.rows}x{args.cols}, {args.iters} iterations)")
    print(f"{'Operation':<30} | {'Total (ms)':>10} | {'Count':>7} | {'Per-op (ms)':>12} | {'Share (%)':>10}")
    print("-" * 80)
    for s in sorted(stats.values(), key=lambda x: x.total_time_us, reverse=True):
        if s.total_time_us == 0: continue
        share = (s.total_time_ms / total_ms) * 100
        # Divide count by iters to get count per run
        count_per_run = s.count / args.iters
        print(f"{s.name:<30} | {s.total_time_ms / args.iters:>10.2f} | {count_per_run:>7.1f} | {s.mean_time_ms:>12.4f} | {share:>10.2f}%")

if __name__ == "__main__":
    main()
