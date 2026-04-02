from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from .bench import (
    Case,
    line_writer,
    make_case,
    measure,
    polar_reference,
    set_fast_matmul,
    summarize,
)
from .dwh2 import dwh2
from .normalization import normalize_matrix
from .pe5 import PAPER_MUON_ELL, PAPER_NORM_EPS, pe5, pe5_coefficients
from .precond import PolarResult


def _reference_norms(a: torch.Tensor) -> dict[str, float]:
    x = a if a.shape[0] >= a.shape[1] else a.mT.contiguous()
    x64 = x.to(torch.float64)
    gram = x64.mT @ x64
    gram = 0.5 * (gram + gram.mT)
    fro = float(torch.sqrt(torch.trace(gram)).item())
    schatten4 = float(torch.sqrt(torch.linalg.matrix_norm(gram, ord="fro")).item())
    evals = torch.linalg.eigvalsh(gram)
    spectral = math.sqrt(max(float(evals[-1].item()), 0.0))
    return {
        "true_fro": fro,
        "true_schatten4": schatten4,
        "true_spectral": spectral,
    }


def _normalizer_label(method: str, ridge_scale: float, ridge_stat: str) -> str:
    if method == "fro":
        return "fro"
    return f"schatten4_{ridge_stat}_r{ridge_scale:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep TF32-safe normalization recipes on the benchmark cases."
    )
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--ell0", type=float, default=PAPER_MUON_ELL)
    parser.add_argument(
        "--reference",
        type=str,
        default="fp32",
        choices=["none", "fp32", "fp64"],
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Enable low-memory projected-objective audit in float64.",
    )
    parser.add_argument(
        "--audit-device",
        type=str,
        default="same",
        choices=["same", "cpu"],
    )
    parser.add_argument("--audit-chunk-rows", type=int, default=2048)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "gaussian",
            "lognormal_cols",
            "ar1_cols",
            "duplicate_cols",
            "lowrank_noise",
            "ill_conditioned",
            "heavy_tail_t",
            "sparse_like",
            "orthogonal_noisy",
            "rank_1_heavy",
            "adversarial_condition",
        ],
    )
    parser.add_argument("--methods", nargs="+", default=["dwh2", "pe5"])
    parser.add_argument(
        "--normalizers",
        nargs="+",
        default=["fro", "schatten4"],
        choices=["fro", "schatten4"],
    )
    parser.add_argument(
        "--schatten4-ridge-scales",
        nargs="+",
        type=float,
        default=[0.0, 4.0, 8.0, 16.0, 32.0],
    )
    parser.add_argument(
        "--schatten4-ridge-stat",
        type=str,
        default="max",
        choices=["mean", "max"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("runs") / time.strftime("%Y%m%d_%H%M%S") / "norm_sweep.jsonl"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_fast_matmul(args.tf32)
    out_f = line_writer(args.output)

    try:
        with torch.inference_mode():
            coeffs = pe5_coefficients(ell0=args.ell0, steps=5)
            stress_cases = {"duplicate_cols", "lowrank_noise"}
            stress_ell0 = 1e-6

            variants: list[tuple[str, float]] = []
            for method in args.normalizers:
                if method == "fro":
                    variants.append((method, 0.0))
                    continue
                for ridge_scale in args.schatten4_ridge_scales:
                    variants.append((method, ridge_scale))

            for i, case_name in enumerate(args.cases):
                is_stress = case_name in stress_cases
                pe5_ell0 = stress_ell0 if is_stress else args.ell0

                raw_case = make_case(
                    case_name,
                    args.rows,
                    args.cols,
                    device=device,
                    seed=args.seed + 1000 * i,
                )
                norm_ref = _reference_norms(raw_case.a)

                ref = None
                if args.reference != "none":
                    ref_dtype = (
                        torch.float64 if args.reference == "fp64" else torch.float32
                    )
                    ref = polar_reference(raw_case.a, dtype=ref_dtype)
                    if args.audit and ref.inv_sqrt.is_cuda:
                        torch.cuda.empty_cache()

                case_coeffs = coeffs
                if is_stress and pe5_ell0 != args.ell0:
                    case_coeffs = pe5_coefficients(ell0=pe5_ell0, steps=5)

                for normalizer, ridge_scale in variants:
                    a, normalization = normalize_matrix(
                        raw_case.a,
                        method=normalizer,
                        eps=PAPER_NORM_EPS,
                        schatten4_ridge_scale=ridge_scale,
                        schatten4_ridge_stat=args.schatten4_ridge_stat,
                    )
                    a = a.contiguous()

                    methods: dict[str, Callable[[], object]] = {
                        "dwh2": lambda a=a, ell=args.ell0: dwh2(a, ell0=ell),
                        "pe5": lambda a=a, cs=case_coeffs, ell=pe5_ell0: pe5(
                            a, ell0=ell, coeffs=cs
                        ),
                    }

                    for method_name in args.methods:
                        out, times = measure(
                            methods[method_name], args.trials, args.warmup
                        )
                        assert isinstance(out, PolarResult)

                        base = asdict(
                            summarize(
                                case=Case(name=raw_case.name, a=raw_case.a),
                                q=out.q,
                                ref=ref,
                                times=times,
                                method=method_name,
                                normalization=normalization,
                                trials=args.trials,
                                stats=out.stats,
                                is_stress=is_stress,
                                audit=args.audit,
                                audit_device=args.audit_device,
                                audit_chunk_rows=args.audit_chunk_rows,
                            )
                        )
                        scale = normalization.scale
                        row = {
                            **base,
                            **norm_ref,
                            "normalizer": _normalizer_label(
                                normalizer, ridge_scale, args.schatten4_ridge_stat
                            ),
                            "schatten4_ridge_scale": ridge_scale,
                            "schatten4_ridge_stat": normalization.ridge_stat,
                            "est_over_true_fro": scale / norm_ref["true_fro"],
                            "est_over_true_schatten4": scale
                            / norm_ref["true_schatten4"],
                            "est_over_true_spectral": scale / norm_ref["true_spectral"],
                            "under_true_schatten4": scale < norm_ref["true_schatten4"],
                            "under_true_spectral": scale < norm_ref["true_spectral"],
                        }
                        line = json.dumps(row)
                        if not args.quiet:
                            print(line, flush=True)
                        if out_f is not None:
                            out_f.write(line + "\n")
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
