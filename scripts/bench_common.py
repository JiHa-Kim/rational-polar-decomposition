from __future__ import annotations

import math
from typing import Final

import torch

DTYPE_MAP: Final = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

DEFAULT_CASES: Final = (
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
)

DEFAULT_SHAPES: Final = (
    "16384x4096",
    "8192x4096",
    "4096x4096",
)


class CaseGenerator:
    @staticmethod
    def _randn(shape, device, seed: int, dtype: torch.dtype) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    @classmethod
    def make_case(cls, name: str, m: int, n: int, device, seed: int) -> torch.Tensor:
        randn = cls._randn
        f32 = torch.float32

        if name == "gaussian":
            return randn((m, n), device, seed, f32)

        if name == "lognormal_cols":
            x = randn((m, n), device, seed, f32)
            scales = torch.exp(1.5 * randn((n,), device, seed + 1, f32))
            scales = scales / scales.median().clamp_min(1e-8)
            return x * scales[None, :]

        if name == "ar1_cols":
            rho = 0.995
            x = randn((m, n), device, seed, f32)
            coeff = math.sqrt(max(1.0 - rho * rho, 0.0))
            powers = torch.pow(
                torch.tensor(rho, device=device, dtype=x.dtype),
                torch.arange(n, device=device, dtype=x.dtype),
            )
            scale = torch.full((n,), coeff, device=device, dtype=x.dtype)
            scale[0] = 1.0
            scale[1:] /= powers[1:]
            x = torch.cumsum(x * scale[None, :], dim=1)
            x.mul_(powers[None, :])
            return x

        if name == "duplicate_cols":
            k = max(64, n // 16)
            base = randn((m, k), device, seed, f32)
            reps = (n + k - 1) // k
            tiled = base.repeat(1, reps)[:, :n]
            noise = 1e-3 * randn((m, n), device, seed + 1, f32)
            return tiled + noise

        if name == "lowrank_noise":
            r = min(64, n // 8)
            u = randn((m, r), device, seed, f32)
            v = randn((r, n), device, seed + 1, f32)
            noise = 1e-3 * randn((m, n), device, seed + 2, f32)
            return u @ v + noise

        if name == "ill_conditioned":
            x = randn((m, n), device, seed, f32)
            v = torch.linalg.qr(randn((n, n), device, seed + 1, f32))[0]
            s = torch.logspace(0, -6, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v

        if name == "heavy_tail_t":
            z = randn((m, n), device, seed, f32)
            chi2 = randn((m, n), device, seed + 1, f32).square_()
            tail = randn((m, n), device, seed + 2, f32)
            chi2.addcmul_(tail, tail).mul_(0.5).clamp_min_(1e-4).sqrt_()
            return z / chi2

        if name == "sparse_like":
            base = randn((m, n), device, seed, f32)
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 1)
            mask = torch.rand((m, n), device=device, generator=gen) > 0.95
            return base * mask.float()

        if name == "orthogonal_noisy":
            x = 1e-4 * randn((m, n), device, seed + 1, f32)
            k = min(m, n)
            x[:k, :k].diagonal().add_(1.0)
            return x

        if name == "rank_1_heavy":
            u = randn((m, 1), device, seed, f32)
            v = randn((1, n), device, seed + 1, f32)
            noise = 1e-6 * randn((m, n), device, seed + 2, f32)
            return u @ v + noise

        if name == "adversarial_condition":
            x = randn((m, n), device, seed, f32)
            v = torch.linalg.qr(randn((n, n), device, seed + 1, f32))[0]
            s = torch.linspace(1.0, 1e-7, steps=n, device=device, dtype=x.dtype)
            return (x * s[None, :]) @ v

        raise ValueError(f"Unknown case: {name}")


class MetricsSuite:
    @staticmethod
    def _symmetrize_(a: torch.Tensor, scratch: torch.Tensor) -> None:
        scratch.copy_(a.mT)
        a.add_(scratch).mul_(0.5)

    @staticmethod
    def _views(
        a_norm: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a_norm.shape[0] < a_norm.shape[1]:
            return a_norm.mT, q.mT
        return a_norm, q

    @staticmethod
    def _get_metric_qbuf(workspace) -> torch.Tensor:
        qbuf = getattr(workspace, "_metric_qbuf", None)
        xbuf = workspace.xbuf
        if qbuf is None or qbuf.shape != xbuf.shape or qbuf.dtype != torch.float32:
            qbuf = torch.empty_like(xbuf)
            setattr(workspace, "_metric_qbuf", qbuf)
        return qbuf

    @classmethod
    def all_stats(
        cls,
        a_norm: torch.Tensor,
        q: torch.Tensor,
        gram_norm: torch.Tensor,
        workspace,
    ) -> dict[str, float]:
        X, Q = cls._views(a_norm, q)
        m, _ = X.shape

        eps = 1e-30
        S = workspace.gram
        H = workspace.buf
        xbuf = workspace.xbuf
        scratch = workspace.scratch
        tmp = workspace.tmp
        qbuf = cls._get_metric_qbuf(workspace)

        S.zero_()
        H.zero_()
        for start in range(0, m, int(workspace.block_rows)):
            rows = min(int(workspace.block_rows), m - start)
            xbuf[:rows].copy_(X[start : start + rows])
            qbuf[:rows].copy_(Q[start : start + rows])
            S.addmm_(qbuf[:rows].mT, qbuf[:rows])
            H.addmm_(qbuf[:rows].mT, xbuf[:rows])

        cls._symmetrize_(S, scratch)

        torch.mm(S, S, out=tmp)
        tmp.sub_(S)
        e_proj_num = torch.sum(tmp * tmp, dtype=torch.float64).sqrt()
        e_proj_den = torch.sum(S * S, dtype=torch.float64).sqrt().clamp_min_(eps)
        e_proj = float((e_proj_num / e_proj_den).item())

        tmp.copy_(S).neg_().diagonal().add_(1.0)
        torch.mm(tmp, gram_norm, out=scratch)
        e_supp_num = torch.sum(scratch * scratch, dtype=torch.float64).sqrt()
        e_supp_den = (
            torch.sum(gram_norm * gram_norm, dtype=torch.float64).sqrt().clamp_min_(eps)
        )
        e_supp = float((e_supp_num / e_supp_den).item())

        P = workspace.m0
        P.copy_(H.mT).add_(H).mul_(0.5)

        skew = scratch.copy_(H).sub_(P)
        e_skew_num = torch.sum(skew * skew, dtype=torch.float64).sqrt()
        e_skew_den = torch.sum(P * P, dtype=torch.float64).sqrt().clamp_min_(eps)
        e_skew = float((e_skew_num / e_skew_den).item())

        torch.mm(P, P, out=tmp)
        tmp.sub_(gram_norm)
        e_p2_num = torch.sum(tmp * tmp, dtype=torch.float64).sqrt()
        e_p2 = float((e_p2_num / e_supp_den).item())

        trG = float(torch.diagonal(gram_norm).sum(dtype=torch.float64).item())
        trPTH = float(torch.sum(P * H, dtype=torch.float64).item())
        torch.mm(S, P, out=tmp)
        trPTSP = float(torch.sum(P * tmp, dtype=torch.float64).item())

        e_rec_sq = max(trG - 2.0 * trPTH + trPTSP, 0.0)
        e_rec = float(math.sqrt(e_rec_sq) / (math.sqrt(max(trG, 0.0)) + eps))
        r_stable = float((trG * trG / (e_supp_den * e_supp_den)).item())

        return {
            "ortho_proj": e_proj,
            "ortho_supp": e_supp,
            "p_skew_rel_fro": e_skew,
            "p2_gram_rel_fro": e_p2,
            "rec_resid": e_rec,
            "stable_rank": r_stable,
        }
