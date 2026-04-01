from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - fallback when Triton is unavailable
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None and tl is not None


if TRITON_AVAILABLE:

    @triton.jit
    def _scale_symmetrize_kernel(
        src_ptr,
        dst_ptr,
        s_ptr,
        n,
        stride_src_0,
        stride_src_1,
        stride_dst_0,
        stride_dst_1,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)

        a = tl.load(
            src_ptr + offs_m[:, None] * stride_src_0 + offs_n[None, :] * stride_src_1,
            mask=mask,
            other=0.0,
        )
        at = tl.load(
            src_ptr + offs_n[None, :] * stride_src_0 + offs_m[:, None] * stride_src_1,
            mask=mask,
            other=0.0,
        )
        s_m = tl.load(s_ptr + offs_m, mask=offs_m < n, other=0.0)
        s_n = tl.load(s_ptr + offs_n, mask=offs_n < n, other=0.0)
        out = 0.5 * (a + at) * (s_m[:, None] * s_n[None, :])

        tl.store(
            dst_ptr + offs_m[:, None] * stride_dst_0 + offs_n[None, :] * stride_dst_1,
            out,
            mask=mask,
        )

    @triton.jit
    def _affine_diag_kernel(
        src_ptr,
        dst_ptr,
        n,
        alpha,
        diag_add,
        stride_src_0,
        stride_src_1,
        stride_dst_0,
        stride_dst_1,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)

        x = tl.load(
            src_ptr + offs_m[:, None] * stride_src_0 + offs_n[None, :] * stride_src_1,
            mask=mask,
            other=0.0,
        )
        y = x * alpha + tl.where(
            offs_m[:, None] == offs_n[None, :],
            diag_add,
            0.0,
        )
        tl.store(
            dst_ptr + offs_m[:, None] * stride_dst_0 + offs_n[None, :] * stride_dst_1,
            y,
            mask=mask,
        )


def can_fuse_scale_symmetrize(a: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and a.is_cuda
        and a.dtype == torch.float32
        and a.ndim == 2
        and a.shape[0] == a.shape[1]
        and a.shape[0] >= 1024
        and a.is_contiguous()
        and out.is_contiguous()
    )


def can_affine_diag(src: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and src.is_cuda
        and src.dtype == torch.float32
        and src.ndim == 2
        and src.shape[0] == src.shape[1]
        and src.shape[0] >= 1024
        and src.is_contiguous()
        and out.is_contiguous()
    )


def scale_symmetrize(
    a: torch.Tensor,
    s: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not can_fuse_scale_symmetrize(a, out):
        raise RuntimeError(
            "Triton scale-symmetrize kernel is unavailable for this input"
        )

    n = a.shape[0]
    grid = (triton.cdiv(n, 32), triton.cdiv(n, 32))
    _scale_symmetrize_kernel[grid](
        a,
        out,
        s,
        n,
        a.stride(0),
        a.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=32,
        block_n=32,
        num_warps=4,
    )
    return out


def affine_diag(
    src: torch.Tensor,
    out: torch.Tensor,
    *,
    alpha: float,
    diag_add: float,
) -> torch.Tensor:
    if not can_affine_diag(src, out):
        raise RuntimeError("Triton affine-diag kernel is unavailable for this input")

    n = src.shape[0]
    grid = (triton.cdiv(n, 32), triton.cdiv(n, 32))
    _affine_diag_kernel[grid](
        src,
        out,
        n,
        alpha,
        diag_add,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=32,
        block_n=32,
        num_warps=4,
    )
    return out
