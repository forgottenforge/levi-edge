# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""PyTorch aten dispatch override for small-matrix GEMM.

Transparently replaces torch.mm / torch.bmm with LEVI kernels when
matrix dimensions are <= 256. Falls back to cuBLAS for larger matrices.

Supports two compilation backends:
1. torch.utils.cpp_extension.load_inline (needs MSVC/gcc + CUDA toolkit)
2. CuPy RawKernel (needs cupy — works without C++ compiler)
"""

import torch
import warnings
import os
import logging

logger = logging.getLogger("levi_edge")

# Compiled kernel functions (set by _compile_kernels)
_levi_mm_fn = None
_levi_bmm_fn = None
_compiled = False
_patched = False
_original_mm = None
_original_bmm = None
_lib = None

# Thresholds (from sigma_c analysis on edge GPUs)
LEVI_MAX_DIM = 256
SIMPLE_THRESHOLD = 16384  # M*N <= this → simple kernel


def _compile_cupy_kernels():
    """Compile CUDA kernels via CuPy RawKernel (no C++ compiler needed)."""
    import cupy as cp

    simple_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void levi_simple_f32(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        const int M, const int N, const int K
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            int k = 0;
            for (; k <= K - 4; k += 4) {
                sum += A[row * K + k]     * B[k       * N + col];
                sum += A[row * K + k + 1] * B[(k + 1) * N + col];
                sum += A[row * K + k + 2] * B[(k + 2) * N + col];
                sum += A[row * K + k + 3] * B[(k + 3) * N + col];
            }
            for (; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    ''', 'levi_simple_f32', options=('--use_fast_math',))

    tiled_kernel = cp.RawKernel(r'''
    #define TILE_SIZE 16

    extern "C" __global__
    void levi_tiled_f32(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        const int M, const int N, const int K
    ) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        float sum = 0.0f;

        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            As[ty][tx] = (row < M && t * TILE_SIZE + tx < K) ?
                A[row * K + t * TILE_SIZE + tx] : 0.0f;
            Bs[ty][tx] = (t * TILE_SIZE + ty < K && col < N) ?
                B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    ''', 'levi_tiled_f32', options=('--use_fast_math',))

    batched_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void levi_batched_f32(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        const int batch_count,
        const int M, const int N, const int K
    ) {
        int batch = blockIdx.z;
        if (batch >= batch_count) return;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            const float* A_b = A + batch * M * K;
            const float* B_b = B + batch * K * N;
            float* C_b = C + batch * M * N;

            float sum = 0.0f;
            int k = 0;
            for (; k <= K - 4; k += 4) {
                sum += A_b[row * K + k]     * B_b[k       * N + col];
                sum += A_b[row * K + k + 1] * B_b[(k + 1) * N + col];
                sum += A_b[row * K + k + 2] * B_b[(k + 2) * N + col];
                sum += A_b[row * K + k + 3] * B_b[(k + 3) * N + col];
            }
            for (; k < K; k++) {
                sum += A_b[row * K + k] * B_b[k * N + col];
            }
            C_b[row * N + col] = sum;
        }
    }
    ''', 'levi_batched_f32', options=('--use_fast_math',))

    return simple_kernel, tiled_kernel, batched_kernel


def _compile_kernels():
    """Compile CUDA kernels. Tries load_inline first, falls back to CuPy."""
    global _levi_mm_fn, _levi_bmm_fn, _compiled

    if _compiled:
        return _levi_mm_fn is not None

    # Try 1: torch.utils.cpp_extension (fastest, needs C++ compiler)
    try:
        from .kernels import CUDA_SOURCE, CPP_SOURCE
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="levi_edge_kernels",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["levi_mm", "levi_bmm", "should_use_levi"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "--allow-unsupported-compiler"],
            verbose=os.environ.get("LEVI_VERBOSE", "") == "1",
        )
        _levi_mm_fn = module.levi_mm
        _levi_bmm_fn = module.levi_bmm
        _compiled = True
        logger.info("LEVI Edge: compiled via load_inline (C++ path)")
        return True
    except Exception as e:
        logger.info(f"load_inline unavailable ({e}), trying CuPy...")

    # Try 2: CuPy RawKernel (no C++ compiler needed)
    try:
        import cupy as cp

        simple_k, tiled_k, batched_k = _compile_cupy_kernels()

        def _wrap_torch_cupy(t):
            """Zero-copy wrap PyTorch tensor as CuPy array via raw pointer.

            ~6x faster than cp.from_dlpack() (0.02ms vs 0.13ms per tensor).
            """
            mem = cp.cuda.UnownedMemory(
                t.data_ptr(), t.nelement() * t.element_size(), t
            )
            memptr = cp.cuda.MemoryPointer(mem, 0)
            return cp.ndarray(t.shape, dtype=cp.float32, memptr=memptr)

        def _cupy_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            M, K = a.shape
            _, N = b.shape
            C = torch.empty(M, N, device=a.device, dtype=a.dtype)

            # Raw-pointer wrap: ~0.02ms total vs ~0.13ms with DLPack
            A_cp = _wrap_torch_cupy(a)
            B_cp = _wrap_torch_cupy(b)
            C_cp = _wrap_torch_cupy(C)

            threads = (16, 16)
            blocks = ((N + 15) // 16, (M + 15) // 16)

            if M * N <= SIMPLE_THRESHOLD:
                simple_k(blocks, threads, (A_cp, B_cp, C_cp, M, N, K))
            else:
                tiled_k(blocks, threads, (A_cp, B_cp, C_cp, M, N, K))

            return C

        def _cupy_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, M, K = a.shape
            _, _, N = b.shape
            C = torch.empty(batch, M, N, device=a.device, dtype=a.dtype)

            A_cp = _wrap_torch_cupy(a)
            B_cp = _wrap_torch_cupy(b)
            C_cp = _wrap_torch_cupy(C)

            threads = (16, 16)
            blocks = ((N + 15) // 16, (M + 15) // 16, batch)

            batched_k(blocks, threads, (A_cp, B_cp, C_cp, batch, M, N, K))

            return C

        _levi_mm_fn = _cupy_mm
        _levi_bmm_fn = _cupy_bmm
        _compiled = True
        logger.info("LEVI Edge: compiled via CuPy RawKernel")
        return True

    except Exception as e:
        logger.warning(f"CuPy kernels also failed: {e}")
        _compiled = True  # Mark as tried
        return False


def _is_eligible(tensor_a, tensor_b, op="mm"):
    """Check if tensors qualify for LEVI acceleration."""
    if not (tensor_a.is_cuda and tensor_b.is_cuda):
        return False
    if tensor_a.dtype != torch.float32:
        return False

    if op == "mm":
        if tensor_a.dim() != 2 or tensor_b.dim() != 2:
            return False
        M, K = tensor_a.shape
        _, N = tensor_b.shape
        return M <= LEVI_MAX_DIM and N <= LEVI_MAX_DIM and K <= LEVI_MAX_DIM
    elif op == "bmm":
        if tensor_a.dim() != 3 or tensor_b.dim() != 3:
            return False
        _, M, K = tensor_a.shape
        _, _, N = tensor_b.shape
        return M <= LEVI_MAX_DIM and N <= LEVI_MAX_DIM and K <= LEVI_MAX_DIM
    return False


def patch():
    """Activate LEVI Edge: override torch.mm/bmm for small matrices.

    After calling this, all torch.mm and torch.bmm calls with matrices
    <= 256x256 (float32, CUDA) will use LEVI kernels automatically.
    Larger matrices fall through to cuBLAS unchanged.

    Example::

        import levi_edge
        levi_edge.patch()

        # Now all small matmuls are accelerated
        C = torch.mm(A, B)  # uses LEVI if A,B are small

        levi_edge.unpatch()  # restore original behavior
    """
    global _patched, _original_mm, _original_bmm, _lib

    if _patched:
        logger.info("LEVI Edge already patched")
        return

    if not _compile_kernels():
        warnings.warn(
            "LEVI Edge: kernel compilation failed. "
            "Install CuPy or a C++ compiler with CUDA toolkit. "
            "Falling back to default PyTorch kernels.",
            RuntimeWarning,
        )
        return

    # Store original kernels for fallback
    try:
        _original_mm = torch.library.get_kernel("aten::mm.default", "CUDA")
    except (AttributeError, RuntimeError):
        _original_mm = None

    try:
        _original_bmm = torch.library.get_kernel("aten::bmm.default", "CUDA")
    except (AttributeError, RuntimeError):
        _original_bmm = None

    # Register overrides
    _lib = torch.library.Library("aten", "IMPL")

    def levi_mm_dispatch(self, mat2):
        if _is_eligible(self, mat2, "mm"):
            return _levi_mm_fn(self.contiguous(), mat2.contiguous())
        if _original_mm is not None:
            return _original_mm(self, mat2)
        return torch._C._VariableFunctions.mm(self, mat2)

    def levi_bmm_dispatch(self, mat2):
        if _is_eligible(self, mat2, "bmm"):
            return _levi_bmm_fn(self.contiguous(), mat2.contiguous())
        if _original_bmm is not None:
            return _original_bmm(self, mat2)
        return torch._C._VariableFunctions.bmm(self, mat2)

    try:
        _lib.impl("mm", levi_mm_dispatch, "CUDA")
        _lib.impl("bmm", levi_bmm_dispatch, "CUDA")
        _patched = True
        logger.info(
            "LEVI Edge active: mm/bmm accelerated for matrices <= %dx%d",
            LEVI_MAX_DIM,
            LEVI_MAX_DIM,
        )
    except Exception as e:
        logger.warning(f"Failed to register LEVI dispatch: {e}")
        _lib = None


def unpatch():
    """Deactivate LEVI Edge, restore original PyTorch kernels."""
    global _patched, _lib

    if not _patched:
        return

    if _lib is not None:
        del _lib
        _lib = None

    _patched = False
    logger.info("LEVI Edge deactivated")


def is_active():
    """Return True if LEVI Edge is currently patching torch.mm/bmm."""
    return _patched


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Direct LEVI matrix multiply (without dispatch override).

    Always uses LEVI kernels for eligible tensors, regardless of
    whether patch() has been called.

    Args:
        a: (M, K) float32 CUDA tensor
        b: (K, N) float32 CUDA tensor

    Returns:
        (M, N) float32 CUDA tensor
    """
    _compile_kernels()
    if _levi_mm_fn is not None and _is_eligible(a, b, "mm"):
        return _levi_mm_fn(a.contiguous(), b.contiguous())
    return torch.mm(a, b)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Direct LEVI batched matrix multiply.

    Args:
        a: (B, M, K) float32 CUDA tensor
        b: (B, K, N) float32 CUDA tensor

    Returns:
        (B, M, N) float32 CUDA tensor
    """
    _compile_kernels()
    if _levi_bmm_fn is not None and _is_eligible(a, b, "bmm"):
        return _levi_bmm_fn(a.contiguous(), b.contiguous())
    return torch.bmm(a, b)
