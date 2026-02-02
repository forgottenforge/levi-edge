# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""CUDA kernel source code for LEVI Edge.

Contains optimized GEMM kernels for small matrices (<=256x256) that
outperform cuBLAS by avoiding its dispatch overhead.
"""

# --------------------------------------------------------------------------- #
# Kernel 1: Simple cache-friendly kernel for tiny matrices (up to 128x128)
#
# Strategy: Each thread computes one output element. 4x loop unrolling
# reduces loop overhead. No shared memory = no sync overhead.
# Best for: M*N <= 16384 (128x128)
# --------------------------------------------------------------------------- #

SIMPLE_KERNEL = r'''
#include <cuda_fp16.h>

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

        // 4x unrolled inner loop
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

extern "C" __global__
void levi_simple_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }

        C[row * N + col] = sum;
    }
}
'''

# --------------------------------------------------------------------------- #
# Kernel 2: Tiled shared-memory kernel for medium matrices (128x128..256x256)
#
# Strategy: 16x16 tiles loaded into shared memory reduce global memory
# accesses by factor of TILE_SIZE. Full #pragma unroll on inner loop.
# Best for: 16384 < M*N <= 65536 (256x256)
# --------------------------------------------------------------------------- #

TILED_KERNEL = r'''
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

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

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
'''

# --------------------------------------------------------------------------- #
# Kernel 3: Batched small mm — for torch.bmm override
# --------------------------------------------------------------------------- #

BATCHED_SIMPLE_KERNEL = r'''
extern "C" __global__
void levi_batched_simple_f32(
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
'''

# --------------------------------------------------------------------------- #
# Combined source for load_inline compilation
# --------------------------------------------------------------------------- #

CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ---- Simple kernel (small matrices) ----
__global__ void levi_simple_f32(
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

// ---- Tiled kernel (medium matrices) ----
__global__ void levi_tiled_f32(
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

// ---- Batched simple kernel ----
__global__ void levi_batched_f32(
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

// ==================== PyTorch Bindings ====================

// Threshold: use simple kernel for M*N <= this, tiled for larger
#define SIMPLE_THRESHOLD 16384   // 128x128
#define LEVI_THRESHOLD   65536   // 256x256 — above this, cuBLAS wins

torch::Tensor levi_mm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Expected CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Expected float32");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int total = M * N;
    if (total <= SIMPLE_THRESHOLD) {
        levi_simple_f32<<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(), B.data_ptr<float>(),
            C.data_ptr<float>(), M, N, K);
    } else {
        levi_tiled_f32<<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(), B.data_ptr<float>(),
            C.data_ptr<float>(), M, N, K);
    }

    return C;
}

torch::Tensor levi_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix dimensions");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Expected CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Expected float32");

    A = A.contiguous();
    B = B.contiguous();

    int batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto C = torch::empty({batch, M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, batch);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    levi_batched_f32<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), batch, M, N, K);

    return C;
}

bool should_use_levi(int M, int N, int K) {
    return (M * N <= LEVI_THRESHOLD) && (M <= 256) && (N <= 256) && (K <= 256);
}
'''

CPP_SOURCE = r'''
torch::Tensor levi_mm(torch::Tensor A, torch::Tensor B);
torch::Tensor levi_bmm(torch::Tensor A, torch::Tensor B);
bool should_use_levi(int M, int N, int K);
'''
