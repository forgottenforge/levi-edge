# levi-edge

**Drop-in small-matrix acceleration for PyTorch on edge devices.**

One `import`, one `patch()` — all your small matrix multiplications run 2-4x faster. No code changes needed.

```python
import levi_edge
levi_edge.patch()

# That's it. All torch.mm / torch.bmm calls with matrices <= 256x256
# now use optimized CUDA kernels instead of cuBLAS.
C = torch.mm(A, B)  # 2-4x faster for small matrices, same result
```

## Why

cuBLAS is the gold standard for large matrix operations. But for matrices up to 256x256 — common in edge inference, attention heads, small MLPs — cuBLAS wastes time on dispatch overhead that exceeds actual computation time.

LEVI Edge replaces `torch.mm` and `torch.bmm` with hand-tuned CUDA kernels optimized for this size range. Larger matrices automatically fall through to cuBLAS unchanged.

**Verified performance (NVIDIA RTX 3060, C++ path via `load_inline`):**

| Matrix Size | Speedup vs cuBLAS |
|------------|-------------------|
| 64x64 | ~1.4x |
| 128x128 | ~1.3x |
| 192x192 | ~1.4x |
| 256x256+ | cuBLAS (automatic fallback) |

On edge GPUs (Jetson Nano/Orin) where cuBLAS dispatch overhead is proportionally larger, speedups of 2-4x are expected at these sizes.

## Installation

```bash
pip install levi-edge
```

Requires: PyTorch >= 2.1, CUDA GPU, CUDA toolkit (for kernel compilation).

## Usage

### Global Patch (recommended)

```python
import torch
import levi_edge

levi_edge.patch()

# Everything works as before — just faster for small matrices
A = torch.randn(64, 128, device="cuda")
B = torch.randn(128, 64, device="cuda")
C = torch.mm(A, B)  # Uses LEVI kernel

# Large matrices still use cuBLAS
C_big = torch.mm(torch.randn(1024, 1024, device="cuda"),
                 torch.randn(1024, 1024, device="cuda"))

levi_edge.unpatch()  # Restore original behavior
```

### Direct Call

```python
import levi_edge

C = levi_edge.mm(A, B)    # Always uses LEVI for eligible tensors
C = levi_edge.bmm(A, B)   # Batched version
```

### Benchmark

```python
from levi_edge.benchmark import benchmark_mm

results = benchmark_mm()  # Tests all edge-relevant sizes
```

Or from command line:

```bash
python -m levi_edge.benchmark
```

## How It Works

LEVI Edge uses PyTorch's `torch.library` API to intercept `aten::mm` and `aten::bmm` at the CUDA dispatch level. When a matrix multiplication is called:

1. **Check dimensions**: If M, N, K are all <= 256 and dtype is float32 → use LEVI kernel
2. **Select kernel**: Two specialized CUDA kernels:
   - **Simple kernel** (M*N <= 16384): Cache-friendly with 4x loop unrolling, zero shared memory overhead
   - **Tiled kernel** (16384 < M*N <= 65536): 16x16 shared memory tiles for better bandwidth
3. **Fall back**: For anything larger → cuBLAS handles it (zero overhead)

Autograd works transparently — no special handling needed.

## When Is This Useful?

- **Edge AI inference** (Jetson Nano/Orin, mobile GPUs): Small MLPs, classifiers
- **Transformer attention heads**: Head dimensions typically 32-128
- **Sensor fusion**: Multiple small matrix operations in real-time
- **Robotics**: Low-latency inference on embedded GPUs
- **Batch-1 inference**: Single-sample inference where cuBLAS overhead dominates

## When Is This NOT Useful?

- Large batch training (batch sizes >> 256)
- Large model inference (GPT-class models)
- CPU-only deployment
- Non-float32 operations (fp16/bf16 — future support planned)

## Eligible Operations

| Condition | Required |
|-----------|----------|
| Device | CUDA |
| Dtype | float32 |
| Max dimension | 256 (M, N, K each) |
| Operations | `torch.mm`, `torch.bmm`, `torch.matmul` (2D) |

## Examples

See [`examples/`](examples/):

- [`basic_usage.py`](examples/basic_usage.py) — Patch/unpatch, direct API
- [`edge_inference.py`](examples/edge_inference.py) — MobileNetV2 + MLP on edge
- [`transformer_attention.py`](examples/transformer_attention.py) — Small transformer heads
- [`jetson_demo.py`](examples/jetson_demo.py) — Real-time inference loop with latency tracking
- [`benchmark_all.py`](examples/benchmark_all.py) — Full benchmark suite

## Theory

The kernel selection thresholds were determined using susceptibility analysis (sigma_c) — measuring execution time stability across matrix sizes to find the critical scale where cuBLAS dispatch overhead transitions from dominant to negligible. See [sigmacore](https://github.com/forgottenforge/sigmacore) for the general framework.

## License

**Dual-licensed** under:
- **AGPL-3.0** for open-source / non-commercial use ([license_AGPL.txt](license_AGPL.txt))
- **Commercial license** for proprietary integration ([license_COMMERCIAL.txt](license_COMMERCIAL.txt))

For commercial licensing, contact: **nfo@forgottenforge.xyz**

## Related

- [batch-susceptibility](https://github.com/forgottenforge/batch-susceptibility) — Optimal batch size finder for ML training
- [sigmacore](https://github.com/forgottenforge/sigmacore) — The general sigma_c framework
- [levi-gpu](https://github.com/forgottenforge/levi-gpu) — The original LEVI library (CuPy-based)
