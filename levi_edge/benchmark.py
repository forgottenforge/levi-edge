# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Benchmarking tools for LEVI Edge vs cuBLAS."""

import torch
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json


@dataclass
class BenchmarkResult:
    """Result of a single matrix size benchmark."""
    size_m: int
    size_n: int
    size_k: int
    cublas_ms: float
    levi_ms: float
    speedup: float
    max_error: float
    passed: bool
    kernel: str


def _time_fn(fn, warmup=10, repeats=50):
    """Time a CUDA function using torch.cuda.Event for accuracy."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    # Use median for stability
    return times[len(times) // 2]


def benchmark_mm(
    sizes: Optional[List[int]] = None,
    repeats: int = 50,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Benchmark LEVI Edge vs cuBLAS across matrix sizes.

    Args:
        sizes: List of square matrix sizes to test. Default: edge-relevant sizes.
        repeats: Number of timed iterations per size.
        verbose: Print results to stdout.

    Returns:
        List of BenchmarkResult for each size.
    """
    from . import backend

    if sizes is None:
        sizes = [16, 32, 64, 96, 128, 192, 256, 384, 512]

    module = backend._compile_kernels()
    if module is None:
        raise RuntimeError(
            "Cannot compile LEVI kernels. Ensure CUDA toolkit is installed."
        )

    results = []

    if verbose:
        print("=" * 70)
        print("LEVI Edge Benchmark: LEVI kernels vs cuBLAS")
        print("=" * 70)
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu}")
        print(f"Repeats: {repeats}, timing: median")
        print("-" * 70)
        print(f"{'Size':>8} {'cuBLAS ms':>12} {'LEVI ms':>12} {'Speedup':>10} "
              f"{'Error':>12} {'Status':>8}")
        print("-" * 70)

    for size in sizes:
        M = N = K = size
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # cuBLAS timing
        t_cublas = _time_fn(lambda: torch.mm(A, B), repeats=repeats)

        # LEVI timing
        use_levi = M <= 256 and N <= 256 and K <= 256
        if use_levi:
            t_levi = _time_fn(
                lambda: module.levi_mm(A, B), repeats=repeats
            )
            kernel = "simple" if M * N <= 16384 else "tiled"
        else:
            t_levi = t_cublas
            kernel = "cublas-fallback"

        # Validate
        C_ref = torch.mm(A, B)
        if use_levi:
            C_levi = module.levi_mm(A, B)
            max_err = float((C_levi - C_ref).abs().max())
        else:
            max_err = 0.0

        passed = max_err < 1e-3
        speedup = t_cublas / t_levi if t_levi > 0 else 0.0

        result = BenchmarkResult(
            size_m=M, size_n=N, size_k=K,
            cublas_ms=t_cublas, levi_ms=t_levi,
            speedup=speedup, max_error=max_err,
            passed=passed, kernel=kernel,
        )
        results.append(result)

        if verbose:
            marker = "LEVI" if speedup > 1.05 else ("~" if speedup > 0.95 else "cuBLAS")
            print(
                f"{size:>4}x{size:<4} {t_cublas:>10.4f}ms {t_levi:>10.4f}ms "
                f"{speedup:>8.2f}x  {max_err:>10.2e}  "
                f"{'OK' if passed else 'FAIL':>6} {marker}"
            )

    if verbose:
        print("-" * 70)
        wins = [r for r in results if r.speedup > 1.05 and r.passed]
        if wins:
            avg_speedup = sum(r.speedup for r in wins) / len(wins)
            best = max(wins, key=lambda r: r.speedup)
            print(f"LEVI wins on {len(wins)}/{len(results)} sizes")
            print(f"Average speedup (where LEVI wins): {avg_speedup:.2f}x")
            print(f"Best: {best.size_m}x{best.size_n} at {best.speedup:.2f}x")
        else:
            print("No sizes where LEVI outperformed cuBLAS.")
        print("=" * 70)

    return results


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    repeats: int = 50,
    verbose: bool = True,
) -> Dict:
    """Benchmark a PyTorch model with and without LEVI Edge.

    Args:
        model: PyTorch model (must be on CUDA).
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224)).
        repeats: Number of timed forward passes.
        verbose: Print results.

    Returns:
        Dict with baseline_ms, levi_ms, speedup.
    """
    from . import backend

    model = model.cuda().eval()
    x = torch.randn(*input_shape, device="cuda")

    # Baseline (unpatch if currently patched)
    was_active = backend.is_active()
    if was_active:
        backend.unpatch()

    with torch.no_grad():
        t_baseline = _time_fn(lambda: model(x), repeats=repeats)

    # LEVI
    backend.patch()
    with torch.no_grad():
        t_levi = _time_fn(lambda: model(x), repeats=repeats)

    if not was_active:
        backend.unpatch()

    speedup = t_baseline / t_levi if t_levi > 0 else 0.0

    result = {
        "baseline_ms": t_baseline,
        "levi_ms": t_levi,
        "speedup": speedup,
        "input_shape": list(input_shape),
    }

    if verbose:
        print(f"Model benchmark ({model.__class__.__name__}):")
        print(f"  Baseline: {t_baseline:.2f} ms")
        print(f"  LEVI:     {t_levi:.2f} ms")
        print(f"  Speedup:  {speedup:.2f}x")

    return result


def save_report(results: List[BenchmarkResult], path: str = "levi_benchmark.json"):
    """Save benchmark results to JSON."""
    data = {
        "tool": "levi-edge",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Report saved to {path}")
