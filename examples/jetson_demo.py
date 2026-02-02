#!/usr/bin/env python3
# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Jetson Nano / Orin demo: real-time inference with LEVI Edge.

This example simulates a real-time edge inference pipeline:
- Small classifier MLP (typical for sensor fusion, robotics)
- Continuous inference loop at target FPS
- Shows latency improvement from LEVI Edge
"""

import torch
import time
import levi_edge


def create_edge_model():
    """Small MLP typical for edge inference (sensor fusion, anomaly detection)."""
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
        torch.nn.Softmax(dim=-1),
    )


def run_inference_loop(model, input_size, n_frames, label):
    """Simulate real-time inference and measure latency."""
    latencies = []
    x = torch.randn(1, input_size, device="cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            model(x)
    torch.cuda.synchronize()

    # Timed run
    with torch.no_grad():
        for i in range(n_frames):
            # Simulate new sensor data
            x = torch.randn(1, input_size, device="cuda")

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(x)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))

    latencies.sort()
    median = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"  [{label}] Median: {median:.4f}ms  P95: {p95:.4f}ms  "
          f"P99: {p99:.4f}ms  Max FPS: {1000/median:.0f}")

    return median


def main():
    print("LEVI Edge — Jetson / Edge Device Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()

    model = create_edge_model().cuda().eval()
    n_frames = 1000

    print(f"Model: 4-layer MLP (64 -> 128 -> 64 -> 32 -> 10)")
    print(f"Simulating {n_frames} inference frames...")
    print()

    # Baseline
    print("Without LEVI Edge:")
    t_base = run_inference_loop(model, 64, n_frames, "cuBLAS")

    # With LEVI
    levi_edge.patch()
    print("\nWith LEVI Edge:")
    t_levi = run_inference_loop(model, 64, n_frames, "LEVI")
    levi_edge.unpatch()

    # Summary
    speedup = t_base / t_levi
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Latency reduction: {t_base - t_levi:.4f} ms per frame")

    if speedup > 1.0:
        fps_base = 1000 / t_base
        fps_levi = 1000 / t_levi
        print(f"FPS improvement: {fps_base:.0f} -> {fps_levi:.0f} "
              f"(+{fps_levi - fps_base:.0f} FPS)")

    # Batch inference (multiple sensors)
    print("\n\n--- Batch Inference (8 sensors simultaneously) ---")
    x_batch = torch.randn(8, 64, device="cuda")

    with torch.no_grad():
        # Baseline
        latencies_base = []
        for _ in range(500):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x_batch)
            e.record()
            torch.cuda.synchronize()
            latencies_base.append(s.elapsed_time(e))

        levi_edge.patch()
        latencies_levi = []
        for _ in range(500):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x_batch)
            e.record()
            torch.cuda.synchronize()
            latencies_levi.append(s.elapsed_time(e))
        levi_edge.unpatch()

    latencies_base.sort()
    latencies_levi.sort()
    mb = latencies_base[250]
    ml = latencies_levi[250]
    print(f"  cuBLAS median: {mb:.4f} ms")
    print(f"  LEVI median:   {ml:.4f} ms")
    print(f"  Speedup:       {mb/ml:.2f}x")


if __name__ == "__main__":
    main()
