#!/usr/bin/env python3
# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Edge inference benchmark: MobileNetV2 with LEVI Edge.

Shows speedup on a typical edge-AI model (MobileNetV2) where many
internal operations use small matrix multiplications.
"""

import torch
import levi_edge
from levi_edge.benchmark import _time_fn


def main():
    print("LEVI Edge — Edge Inference Demo")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load a typical edge model
    try:
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None, num_classes=10)
    except ImportError:
        print("torchvision not installed. Install with: pip install torchvision")
        return

    model = model.cuda().eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")

    # Baseline
    print("\nBaseline (cuBLAS)...")
    with torch.no_grad():
        t_base = _time_fn(lambda: model(x), warmup=20, repeats=100)
    print(f"  Inference time: {t_base:.2f} ms")

    # With LEVI Edge
    print("\nWith LEVI Edge...")
    levi_edge.patch()
    with torch.no_grad():
        t_levi = _time_fn(lambda: model(x), warmup=20, repeats=100)
    levi_edge.unpatch()
    print(f"  Inference time: {t_levi:.2f} ms")

    speedup = t_base / t_levi
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.0:
        print(f"LEVI Edge saves {t_base - t_levi:.2f} ms per inference")
        print(f"At 30 FPS that's {(t_base - t_levi) * 30 / 1000:.1f}s saved per second")
    else:
        print("MobileNetV2 uses mostly large convolutions — LEVI benefit is small.")
        print("Try models with more small linear layers (transformers, MLPs).")

    # Also test with a small MLP (more representative of edge workloads)
    print("\n\n--- Small MLP (typical edge classifier) ---")
    mlp = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ).cuda().eval()

    x_mlp = torch.randn(1, 128, device="cuda")

    with torch.no_grad():
        t_base = _time_fn(lambda: mlp(x_mlp), warmup=50, repeats=200)

    levi_edge.patch()
    with torch.no_grad():
        t_levi = _time_fn(lambda: mlp(x_mlp), warmup=50, repeats=200)
    levi_edge.unpatch()

    print(f"Baseline: {t_base:.4f} ms")
    print(f"LEVI:     {t_levi:.4f} ms")
    print(f"Speedup:  {t_base / t_levi:.2f}x")


if __name__ == "__main__":
    main()
