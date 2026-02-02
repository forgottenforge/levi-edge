#!/usr/bin/env python3
# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Basic usage: accelerate small matrix multiplications in PyTorch."""

import torch
import levi_edge


def main():
    print("LEVI Edge — Basic Usage Demo")
    print("=" * 50)

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. LEVI Edge requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"LEVI Edge: {levi_edge.__version__}")

    # --- Method 1: Global patch (recommended) ---
    print("\n--- Method 1: Global Patch ---")
    levi_edge.patch()
    print(f"Active: {levi_edge.is_active()}")

    A = torch.randn(64, 128, device="cuda")
    B = torch.randn(128, 64, device="cuda")
    C = torch.mm(A, B)  # Automatically uses LEVI kernel
    print(f"torch.mm(64x128, 128x64) = {C.shape}  [uses LEVI]")

    # Large matrices still use cuBLAS
    A_big = torch.randn(1024, 1024, device="cuda")
    B_big = torch.randn(1024, 1024, device="cuda")
    C_big = torch.mm(A_big, B_big)  # Falls through to cuBLAS
    print(f"torch.mm(1024x1024, 1024x1024) = {C_big.shape}  [uses cuBLAS]")

    levi_edge.unpatch()

    # --- Method 2: Direct call ---
    print("\n--- Method 2: Direct Call ---")
    C2 = levi_edge.mm(A, B)
    print(f"levi_edge.mm(64x128, 128x64) = {C2.shape}")

    # Validate correctness
    C_ref = torch.mm(A, B)
    error = (C2 - C_ref).abs().max().item()
    print(f"Max error vs cuBLAS: {error:.2e}")

    # --- Batched matmul ---
    print("\n--- Batched Matrix Multiply ---")
    levi_edge.patch()
    A_batch = torch.randn(8, 32, 64, device="cuda")
    B_batch = torch.randn(8, 64, 32, device="cuda")
    C_batch = torch.bmm(A_batch, B_batch)  # Uses LEVI
    print(f"torch.bmm(8x32x64, 8x64x32) = {C_batch.shape}")
    levi_edge.unpatch()

    print("\nDone!")


if __name__ == "__main__":
    main()
