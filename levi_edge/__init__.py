# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""LEVI Edge: Drop-in small-matrix acceleration for PyTorch.

Replaces torch.mm / torch.bmm with optimized CUDA kernels for matrices
<= 256x256, where cuBLAS dispatch overhead dominates computation time.

Designed for edge devices (Jetson, mobile GPUs) where small matrix
multiplications are the bottleneck.

Quick start::

    import levi_edge
    levi_edge.patch()  # activate — all small matmuls now use LEVI

    C = torch.mm(A, B)  # automatically uses LEVI if A,B are small CUDA f32

    levi_edge.unpatch()  # deactivate

Or use directly without patching::

    C = levi_edge.mm(A, B)
"""

__version__ = "0.1.0"

from .backend import patch, unpatch, is_active, mm, bmm

__all__ = ["patch", "unpatch", "is_active", "mm", "bmm"]
