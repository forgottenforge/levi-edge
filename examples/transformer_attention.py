#!/usr/bin/env python3
# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Transformer attention benchmark: where LEVI Edge shines.

Small-model transformers on edge devices have attention heads with
small matrix multiplications (seq_len x head_dim, typically 32-128).
This is exactly where LEVI Edge outperforms cuBLAS.
"""

import torch
import torch.nn as nn
import levi_edge
from levi_edge.benchmark import _time_fn


class TinyTransformer(nn.Module):
    """Minimal transformer for edge deployment."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)  # average pool over sequence
        return self.classifier(x)


def main():
    print("LEVI Edge — Transformer Attention Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        {"d_model": 32, "nhead": 2, "num_layers": 1, "seq_len": 16, "label": "Tiny"},
        {"d_model": 64, "nhead": 4, "num_layers": 2, "seq_len": 32, "label": "Small"},
        {"d_model": 128, "nhead": 4, "num_layers": 2, "seq_len": 64, "label": "Medium"},
        {"d_model": 256, "nhead": 8, "num_layers": 4, "seq_len": 128, "label": "Large-Edge"},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        seq_len = cfg.pop("seq_len")
        d_model = cfg["d_model"]

        print(f"\n--- {label}: d={d_model}, seq={seq_len} ---")

        model = TinyTransformer(**cfg).cuda().eval()
        x = torch.randn(1, seq_len, d_model, device="cuda")

        # Baseline
        with torch.no_grad():
            t_base = _time_fn(lambda: model(x), warmup=20, repeats=100)

        # LEVI
        levi_edge.patch()
        with torch.no_grad():
            t_levi = _time_fn(lambda: model(x), warmup=20, repeats=100)
        levi_edge.unpatch()

        speedup = t_base / t_levi
        print(f"  cuBLAS: {t_base:.3f} ms")
        print(f"  LEVI:   {t_levi:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Head dimension = d_model / nhead
        head_dim = d_model // cfg["nhead"]
        print(f"  (attention head matmul: {seq_len}x{head_dim} — "
              f"{'LEVI territory' if head_dim <= 128 else 'cuBLAS territory'})")


if __name__ == "__main__":
    main()
