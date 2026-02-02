#!/usr/bin/env python3
# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Full benchmark suite: LEVI Edge vs cuBLAS across all edge-relevant sizes."""

from levi_edge.benchmark import benchmark_mm, save_report


def main():
    print("LEVI Edge — Full Benchmark Suite")
    print()

    results = benchmark_mm(
        sizes=[8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512],
        repeats=100,
        verbose=True,
    )

    save_report(results, "levi_benchmark.json")

    # Summary
    wins = [r for r in results if r.speedup > 1.05 and r.passed]
    if wins:
        print(f"\nLEVI Edge is faster for {len(wins)} out of {len(results)} sizes.")
        print("These are typical matrix dimensions for edge AI inference.")
    print()


if __name__ == "__main__":
    main()
