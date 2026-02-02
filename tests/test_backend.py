# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
"""Tests for LEVI Edge backend."""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


class TestDirectAPI:
    """Test levi_edge.mm / levi_edge.bmm direct calls."""

    def test_mm_small(self):
        """Small matrix multiply should match cuBLAS."""
        import levi_edge

        A = torch.randn(64, 128, device="cuda")
        B = torch.randn(128, 64, device="cuda")

        C_levi = levi_edge.mm(A, B)
        C_ref = torch.mm(A, B)

        assert C_levi.shape == C_ref.shape
        assert torch.allclose(C_levi, C_ref, atol=1e-3)

    def test_mm_various_sizes(self):
        """Test across multiple edge-relevant sizes."""
        import levi_edge

        for size in [16, 32, 64, 96, 128, 192, 256]:
            A = torch.randn(size, size, device="cuda")
            B = torch.randn(size, size, device="cuda")

            C_levi = levi_edge.mm(A, B)
            C_ref = torch.mm(A, B)

            assert torch.allclose(C_levi, C_ref, atol=1e-3), \
                f"Failed at size {size}: max error {(C_levi - C_ref).abs().max()}"

    def test_mm_non_square(self):
        """Non-square matrices should work."""
        import levi_edge

        A = torch.randn(32, 64, device="cuda")
        B = torch.randn(64, 128, device="cuda")

        C_levi = levi_edge.mm(A, B)
        C_ref = torch.mm(A, B)

        assert C_levi.shape == (32, 128)
        assert torch.allclose(C_levi, C_ref, atol=1e-3)

    def test_bmm(self):
        """Batched matrix multiply should match cuBLAS."""
        import levi_edge

        A = torch.randn(4, 32, 64, device="cuda")
        B = torch.randn(4, 64, 32, device="cuda")

        C_levi = levi_edge.bmm(A, B)
        C_ref = torch.bmm(A, B)

        assert C_levi.shape == C_ref.shape
        assert torch.allclose(C_levi, C_ref, atol=1e-3)

    def test_mm_large_falls_back(self):
        """Large matrices should fall back to cuBLAS (still correct)."""
        import levi_edge

        A = torch.randn(512, 512, device="cuda")
        B = torch.randn(512, 512, device="cuda")

        C = levi_edge.mm(A, B)
        C_ref = torch.mm(A, B)

        assert torch.allclose(C, C_ref, atol=1e-3)


class TestPatchUnpatch:
    """Test global patch/unpatch mechanism."""

    def test_patch_activates(self):
        """patch() should activate LEVI Edge."""
        import levi_edge

        levi_edge.patch()
        assert levi_edge.is_active()
        levi_edge.unpatch()
        assert not levi_edge.is_active()

    def test_patched_mm_is_correct(self):
        """After patch(), torch.mm should still produce correct results."""
        import levi_edge

        A = torch.randn(64, 64, device="cuda")
        B = torch.randn(64, 64, device="cuda")

        # Get reference BEFORE patching
        C_ref = torch.mm(A, B)

        levi_edge.patch()
        C_patched = torch.mm(A, B)
        levi_edge.unpatch()

        assert torch.allclose(C_patched, C_ref, atol=1e-3)

    def test_patched_large_still_works(self):
        """Large matrices should still work after patching."""
        import levi_edge

        A = torch.randn(1024, 1024, device="cuda")
        B = torch.randn(1024, 1024, device="cuda")

        C_ref = torch.mm(A, B)

        levi_edge.patch()
        C_patched = torch.mm(A, B)
        levi_edge.unpatch()

        assert torch.allclose(C_patched, C_ref, atol=1e-2)

    def test_double_patch_safe(self):
        """Calling patch() twice should not crash."""
        import levi_edge

        levi_edge.patch()
        levi_edge.patch()  # Should be no-op
        assert levi_edge.is_active()
        levi_edge.unpatch()

    def test_unpatch_without_patch_safe(self):
        """Calling unpatch() without patch() should not crash."""
        import levi_edge

        levi_edge.unpatch()  # Should be no-op


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_single_element(self):
        """1x1 matrix multiply."""
        import levi_edge

        A = torch.randn(1, 1, device="cuda")
        B = torch.randn(1, 1, device="cuda")

        C = levi_edge.mm(A, B)
        C_ref = torch.mm(A, B)
        assert torch.allclose(C, C_ref, atol=1e-5)

    def test_cpu_tensor_not_intercepted(self):
        """CPU tensors should not be affected by LEVI."""
        import levi_edge

        A = torch.randn(64, 64)
        B = torch.randn(64, 64)

        levi_edge.patch()
        C = torch.mm(A, B)  # Should use CPU, not crash
        levi_edge.unpatch()

        C_ref = torch.mm(A, B)
        assert torch.allclose(C, C_ref)

    def test_float64_not_intercepted(self):
        """Non-float32 should fall back to cuBLAS."""
        import levi_edge

        A = torch.randn(64, 64, device="cuda", dtype=torch.float64)
        B = torch.randn(64, 64, device="cuda", dtype=torch.float64)

        levi_edge.patch()
        C = torch.mm(A, B)
        levi_edge.unpatch()

        C_ref = torch.mm(A, B)
        assert torch.allclose(C, C_ref)

    def test_gradients_work(self):
        """Autograd should work through patched mm."""
        import levi_edge

        A = torch.randn(32, 32, device="cuda", requires_grad=True)
        B = torch.randn(32, 32, device="cuda", requires_grad=True)

        levi_edge.patch()
        C = torch.mm(A, B)
        loss = C.sum()
        loss.backward()
        levi_edge.unpatch()

        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == A.shape


class TestBenchmark:
    """Test benchmark module."""

    def test_benchmark_runs(self):
        """Benchmark should complete without errors."""
        from levi_edge.benchmark import benchmark_mm

        results = benchmark_mm(sizes=[32, 64], repeats=5, verbose=False)
        assert len(results) == 2
        assert all(r.passed for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
