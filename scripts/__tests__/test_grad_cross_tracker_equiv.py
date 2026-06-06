"""Equivalence test: memory-efficient sbar/stilde update vs. the
materializing einsum currently in finetune.GradCrossTermTracker.step.

Run:
    python scripts/tests/test_grad_cross_tracker_equiv.py
"""

import torch


def einsum_version(z, gy):
    """Current finetune.py implementation (OOMs for realistic shapes)."""
    B, T, Di = z.shape
    gynorm2 = gy.pow(2).sum(-1)  # (B, T)
    sbar = torch.einsum("bti,btj,bt->ij", z, z, gynorm2) / (B * T)
    stilde = (torch.einsum("bti,btj->ij", z, z) / (B * T)) * gynorm2.mean()
    return sbar, stilde


def matmul_version(z, gy):
    """Proposed memory-efficient rewrite."""
    B, T, Di = z.shape
    gynorm2 = gy.pow(2).sum(-1)  # (B, T)
    z_flat = z.reshape(-1, Di)  # (B*T, Di)
    gnorm_flat = gynorm2.reshape(-1)  # (B*T,)
    sbar = (z_flat * gnorm_flat.unsqueeze(-1)).T @ z_flat / (B * T)
    stilde = (z_flat.T @ z_flat) / (B * T) * gnorm_flat.mean()
    return sbar, stilde


def main():
    torch.manual_seed(0)
    # Use small shapes so the einsum version doesn't OOM.
    cases = [
        (4, 7, 16, 8),  # B, T, Di, Do
        (2, 197, 64, 64),  # ViT-ish T
        (8, 1, 32, 32),  # no token dim
    ]
    atol, rtol = 1e-5, 1e-5
    for B, T, Di, Do in cases:
        z = torch.randn(B, T, Di, dtype=torch.float64)
        gy = torch.randn(B, T, Do, dtype=torch.float64)
        s_e, st_e = einsum_version(z, gy)
        s_m, st_m = matmul_version(z, gy)
        max_s = (s_e - s_m).abs().max().item()
        max_st = (st_e - st_m).abs().max().item()
        torch.testing.assert_close(s_e, s_m, atol=atol, rtol=rtol)
        torch.testing.assert_close(st_e, st_m, atol=atol, rtol=rtol)
        print(
            f"OK | B={B} T={T} Di={Di} Do={Do} | "
            f"max|Δsbar|={max_s:.2e}  max|Δstilde|={max_st:.2e}"
        )
    print("\nAll equivalence checks passed.")


if __name__ == "__main__":
    main()
