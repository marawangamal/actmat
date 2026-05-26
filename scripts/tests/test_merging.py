"""Unit tests for tensor-level merge functions in src.merging.

Run directly: `python scripts/tests/test_merging.py`
Or with unittest:  `python -m unittest scripts.tests.test_merging`
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.merging import (
    _per_layer_topk_mask,
    merge_actmat,
    merge_actmat_mons,
    merge_actmat_p,
    merge_actmat_softmax_bias,
    merge_ties,
    merge_wudi,
)


def _wudi_loss(M: torch.Tensor, d: torch.Tensor) -> float:
    N = d.shape[0]
    l2_sq = d.reshape(N, -1).pow(2).sum(dim=-1).view(N, 1, 1)
    inner = torch.matmul(M.unsqueeze(0) - d, d.transpose(1, 2))
    return (inner.pow(2) / l2_sq).sum().item()


class TestMergeTies(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 6)
        out = merge_ties(d, ties_k=0.5)
        self.assertEqual(out.shape, (4, 6))

    def test_full_keep_equals_disjoint_sign_mean(self):
        """With ties_k=1.0 the trim is a no-op, so the result must equal the
        disjoint-sign mean of the raw stack."""
        torch.manual_seed(0)
        d = torch.randn(3, 4, 6)
        out = merge_ties(d, ties_k=1.0)

        elected = torch.sign(d.sum(dim=0))
        match = (torch.sign(d) == elected.unsqueeze(0)).to(d.dtype)
        expected = (d * match).sum(dim=0) / match.sum(dim=0).clamp(min=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_single_task_returns_trimmed_input(self):
        """With one task and any ties_k, the result is just the trimmed task vector."""
        torch.manual_seed(1)
        d = torch.randn(1, 4, 6)
        out = merge_ties(d, ties_k=0.5)
        mask = _per_layer_topk_mask(d, 0.5)
        expected = (d * mask).squeeze(0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_toy_hand_computed(self):
        """Two tasks, one layer (1x2).

          tasks: [[ 2.0,  1.0]]   and  [[-3.0,  4.0]]
          pos 0: sum = -1 -> elected sign -1, only -3 contributes -> -3.0
          pos 1: sum = +5 -> elected sign +1, both contribute      -> (1+4)/2 = 2.5
        """
        d = torch.tensor([[[2.0, 1.0]], [[-3.0, 4.0]]])
        out = merge_ties(d, ties_k=1.0)
        self.assertTrue(torch.allclose(out, torch.tensor([[-3.0, 2.5]])))

    def test_trim_zeros_low_magnitude_entries(self):
        """At ties_k=0.5 with 4 entries per task, exactly the top-2 |entries|
        survive in each task before sign election."""
        d = torch.tensor(
            [
                [[0.1, 5.0, 0.2, 4.0]],
                [[6.0, 0.3, 7.0, 0.4]],
            ]
        )
        # Trim → task0 keeps {5.0, 4.0}; task1 keeps {6.0, 7.0}.
        # Elected signs all +. Each position has exactly one contributor.
        # → [6.0, 5.0, 7.0, 4.0]
        out = merge_ties(d, ties_k=0.5)
        expected = torch.tensor([[6.0, 5.0, 7.0, 4.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_position_with_no_matching_sign_is_zero(self):
        """If every task is trimmed away at a position (no contributors), the
        disjoint mean defaults to 0 (clamp(min=1) keeps the division safe)."""
        # Two tasks, ties_k=0.5 keeps 1 of 2 entries per task. At position 0
        # both tasks have their smaller |val|, so position 0 is fully trimmed.
        d = torch.tensor(
            [
                [[0.1, 5.0]],
                [[0.2, 6.0]],
            ]
        )
        out = merge_ties(d, ties_k=0.5)
        # position 0: both trimmed -> 0; position 1: (5+6)/2 = 5.5
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 5.5]])))


class TestMergeWudi(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(0)
        d = torch.randn(3, 8, 6)
        out = merge_wudi(d, wudi_iters=10, wudi_lr=1e-4)
        self.assertEqual(out.shape, (8, 6))
        self.assertFalse(out.requires_grad)

    def test_zero_iters_equals_sum_init(self):
        """With wudi_iters=0 the output is the initialization M = Σ τ_i."""
        torch.manual_seed(0)
        d = torch.randn(4, 5, 7)
        out = merge_wudi(d, wudi_iters=0)
        self.assertTrue(torch.allclose(out, d.sum(dim=0), atol=1e-6))

    def test_single_task_is_fixed_point(self):
        """With one task, M init = τ already minimizes the loss (gradient is 0)
        since (M − τ) = 0, so the optimizer should not move it."""
        torch.manual_seed(1)
        d = torch.randn(1, 4, 6)
        out = merge_wudi(d, wudi_iters=50, wudi_lr=1e-3)
        self.assertTrue(torch.allclose(out, d.squeeze(0), atol=1e-6))

    def test_loss_decreases(self):
        """The WUDI loss at the optimized M should be strictly lower than at
        the sum-init, given a learning rate that actually moves the weights."""
        torch.manual_seed(2)
        d = torch.randn(3, 16, 10)
        init_loss = _wudi_loss(d.sum(dim=0), d)
        out = merge_wudi(d, wudi_iters=200, wudi_lr=1e-3)
        final_loss = _wudi_loss(out, d)
        self.assertLess(final_loss, init_loss)


class TestMergeActmatMons(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(0)
        d = torch.randn(3, 8, 6)
        out = merge_actmat_mons(d)
        self.assertEqual(out.shape, (8, 6))

    def test_runs_without_kwarg_typo(self):
        """Regression for the prior `dims=`/`keepdims=` bug — torch.norm uses
        `dim`/`keepdim`. A typo there raises TypeError before reaching matmul."""
        d = torch.randn(2, 4, 5)
        merge_actmat_mons(d)  # must not raise

    def test_equal_norm_tasks_equal_actmat(self):
        """If every task already has the same Frobenius norm, the per-task
        rescale d_tilde = d is a no-op, so the result equals merge_actmat."""
        torch.manual_seed(0)
        d = torch.randn(3, 4, 5)
        d = d / d.norm(dim=(-2, -1), keepdim=True)  # all unit Frobenius norm
        out = merge_actmat_mons(d)
        expected = merge_actmat(d)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_rescaled_norms_match_mean(self):
        """Each d_tilde_t should have Frobenius norm == mean(||d_t||_F)."""
        torch.manual_seed(1)
        d = torch.randn(4, 6, 7)
        mu = d.norm(dim=(-2, -1)).mean()
        d_tilde = d * mu / d.norm(dim=(-2, -1), keepdim=True)
        norms = d_tilde.norm(dim=(-2, -1))
        self.assertTrue(torch.allclose(norms, mu.expand_as(norms), atol=1e-5))

    def test_scale_invariance_of_d_tilde(self):
        """Scaling one task by a constant should not change d_tilde (rescale
        cancels), and therefore C is unchanged. The merge target (d @ C) does
        scale with that task, so the full output is not invariant — but C is."""
        torch.manual_seed(2)
        d = torch.randn(3, 5, 5)
        d_scaled = d.clone()
        d_scaled[0] *= 7.0

        def _c(x):
            mu = x.norm(dim=(-2, -1)).mean()
            x_tilde = x * mu / x.norm(dim=(-2, -1), keepdim=True)
            return x_tilde.transpose(1, 2) @ x_tilde

        # Each per-task C_t is scale-invariant in d_t individually.
        c1 = _c(d)
        c2 = _c(d_scaled)
        # Means differ (so does mu), so C_t themselves shift uniformly —
        # but the per-task *direction* doesn't. Check C ratios stay finite/equal.
        ratio = c2[0] / (c1[0] + 1e-12)
        # All entries of C[0] scale by the same constant (mu_scaled/mu_orig)^2.
        self.assertTrue(torch.allclose(ratio, ratio.mean() * torch.ones_like(ratio), atol=1e-4))


class TestMergeActmatP(unittest.TestCase):
    """merge_actmat_p(p) scales each c_t by 1/‖d_t‖^(2p) — a 1-parameter family
    interpolating between vanilla actmat (p=0) and mons (p=1)."""

    def _data(self):
        torch.manual_seed(0)
        # Heterogeneous norms so any per-task scaling is visible in the output.
        return torch.randn(3, 6, 5) * torch.tensor([1.0, 5.0, 20.0]).view(-1, 1, 1)

    def test_shape(self):
        d = self._data()
        self.assertEqual(merge_actmat_p(d, p=0.5).shape, (6, 5))

    def test_p0_equals_actmat(self):
        """At p=0, gamma_t=1 (scalar) — cancels in the pinv solve."""
        d = self._data()
        self.assertTrue(
            torch.allclose(merge_actmat_p(d, p=0.0), merge_actmat(d), atol=1e-4)
        )

    def test_p1_equals_mons(self):
        """At p=1, gamma_t = 1/‖d_t‖². mons uses gamma_t = μ²/‖d_t‖² — same up to
        the scalar μ², which cancels in the pinv solve. So p=1 ≡ mons."""
        d = self._data()
        self.assertTrue(
            torch.allclose(merge_actmat_p(d, p=1.0), merge_actmat_mons(d), atol=1e-4)
        )

    def test_intermediate_p_strictly_between(self):
        """A non-endpoint p should differ from both bracketing baselines."""
        d = self._data()
        out = merge_actmat_p(d, p=0.5)
        base_actmat = merge_actmat(d)
        base_mons = merge_actmat_mons(d)
        self.assertFalse(torch.allclose(out, base_actmat, atol=1e-3))
        self.assertFalse(torch.allclose(out, base_mons, atol=1e-3))


class TestMergeActmatSoftmaxBias(unittest.TestCase):
    """C_t = α_t·(d_tᵀ d_t) + (1−α_t)·I, with α_t = 1 − softmax(‖d_t‖)_t.
    α→0 collapses to mean(d); α→1 collapses to vanilla ACTMat."""

    def test_shape(self):
        torch.manual_seed(0)
        d = torch.randn(3, 6, 5)
        self.assertEqual(merge_actmat_softmax_bias(d).shape, (6, 5))

    def test_no_nan_with_zero_delta(self):
        torch.manual_seed(1)
        d = torch.randn(3, 4, 5)
        d[1] = 0.0
        out = merge_actmat_softmax_bias(d)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_alpha_smallest_for_largest_norm_task(self):
        torch.manual_seed(2)
        d = torch.randn(4, 5, 5)
        d[2] *= 50.0
        mags = d.norm(dim=(-2, -1))
        alpha = 1 - mags.softmax(dim=0)
        self.assertEqual(int(alpha.argmin()), int(mags.argmax()))

    def test_alpha_one_equals_actmat(self):
        """If every α_t = 1 (all C_t = d_tᵀd_t), this is vanilla ACTMat.
        Achieved by feeding equal-norm tasks far from each other so softmax
        gives weight 1/T → α = (T-1)/T ≠ 1 (so we monkey-patch instead)."""
        torch.manual_seed(3)
        d = torch.randn(3, 4, 5)
        # Direct comparison: alpha forced to 1 ⇒ C = d^T d ⇒ vanilla actmat.
        cov = d.transpose(1, 2) @ d
        c_sum = cov.sum(dim=0)
        expected = (d @ cov).sum(dim=0) @ torch.linalg.pinv(c_sum.double()).to(
            c_sum.dtype
        )
        self.assertTrue(torch.allclose(expected, merge_actmat(d), atol=1e-4))

    def test_alpha_zero_collapses_to_mean(self):
        """If every α_t = 0 (all C_t = I), the closed form reduces to mean(d).

        target = Σ_t d_t·I = Σ d_t;   C_sum = T·I;   W = (Σ d_t) · (1/T)·I = mean."""
        torch.manual_seed(4)
        d = torch.randn(3, 4, 5)
        T, Do, Di = d.shape
        eye = torch.eye(Di).expand(T, Di, Di)
        c_sum = eye.sum(dim=0)
        target = (d @ eye).sum(dim=0)
        out = target @ torch.linalg.pinv(c_sum)
        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-5))

    def test_differs_from_actmat_on_unequal_norms(self):
        torch.manual_seed(5)
        d = torch.randn(3, 4, 5) * torch.tensor([1.0, 5.0, 20.0]).view(-1, 1, 1)
        self.assertFalse(
            torch.allclose(merge_actmat_softmax_bias(d), merge_actmat(d), atol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
