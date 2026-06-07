"""Unit tests for tensor-level merge functions in src.mergingv2.

Run directly: `python scripts/__tests__/test_mergingv2.py`
Or with unittest:  `python -m unittest scripts.__tests__.test_mergingv2`
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.mergingv2 import _interp_cov, merge_regmean, merge_regmean_interp


def _angle(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angle (radians) between two matrices under the Frobenius inner product."""
    cos = (a * b).sum() / (a.norm() * b.norm())
    return torch.arccos(cos.clamp(-1.0, 1.0)).item()


def _random_psd(dim: int, n_samples: int = 64) -> torch.Tensor:
    x = torch.randn(n_samples, dim)
    return x.T @ x / n_samples


class TestInterpCov(unittest.TestCase):
    def test_achieved_angle_matches_requested(self):
        """The returned matrix must sit at exactly the requested angular
        distance from the true covariance (for angles below angle(c, I))."""
        torch.manual_seed(0)
        # float64: arccos is ill-conditioned at small angles in float32
        c = _random_psd(16).double()
        theta_max = _angle(c, torch.eye(16, dtype=torch.float64))
        for theta in [0.01, 0.1, 0.3, 0.9 * theta_max]:
            ci = _interp_cov(c, theta)
            self.assertAlmostEqual(_angle(c, ci), theta, places=6)

    def test_zero_angle_returns_c(self):
        torch.manual_seed(0)
        c = _random_psd(16)
        self.assertTrue(torch.allclose(_interp_cov(c, 0.0), c, atol=1e-6))

    def test_saturates_at_identity(self):
        """Angles at or beyond angle(c, I) — including past pi/2, where tan
        flips sign — must return the identity endpoint."""
        torch.manual_seed(0)
        c = _random_psd(16)
        eye = torch.eye(16)
        theta_max = _angle(c, eye)
        for theta in [theta_max, theta_max + 0.5, 3.0]:
            self.assertTrue(torch.allclose(_interp_cov(c, theta), eye, atol=1e-5))

    def test_result_on_segment(self):
        """The result must lie on the segment c -> I: ci = (1-s)c + sI with
        s in [0, 1]."""
        torch.manual_seed(0)
        c = _random_psd(16)
        eye = torch.eye(16)
        ci = _interp_cov(c, 0.2)
        # Recover s from the displacement along the chord and check collinearity.
        chord = eye - c
        s = ((ci - c) * chord).sum() / chord.norm() ** 2
        self.assertGreaterEqual(s.item(), 0.0)
        self.assertLessEqual(s.item(), 1.0)
        self.assertTrue(torch.allclose(ci, (1 - s) * c + s * eye, atol=1e-5))

    def test_angle_monotone_in_theta(self):
        torch.manual_seed(0)
        c = _random_psd(16)
        thetas = [0.05, 0.1, 0.2, 0.3, 0.4]
        achieved = [_angle(c, _interp_cov(c, t)) for t in thetas]
        self.assertEqual(achieved, sorted(achieved))

    def test_isotropic_c_unchanged(self):
        """c proportional to I: every point on the segment has angle 0, so the
        degenerate case must return c itself (no 0/0)."""
        c = 2.0 * torch.eye(16)
        self.assertTrue(torch.equal(_interp_cov(c, 0.5), c))


class TestMergeRegmeanInterp(unittest.TestCase):
    def _make_inputs(self, n_tasks=3, dim=16):
        torch.manual_seed(0)
        covs = [_random_psd(dim) for _ in range(n_tasks)]
        d = torch.randn(n_tasks, dim, dim)
        maps = [{"covariance": (lambda ct=ct: ct)} for ct in covs]
        return d, maps

    def test_shape(self):
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.2)
        self.assertEqual(out.shape, (16, 16))

    def test_zero_angle_equals_regmean(self):
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.0)
        expected = merge_regmean(d, stat_fetcher_maps=maps)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_saturated_angle_equals_mean(self):
        """All covariances at the identity endpoint: RegMean with c_t = I
        reduces to the plain mean."""
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=3.0)
        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-5))

    def test_intermediate_angle_strictly_between(self):
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.2)
        regmean = merge_regmean(d, stat_fetcher_maps=maps)
        self.assertFalse(torch.allclose(out, regmean, atol=1e-5))
        self.assertFalse(torch.allclose(out, d.mean(dim=0), atol=1e-5))

    def test_missing_covariance_falls_back_to_mean(self):
        d, maps = self._make_inputs()
        maps[1] = {"covariance": (lambda: None)}
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.2)
        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
