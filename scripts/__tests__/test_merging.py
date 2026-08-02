"""Unit tests for tensor-level merge functions in src.merging.

Run directly: `python scripts/__tests__/test_merging.py`
Or with unittest:  `python -m unittest scripts.__tests__.test_merging`
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.merging import (
    _interp_cov,
    merge_actmat_w,
    merge_actmat_w_fp64,
    merge_regmean,
    merge_regmean_interp,
    merge_regmean_w,
    merge_regmean_w_fp64,
)


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
    """merge_regmean_interp takes angular_distance in units of pi
    (matching generate_error_terms.py), unlike _interp_cov (radians)."""

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
        reduces to the plain mean. angular_distance=0.5 is orthogonal,
        beyond any PSD covariance's angle to I."""
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.5)
        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-5))

    def test_intermediate_angle_strictly_between(self):
        # 0.05*pi ~ 0.16 rad, safely below the ~0.5 rad theta_max of the
        # random 16-dim test covariances (no saturation)
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.05)
        regmean = merge_regmean(d, stat_fetcher_maps=maps)
        self.assertFalse(torch.allclose(out, regmean, atol=1e-5))
        self.assertFalse(torch.allclose(out, d.mean(dim=0), atol=1e-5))

    def test_angular_distance_in_units_of_pi(self):
        """angular_distance=x must interpolate covariances by x*pi radians."""
        d, maps = self._make_inputs()
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.1)
        covs = [m["covariance"]() for m in maps]
        ci = torch.stack([_interp_cov(c, 0.1 * torch.pi) for c in covs])
        expected = (d @ ci).sum(dim=0) @ torch.linalg.pinv(ci.sum(dim=0))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_missing_covariance_falls_back_to_mean(self):
        d, maps = self._make_inputs()
        maps[1] = {"covariance": (lambda: None)}
        out = merge_regmean_interp(d, stat_fetcher_maps=maps, angular_distance=0.2)
        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-6))


class TestMergeRegmeanW(unittest.TestCase):
    def test_merges_full_weights_then_returns_delta(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)
        covs = [_random_psd(4) for _ in range(3)]
        maps = [{"covariance": (lambda ct=ct: ct)} for ct in covs]

        out = merge_regmean_w(d, w0=w0, stat_fetcher_maps=maps)
        c = torch.stack(covs)
        expected = ((d + w0) @ c).sum(dim=0) @ torch.linalg.pinv(c.sum(dim=0)) - w0

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_missing_covariance_falls_back_to_delta_mean(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)
        maps = [{"covariance": lambda: None} for _ in range(3)]

        out = merge_regmean_w(d, w0=w0, stat_fetcher_maps=maps)

        self.assertTrue(torch.allclose(out, d.mean(dim=0), atol=1e-6))


class TestMergeActmatWFp64(unittest.TestCase):
    def test_computes_in_float64(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)

        out = merge_actmat_w_fp64(d, w0=w0)
        d_fp64 = d.double()
        w0_fp64 = w0.double()
        c = d_fp64.transpose(1, 2) @ d_fp64
        expected = (
            ((d_fp64 + w0_fp64) @ c).sum(dim=0)
            @ torch.linalg.pinv(c.sum(dim=0))
            - w0_fp64
        )

        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.allclose(out, expected))

    def test_matches_fp32_on_well_conditioned_input(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)

        out = merge_actmat_w_fp64(d, w0=w0)
        expected = merge_actmat_w(d, w0=w0)

        self.assertTrue(torch.allclose(out.float(), expected, atol=1e-4))


class TestMergeRegmeanWFp64(unittest.TestCase):
    def test_computes_weights_covariances_and_solve_in_float64(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)
        covs = [_random_psd(4) for _ in range(3)]
        maps = [{"covariance": (lambda ct=ct: ct)} for ct in covs]

        out = merge_regmean_w_fp64(d, w0=w0, stat_fetcher_maps=maps)
        d_fp64 = d.double()
        w0_fp64 = w0.double()
        c = torch.stack(covs).double()
        expected = (
            ((d_fp64 + w0_fp64) @ c).sum(dim=0)
            @ torch.linalg.pinv(c.sum(dim=0))
            - w0_fp64
        )

        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.allclose(out, expected))

    def test_missing_covariance_returns_fp64_delta_mean(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 4)
        w0 = torch.randn(4, 4)
        maps = [{"covariance": lambda: None} for _ in range(3)]

        out = merge_regmean_w_fp64(d, w0=w0, stat_fetcher_maps=maps)

        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.allclose(out, d.double().mean(dim=0)))


if __name__ == "__main__":
    unittest.main()
