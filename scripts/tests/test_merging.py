"""Unit tests for tensor-level merge functions in src.merging.

Run directly: `python scripts/tests/test_merging.py`
Or with unittest:  `python -m unittest scripts.tests.test_merging`
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.merging import _per_layer_topk_mask, merge_ties


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


if __name__ == "__main__":
    unittest.main()
