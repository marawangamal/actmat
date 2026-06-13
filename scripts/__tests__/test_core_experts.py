import os.path as osp
import tempfile
import unittest

import torch

from src.core.experts import Expert
from src.core.merge import merge_experts
from src.vit.experts import ViTExpert


class DictExpert(Expert):
    def __init__(self, state_dict=None):
        self.state_dict = state_dict or {}

    def get_layers(self):
        return self.state_dict.keys()

    def get_layer_params(self, layer_name):
        return self.state_dict[layer_name]

    def save_layer_params(self, tensor, layer_name, metadata=None):
        self.state_dict[layer_name] = tensor


class TestCoreMerge(unittest.TestCase):
    def test_mean_merge_updates_destination(self):
        base = DictExpert(
            {
                "linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "bias": torch.tensor([1.0, 2.0]),
            }
        )
        expert_a = DictExpert(
            {
                "linear.weight": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
                "bias": torch.tensor([3.0, 5.0]),
            }
        )
        expert_b = DictExpert(
            {
                "linear.weight": torch.tensor([[3.0, 6.0], [9.0, 12.0]]),
                "bias": torch.tensor([5.0, 7.0]),
            }
        )
        merged = DictExpert()

        merge_experts(
            base,
            [expert_a, expert_b],
            merged,
            "mean",
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            merged.state_dict["linear.weight"],
            torch.tensor([[2.5, 5.0], [7.5, 10.0]]),
        )
        torch.testing.assert_close(merged.state_dict["bias"], torch.tensor([4.0, 6.0]))


class TestViTExpert(unittest.TestCase):
    def test_mha_mapping_round_trip_and_memory_save(self):
        state_dict = {
            "block.attn.in_proj_weight": torch.arange(18.0).reshape(6, 3),
            "block.attn.out_proj.weight": torch.arange(9.0).reshape(3, 3),
            "mlp.weight": torch.ones(2, 2),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = osp.join(tmpdir, "model.pt")
            torch.save(state_dict, weights_path)

            expert = ViTExpert(weights_path=weights_path)
            self.assertIn("block.attn.q.weight", expert.state_dict)
            self.assertIn("block.attn.k.weight", expert.state_dict)
            self.assertIn("block.attn.v.weight", expert.state_dict)
            self.assertIn("block.attn.o.weight", expert.state_dict)
            self.assertNotIn("block.attn.in_proj_weight", expert.state_dict)

            merged = ViTExpert()
            merged.save_layer_params(
                expert.get_layer_params("mlp.weight") + 1, "mlp.weight"
            )
            torch.testing.assert_close(
                merged.state_dict["mlp.weight"], torch.full((2, 2), 2.0)
            )

            native = expert.model_state_dict
            torch.testing.assert_close(
                native["block.attn.in_proj_weight"],
                state_dict["block.attn.in_proj_weight"],
            )
            torch.testing.assert_close(
                native["block.attn.out_proj.weight"],
                state_dict["block.attn.out_proj.weight"],
            )

    def test_mha_packed_preserves_native_state_dict(self):
        state_dict = {
            "block.attn.in_proj_weight": torch.arange(18.0).reshape(6, 3),
            "block.attn.out_proj.weight": torch.arange(9.0).reshape(3, 3),
            "mlp.weight": torch.ones(2, 2),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = osp.join(tmpdir, "model.pt")
            torch.save(state_dict, weights_path)

            expert = ViTExpert(weights_path=weights_path, mha="packed")
            self.assertIn("block.attn.in_proj_weight", expert.state_dict)
            self.assertNotIn("block.attn.q.weight", expert.state_dict)

            native = expert.model_state_dict
            self.assertIs(native, expert.state_dict)
            torch.testing.assert_close(
                native["block.attn.in_proj_weight"],
                state_dict["block.attn.in_proj_weight"],
            )

    def test_invalid_mha_mode_raises(self):
        with self.assertRaises(ValueError):
            ViTExpert(mha="none")

    def test_cov_key_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = osp.join(tmpdir, "model.pt")
            cov_path = osp.join(tmpdir, "covariance.pt")
            torch.save({"layer.weight": torch.ones(2, 2)}, weights_path)
            torch.save({"image_encoder.layer": torch.eye(2)}, cov_path)

            expert = ViTExpert(weights_path=weights_path, covariance_path=cov_path)
            torch.testing.assert_close(
                expert.get_layer_cov("layer.weight"), torch.eye(2)
            )


if __name__ == "__main__":
    unittest.main()
