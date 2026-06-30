import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.hf.experts import HFExpert


Q = "model.layers.0.self_attn.q_proj.weight"
K = "model.layers.0.self_attn.k_proj.weight"
V = "model.layers.0.self_attn.v_proj.weight"
QKV = "model.layers.0.self_attn.qkv_proj.weight"
O = "model.layers.0.self_attn.o_proj.weight"
MLP = "model.layers.0.mlp.down_proj.weight"


def _write_indexed_model(model_dir, tensors_by_shard):
    weight_map = {}
    total_size = 0
    for shard_name, tensors in tensors_by_shard.items():
        save_file(tensors, model_dir / shard_name, metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
        )

    return weight_map


def _base_tensors():
    return {
        Q: torch.arange(8.0).reshape(4, 2),
        K: torch.arange(8.0).reshape(4, 2) + 100,
        V: torch.arange(8.0).reshape(4, 2) + 200,
        O: torch.ones(2, 2),
        MLP: torch.full((2, 2), 2.0),
    }


class TestHF2Expert(unittest.TestCase):
    def test_split_preserves_native_qkv_keys(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model-00001-of-00002.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                    },
                    "model-00002-of-00002.safetensors": {
                        O: tensors[O],
                        MLP: tensors[MLP],
                    },
                },
            )

            expert = HFExpert(model_dir)

            self.assertEqual(list(expert.get_layers()), [Q, K, V, O, MLP])
            self.assertNotIn(QKV, expert.get_layers())
            torch.testing.assert_close(expert.get_layer_params(Q), tensors[Q])

    def test_packed_exposes_qkv_and_reads_concatenated_tensor(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model-00001-of-00002.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                    },
                    "model-00002-of-00002.safetensors": {
                        O: tensors[O],
                        MLP: tensors[MLP],
                    },
                },
            )

            expert = HFExpert(model_dir, mha="packed")

            self.assertEqual(list(expert.get_layers()), [QKV, O, MLP])
            self.assertNotIn(Q, expert.get_layers())
            self.assertNotIn(K, expert.get_layers())
            self.assertNotIn(V, expert.get_layers())
            self.assertEqual(
                expert.get_layer_metadata(QKV),
                [
                    {
                        "tensor_name": Q,
                        "shard_filename": "model-00001-of-00002.safetensors",
                    },
                    {
                        "tensor_name": K,
                        "shard_filename": "model-00001-of-00002.safetensors",
                    },
                    {
                        "tensor_name": V,
                        "shard_filename": "model-00001-of-00002.safetensors",
                    },
                ],
            )
            torch.testing.assert_close(
                expert.get_layer_params(QKV),
                torch.cat([tensors[Q], tensors[K], tensors[V]], dim=0),
            )

    def test_packed_verbose_prints_summary(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                        O: tensors[O],
                    },
                },
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                HFExpert(model_dir, mha="packed", verbose=True)

            self.assertIn("Packed QKV safetensor index", stdout.getvalue())
            self.assertIn("packed 1 QKV groups", stdout.getvalue())

    def test_packed_init_raises_when_qkv_spans_shards(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model-00001-of-00002.safetensors": {Q: tensors[Q], K: tensors[K]},
                    "model-00002-of-00002.safetensors": {V: tensors[V]},
                },
            )

            with self.assertRaisesRegex(
                ValueError, "span multiple safetensor shards"
            ):
                HFExpert(model_dir, mha="packed")

    def test_packed_init_raises_when_qkv_rows_differ(self):
        tensors = _base_tensors()
        tensors[K] = torch.arange(4.0).reshape(2, 2) + 100
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "row counts differ"):
                HFExpert(model_dir, mha="packed")

    def test_packed_save_writes_native_qkv_tensors(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_dir = tmp_path / "source"
            dest_dir = tmp_path / "dest"
            source_dir.mkdir()
            dest_dir.mkdir()
            _write_indexed_model(
                source_dir,
                {
                    "model.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                    },
                },
            )

            source = HFExpert(source_dir, mha="packed")
            dest = HFExpert(dest_dir, mha="packed")
            dest.save_layer_params(
                source.get_layer_params(QKV) + 1,
                QKV,
                metadata=source.get_layer_metadata(QKV),
            )
            dest.flush()

            with safe_open(
                dest_dir / "model.safetensors", framework="pt", device="cpu"
            ) as f:
                torch.testing.assert_close(f.get_tensor(Q), tensors[Q] + 1)
                torch.testing.assert_close(f.get_tensor(K), tensors[K] + 1)
                torch.testing.assert_close(f.get_tensor(V), tensors[V] + 1)
                self.assertNotIn(QKV, f.keys())

            with open(dest_dir / "model.safetensors.index.json") as f:
                weight_map = json.load(f)["weight_map"]
            self.assertEqual(
                weight_map,
                {
                    Q: "model.safetensors",
                    K: "model.safetensors",
                    V: "model.safetensors",
                },
            )

    def test_packed_leaves_incomplete_qkv_triples_unchanged(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        O: tensors[O],
                    },
                },
            )

            expert = HFExpert(model_dir, mha="packed")

            self.assertEqual(list(expert.get_layers()), [Q, K, O])
            self.assertNotIn(QKV, expert.get_layers())

    def test_packed_stats_use_synthetic_qkv_key(self):
        tensors = _base_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_indexed_model(
                model_dir,
                {
                    "model.safetensors": {
                        Q: tensors[Q],
                        K: tensors[K],
                        V: tensors[V],
                    },
                },
            )
            native_cov_path = model_dir / "native_cov.pt"
            packed_cov_path = model_dir / "packed_cov.pt"
            packed_fisher_path = model_dir / "packed_fisher.pt"
            torch.save({Q.replace(".weight", ""): torch.eye(2)}, native_cov_path)
            torch.save({QKV.replace(".weight", ""): torch.ones(2, 2)}, packed_cov_path)
            torch.save(
                {QKV.replace(".weight", ""): torch.full((2, 2), 3)},
                packed_fisher_path,
            )

            native_stats = HFExpert(
                model_dir, covariance_path=native_cov_path, mha="packed"
            )
            packed_stats = HFExpert(
                model_dir,
                covariance_path=packed_cov_path,
                fisher_path=packed_fisher_path,
                mha="packed",
            )

            self.assertIsNone(native_stats.get_layer_cov(QKV))
            torch.testing.assert_close(
                packed_stats.get_layer_cov(QKV), torch.ones(2, 2)
            )
            torch.testing.assert_close(
                packed_stats.get_layer_fish(QKV), torch.full((2, 2), 3)
            )

    def test_invalid_mha_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Unsupported HFExpert mha mode"):
                HFExpert(tmpdir, mha="none")


if __name__ == "__main__":
    unittest.main()
