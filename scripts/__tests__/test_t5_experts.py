import torch

from src.t5.experts import T5Expert


def test_t5_expert_loads_state_dict_and_sidecars(tmp_path):
    weights_path = tmp_path / "finetuned.pt"
    cov_path = tmp_path / "covariance.pt"
    fisher_path = tmp_path / "fisher.pt"

    state_dict = {
        "encoder.block.0.layer.0.DenseReluDense.wi.weight": torch.eye(2),
        "tokenizer_length": torch.tensor(4, dtype=torch.int64),
    }
    torch.save(state_dict, weights_path)
    torch.save(
        {"encoder.block.0.layer.0.DenseReluDense.wi": torch.ones(2, 2)}, cov_path
    )
    torch.save(
        {"encoder.block.0.layer.0.DenseReluDense.wi": torch.full((2, 2), 2)},
        fisher_path,
    )

    expert = T5Expert(weights_path, cov_path, fisher_path)

    assert list(expert.get_layers()) == [
        "encoder.block.0.layer.0.DenseReluDense.wi.weight"
    ]
    assert torch.equal(
        expert.get_layer_params("encoder.block.0.layer.0.DenseReluDense.wi.weight"),
        torch.eye(2),
    )
    assert torch.equal(
        expert.get_layer_cov("encoder.block.0.layer.0.DenseReluDense.wi.weight"),
        torch.ones(2, 2),
    )
    assert torch.equal(
        expert.get_layer_fish("encoder.block.0.layer.0.DenseReluDense.wi.weight"),
        torch.full((2, 2), 2),
    )


def test_t5_expert_empty_destination_saves_params():
    expert = T5Expert()
    expert.save_layer_params(torch.ones(2, 2), "layer.weight")

    assert torch.equal(expert.model_state_dict["layer.weight"], torch.ones(2, 2))
