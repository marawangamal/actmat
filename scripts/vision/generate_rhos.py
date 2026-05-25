import torch
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("..")
from src import mhap, mhas
from src.vision.task_vectors import NonLinearTaskVector


def _load_stats_dict(path: str) -> dict:
    """Load a statistics dict from a .pt file or a directory of per-layer .pt files."""
    if os.path.isdir(path):
        result = {}
        for fname in os.listdir(path):
            if fname.endswith(".pt"):
                key = fname[:-3]  # strip .pt
                result[key] = torch.load(
                    os.path.join(path, fname), map_location="cpu", weights_only=False
                )
        return result
    else:
        return torch.load(path, map_location="cpu", weights_only=False)


# model = "ViT-B-16"
# results_dir = f"artifacts/results/{model}"
# rootdir = "../artifacts/checkpoints"
datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"]
configs = [
    {
        "model": "ViT-B-16",
        "datasets": [d + "Val" for d in ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"]],
    }
    {
        "model": "ViT-B-16",
        "datasets": ["Math", "Code", "IF"],
    }
]

for config in configs:
    rows = []
    model = config["model"]
    results_dir = f"artifacts/results/{model}"
    rootdir = "artifacts/checkpoints"

    for dataset in tqdm(datasets, desc="datasets"):
        tv_dir = osp.join(rootdir, model, f"{dataset}Val")
        tv = NonLinearTaskVector(tv_dir)
        tv = tv.map(mhas.copy_from_pytorch_state_dict)

        layer_idx = 0
        for layer_key, d_t in tqdm(tv.vector.items(), desc=dataset, leave=False):
            cov_key = tv.param_key_to_cov_key(layer_key)
            cdict = _load_stats_dict(tv.covariance_path)
            if len(d_t.shape) != 2 or cov_key not in cdict.keys():
                continue
            c_t = cdict[cov_key]
            c_t_hat = d_t.T @ d_t
            kappa = torch.linalg.norm(c_t, ord="fro") / torch.linalg.norm(
                c_t_hat, ord="fro"
            )
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "layer_name": layer_key,
                    "layer_idx": layer_idx,
                    "kappa": kappa.item(),
                }
            )
            layer_idx += 1

    # Create a dataframe
    df = pd.DataFrame(rows)

    # Create a "rho" dataframe by merging the results with itself
    # We join on 'model', 'layer_name', and 'layer_idx' to pair datasets for the same layer
    df = df.merge(df, on=["model", "layer_name", "layer_idx"], suffixes=("_i", "_j"))

    # Compute rho = kappa_i / kappa_j
    df["rho"] = df["kappa_i"] / df["kappa_j"]

    # Save
    df.to_csv(os.path.join(results_dir, "rhos.csv"), index=False)
    print(f"Saved results to {os.path.join(results_dir, 'rhos.csv')}")
