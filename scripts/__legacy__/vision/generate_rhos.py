import torch
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("..")
from src import mhap, mhas
from src.vision.task_vectors import NonLinearTaskVector
from src.nlg.task_vectors import ParamFolderTaskVector


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


configs = [
    {
        "model": "ViT-B-16",
        "datasets": ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"],
        "task_vector_cls": NonLinearTaskVector,
        "val_suffix": "Val",
        "tv_transform": lambda tv: tv.map(mhas.copy_from_pytorch_state_dict),
    },
    {
        "model": "Olmo-3-7b",
        "datasets": ["Math", "Code", "IF"],
        "task_vector_cls": ParamFolderTaskVector,
        "val_suffix": "",
        "tv_transform": None,
    },
]

for config in configs:
    rows = []
    model = config["model"]
    datasets = config["datasets"]
    task_vector_cls = config["task_vector_cls"]
    val_suffix = config["val_suffix"]
    tv_transform = config["tv_transform"]

    results_dir = f"artifacts/results/{model}"
    rootdir = "artifacts/checkpoints"
    os.makedirs(results_dir, exist_ok=True)

    for dataset in tqdm(datasets, desc="datasets"):
        tv_dir = osp.join(rootdir, model, f"{dataset}{val_suffix}")
        tv = task_vector_cls(tv_dir)
        if tv_transform is not None:
            tv = tv_transform(tv)

        cdict = _load_stats_dict(tv.covariance_path)

        layer_idx = 0
        for layer_key in tqdm(tv.lazy_keys(), desc=dataset, leave=False):
            cov_key = tv.param_key_to_cov_key(layer_key)
            if cov_key not in cdict:
                continue
            d_t = tv.get_vector_element(layer_key)
            if len(d_t.shape) != 2:
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
