import os
import sys
import argparse
from tqdm import tqdm
import os.path as osp

import numpy as np
import torch
import pandas as pd

from src.utils import expert_dir, group_dir

# sys.path.append("..")
# from src import mhap, mhas
# from src.vision.task_vectors import NonLinearTaskVector
# from src.nlg.task_vectors import ParamFolderTaskVector


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


def get_rand_psd(
    m: int, n: int, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    X = torch.randn(m, n, dtype=dtype, device=device)
    return X @ X.T


parser = argparse.ArgumentParser()
# parser.add_argument("--ckpt_dir", default="artifacts/checkpoints")
# parser.add_argument("--results_dir", default="artifacts/results")
parser.add_argument("--ckpt_dir", default="artifacts/checkpoints-analysisv2-epochs1")
parser.add_argument("--results_dir", default="artifacts/results-analysisv2-epochs1")

parser.add_argument("--model", default="ViT-B-16")
parser.add_argument("--group", default="main")
args = parser.parse_args()

model = args.model
ckpt_dir = osp.join(args.ckpt_dir, model, group_dir(args.group))
results_dir = osp.join(args.results_dir, model, group_dir(args.group))
os.makedirs(results_dir, exist_ok=True)

datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

ad_fn = lambda c: np.arccos(c.clip(-1.0, 1.0)) / np.pi

rows = []
for d in tqdm(datasets, desc="datasets"):
    task_dir = expert_dir(ckpt_dir, d)
    if not os.path.isdir(task_dir):
        continue
    if not all(
        [
            osp.exists(os.path.join(task_dir, s))
            for s in ["gbar.pt", "sbar.pt", "stilde.pt", "covariance.pt"]
        ]
    ):
        continue
    cov = torch.load(os.path.join(task_dir, "covariance.pt"))
    gbar = torch.load(os.path.join(task_dir, "gbar.pt"))
    sbar = torch.load(os.path.join(task_dir, "sbar.pt"))
    stilde = torch.load(os.path.join(task_dir, "stilde.pt"))

    layer_idx = 0
    for l in gbar:
        lc = "image_encoder." + l
        # print(f"{l}: {gbar[l].shape}")
        # cross_cosim = cosine_similarity(gbar[l].T @ gbar[l], sbar[l])
        # corr_cosim = cosine_similarity(sbar[l], stilde[l])
        # drift_cosim = cosine_similarity(stilde[l], cov[lc])
        # tot_cosim = cosine_similarity(gbar[l].T @ gbar[l], cov[lc])

        cross_ad = ad_fn(cosine_similarity(gbar[l].T @ gbar[l], sbar[l]))
        corr_ad = ad_fn(cosine_similarity(sbar[l], stilde[l]))
        drift_ad = ad_fn(cosine_similarity(stilde[l], cov[lc]))
        tot_ad = ad_fn(cosine_similarity(gbar[l].T @ gbar[l], cov[lc]))

        # # deltas
        delta_gsc = (
            tot_ad
            - ad_fn(cosine_similarity(gbar[l].T @ gbar[l], sbar[l]))
            - ad_fn(cosine_similarity(sbar[l], cov[lc]))
        )
        delta_ssc = (
            ad_fn(cosine_similarity(sbar[l], cov[lc]))
            - ad_fn(cosine_similarity(sbar[l], stilde[l]))
            - ad_fn(cosine_similarity(stilde[l], cov[lc]))
        )

        rows.extend(
            [
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": cross_ad.item(),
                    "type": "cross",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": corr_ad.item(),
                    "type": "corr",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": drift_ad.item(),
                    "type": "drift",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": tot_ad.item(),
                    "type": "tot",
                },
                # deltas
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": delta_gsc.item(),
                    "type": r"$\delta_\text{gsc}$",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": delta_ssc.item(),
                    "type": r"$\delta_\text{ssc}$",
                },
            ]
        )
        layer_idx += 1

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(results_dir, 'error_terms.csv')}")
