import os
import sys
import argparse
from tqdm import tqdm
import os.path as osp

import numpy as np
import torch
import pandas as pd


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


def get_rand_psd(
    m: int, n: int, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    X = torch.randn(m, n, dtype=dtype, device=device)
    return X @ X.T


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experts-dir", default="artifacts/checkpoints/ViT-B-16/group-fft-8/experts"
)
parser.add_argument(
    "--output-dir", default="artifacts/results/ViT-B-16/group-fft-8/experts"
)
parser.add_argument("--expert-kind", default="vit")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

ad_fn = lambda c: np.arccos(c.clip(-1.0, 1.0)) / np.pi
stat_key_to_cov_key = {
    "vit": lambda x: "image_encoder." + x,
}

rows = []
for expert in tqdm(os.listdir(args.experts_dir), desc="experts"):
    task_dir = os.path.join(args.experts_dir, expert)
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
        lc = stat_key_to_cov_key.get(args.expert_kind, lambda x: x)(l)
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
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": cross_ad.item(),
                    "type": "cross",
                },
                {
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": corr_ad.item(),
                    "type": "corr",
                },
                {
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": drift_ad.item(),
                    "type": "drift",
                },
                {
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": tot_ad.item(),
                    "type": "tot",
                },
                # deltas
                {
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": delta_gsc.item(),
                    "type": r"$\delta_\text{gsc}$",
                },
                {
                    "dataset": expert,
                    "layer_name": l,
                    "layer_idx": layer_idx,
                    "angular_distance": delta_ssc.item(),
                    "type": r"$\delta_\text{ssc}$",
                },
            ]
        )
        layer_idx += 1

df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.output_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(args.output_dir, 'error_terms.csv')}")
