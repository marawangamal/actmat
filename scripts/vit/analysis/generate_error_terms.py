import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experts-dir", default="artifacts/checkpoints/ViT-B-16/group-fft-8/experts"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/results/ViT-B-16/group-fft-8"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
        expert_dir = osp.join(args.experts_dir, d)
        if not osp.isdir(expert_dir):
            continue
        if not all(
            osp.exists(osp.join(expert_dir, s))
            for s in ["gbar.pt", "sbar.pt", "stilde.pt", "covariance.pt"]
        ):
            continue
        cov = torch.load(osp.join(expert_dir, "covariance.pt"), map_location="cpu")
        gbar = torch.load(osp.join(expert_dir, "gbar.pt"), map_location="cpu")
        sbar = torch.load(osp.join(expert_dir, "sbar.pt"), map_location="cpu")
        stilde = torch.load(osp.join(expert_dir, "stilde.pt"), map_location="cpu")

        layer_idx = 0
        for l in gbar:
            lc = "image_encoder." + l

            cross_ad = ad_fn(cosine_similarity(gbar[l].T @ gbar[l], sbar[l]))
            corr_ad = ad_fn(cosine_similarity(sbar[l], stilde[l]))
            drift_ad = ad_fn(cosine_similarity(stilde[l], cov[lc]))
            tot_ad = ad_fn(cosine_similarity(gbar[l].T @ gbar[l], cov[lc]))

            # deltas
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
                    # upper bound
                    {
                        "dataset": d,
                        "layer_name": l,
                        "layer_idx": layer_idx,
                        "angular_distance": cross_ad.item()
                        + corr_ad.item()
                        + drift_ad.item(),
                        "type": "upper",
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

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = osp.join(args.output_dir, "error_terms.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
