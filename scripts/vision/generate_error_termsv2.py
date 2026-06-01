import argparse
import torch
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import sys

from src.utils import expert_dir

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


# --model ViT-B-16 --max-samples 1280
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="artifacts/checkpoints-analysisv2-epochs1")
parser.add_argument("--results_dir", default="artifacts/results-analysisv2-epochs1")
parser.add_argument("--model", default="ViT-B-16")
args = parser.parse_args()

model = args.model
ckpt_dir = osp.join(args.ckpt_dir, model)
results_dir = osp.join(args.results_dir, model)
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

    for l in gbar:
        lc = "image_encoder." + l
        # print(f"{l}: {gbar[l].shape}")
        cross_cosim = cosine_similarity(gbar[l].T @ gbar[l], sbar[l])
        corr_cosim = cosine_similarity(sbar[l], stilde[l])
        drift_cosim = cosine_similarity(stilde[l], cov[lc])
        tot_cosim = cosine_similarity(gbar[l].T @ gbar[l], cov[lc])
        rows.extend(
            [
                {
                    "dataset": d,
                    "layer_name": l,
                    "cosine_similarity": cross_cosim.item(),
                    "type": "cross",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "cosine_similarity": corr_cosim.item(),
                    "type": "corr",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "cosine_similarity": drift_cosim.item(),
                    "type": "drift",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": l,
                    "cosine_similarity": tot_cosim.item(),
                    "type": "tot",
                    "mode": "true",
                },
            ]
        )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(results_dir, 'error_terms.csv')}")
