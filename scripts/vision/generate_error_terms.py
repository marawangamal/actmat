import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
import sys

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
parser.add_argument("--model", default="ViT-B-16")
parser.add_argument("--max-samples", type=int, required=True)
args = parser.parse_args()

model = args.model
suffix = f"max_samples_{args.max_samples}"
checkpoints_dir = f"artifacts/checkpoints-analysis/{model}/{suffix}"
results_dir = f"artifacts/results-analysis/{model}/{suffix}"
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
    task_dir = os.path.join(checkpoints_dir, d + "Val")
    if not os.path.isdir(task_dir):
        continue
    cdict = torch.load(os.path.join(task_dir, "covariance.pt"))
    cdict = {k.replace(".", "_"): v for k, v in cdict.items()}
    filenames = [f for f in os.listdir(task_dir) if "grad_cross" in f]
    if not filenames:
        continue
    for filename in tqdm(filenames, desc=d, leave=False):
        layer_name = filename.replace(".pt", "").replace("grad_cross_matrix_", "")
        gcm = torch.load(os.path.join(task_dir, filename))
        cov = cdict["image_encoder_" + layer_name]
        m, n = gcm["gbar"].shape
        cosim_cross = cosine_similarity(gcm["gbar"].T @ gcm["gbar"], gcm["sbar"])
        cosim_corr = cosine_similarity(gcm["sbar"], gcm["stilde"])
        cosim_drift = cosine_similarity(gcm["stilde"], cov)
        cosim_tot = cosine_similarity(gcm["gbar"].T @ gcm["gbar"], cov)
        # # deltas
        # delta_gsc = (
        #     cosim_tot
        #     - cosine_similarity(gcm["gbar"].T @ gcm["gbar"], gcm["sbar"])
        #     - cosine_similarity(gcm["sbar"], cov)
        # )
        # delta_ssc = (
        #     cosine_similarity(gcm["sbar"], cov)
        #     - cosine_similarity(gcm["sbar"], gcm["stilde"])
        #     - cosine_similarity(gcm["stilde"], cov)
        # )

        # Controls:
        cosim_cross_ctrl = cosine_similarity(
            gcm["gbar"].T @ gcm["gbar"], get_rand_psd(n, n)
        )
        cosim_corr_ctrl = cosine_similarity(gcm["sbar"], get_rand_psd(n, n))

        # TODO: add drift term
        # cov_terminal = torch.randn_like(gcm["sbar"])
        # cosim_drift = cosine_similarity(gcm["sbar"], cov_terminal)
        rows.extend(
            [
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_cross.item(),
                    "type": "cross",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_corr.item(),
                    "type": "corr",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_drift.item(),
                    "type": "drift",
                    "mode": "true",
                },
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_tot.item(),
                    "type": "tot",
                    "mode": "true",
                },
                # Controls:
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_cross_ctrl.item(),
                    "type": "cross",
                    "mode": "ctrl",
                },
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_corr_ctrl.item(),
                    "type": "corr",
                    "mode": "ctrl",
                },
                # {
                #     "dataset": d,
                #     "layer_name": layer_name,
                #     "cosine_similarity": cosim_drift.item(),
                #     "type": "drift",
                # },
            ]
        )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(results_dir, 'error_terms.csv')}")
