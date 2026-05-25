Fig. 8: Scaling Coeff Analysis
import sys
import os
import math
from tqdm import tqdm
from itertools import product

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..") # Add src to path
import src.mhas as mhas
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector


MODELS = ["ViT-B-16"]
DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"]
FT_METHODS = ["standard"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_SAMPLES = 10
MAX_LAYERS = None
RESULTS_ROOT = "../results"
CHECKPOINTS_ROOT = "../checkpoints"
SAVE_PATH = os.path.join(RESULTS_ROOT, f"eigcov_scaling_coef_n{NUM_SAMPLES}_b{BATCH_SIZE}.csv")

param_name_to_module_name = lambda name: "image_encoder." + name.replace(".weight", "")
def dist_cos_noabs(A: torch.Tensor, B: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    a = A.flatten()
    b = B.flatten()
    cos_theta = torch.dot(a, b) / (a.norm() * b.norm())
    return (1 - cos_theta)

max_layers_per_combo = MAX_LAYERS + 1 if MAX_LAYERS is not None else None
TOTAL = len(MODELS) * len(FT_METHODS) * len(DATASETS) * (max_layers_per_combo or 0)
global_step = 0

rows = []
for model_name in MODELS:
    results_dir = os.path.join(RESULTS_ROOT, model_name)
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, model_name)

    for ft_method, dataset in product(FT_METHODS, DATASETS):
        # Load covariance
        covs = np.load(f"{results_dir}/covariances_strain_n{NUM_SAMPLES}_b{BATCH_SIZE}_tsm_attnsplit_efull_ft{ft_method}/covariance_{dataset}.npz", allow_pickle=True)

        # Load task vector
        if ft_method == "linear":
            pretrained_checkpoint = f"{checkpoint_dir}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{checkpoint_dir}/{dataset}Val/linear_finetuned.pt"
            pretrained_nonlinear_checkpoint = f"{checkpoint_dir}/{dataset}Val/zeroshot.pt"

            nonlinear_encoder = torch.load(
                pretrained_nonlinear_checkpoint, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder

            tv = LinearizedTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
            encoder = tv.apply_to_nonlinear(
                pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
            )
            task_vector = NonLinearTaskVector(
                vector=encoder.state_dict(),
            )
        elif ft_method == "lora":
            pretrained_checkpoint = f"{checkpoint_dir}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{checkpoint_dir}/{dataset}Val/lora_finetuned.pt"
            task_vector = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
        else:
            pretrained_checkpoint = f"{checkpoint_dir}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{checkpoint_dir}/{dataset}Val/finetuned.pt"
            task_vector = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )

        task_vector = task_vector.map(mhas.copy_from_pytorch_state_dict)

        num_added = 0
        if TOTAL == 0 and MAX_LAYERS is None:
            TOTAL = len(MODELS) * len(FT_METHODS) * len(DATASETS) * len(task_vector.vector)
        for k, d in task_vector.vector.items():
            kp = param_name_to_module_name(k)
            if len(d.shape) != 2 or kp not in covs.keys():
                continue
            d = d.to(DEVICE)
            c = torch.from_numpy(covs[kp]).to(d.dtype).to(DEVICE)
            c_hat_ec = d.T @ d
            kappa = torch.linalg.norm(c_hat_ec, ord="fro") / torch.linalg.norm(c, ord="fro")
            kappa_star = torch.trace(c.T @ c_hat_ec) / (torch.linalg.norm(c_hat_ec, ord="fro")**2)
            rows.append({
                "model": model_name, 
                "dataset": dataset, 
                "layer_name": k,
                "layer_idx": num_added, 
                "kappa": kappa_star.item(),
            })
            global_step += 1
            print(f"[{global_step}/{TOTAL}] {model_name} | {ft_method} | {dataset} | layer {num_added} | {k.split('.')[-2]} | Kappa: {kappa.item()}")
            num_added += 1
            if MAX_LAYERS is not None and num_added > MAX_LAYERS:
                break

# 0. Create a dataframe
df = pd.DataFrame(rows)

# 1. Create a "rho" dataframe by merging the results with itself 
# We join on 'model', 'layer_name', and 'layer_idx' to pair datasets for the same layer
df = df.merge(
    df, 
    on=['model', 'layer_name', 'layer_idx'], 
    suffixes=('_i', '_j')
)

# 2. Compute rho = kappa_i / kappa_j
df['rho'] = df['kappa_i'] / df['kappa_j']
df.to_csv(SAVE_PATH, index=False)