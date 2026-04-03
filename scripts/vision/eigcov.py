"""Precompute per-layer mid-checkpoint covariance matrices for use with regmean.

For each dataset and each 2D weight matrix, computes:
    Delta_k = W_T - W_k          (weight delta from mid checkpoint to finetuned)
    C_k     = Delta_k.T @ Delta_k  (covariance used by regmean merge)

Keys in the output .pt match the format expected by merge_regmean, which looks
them up via param_key_to_cov_key():  "image_encoder.<module_path>" (no ".weight"
suffix).

Output is saved to the checkpoint directory as covariance.pt.

Usage:
    python scripts/vision/eigcov.py \
        --model ViT-B-16 \
        --mid-checkpoint-step 500

    python scripts/vision/eval_task_addition.py \
        --merge-func regmean
"""

import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.args import parse_arguments


ALL_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def param_key_to_cov_key(key: str) -> str:
    return "image_encoder." + key.replace(".weight", "")


if __name__ == "__main__":
    args = parse_arguments()

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    datasets = args.eval_datasets if args.eval_datasets is not None else ALL_DATASETS

    for dataset in datasets:
        checkpoint_dir = f"{args.save}/{dataset}Val"
        cov_path = os.path.join(checkpoint_dir, "covariance.pt")

        if args.finetuning_mode == "lora":
            ft_path = f"{checkpoint_dir}/lora_finetuned.pt"
        else:
            ft_path = f"{checkpoint_dir}/finetuned.pt"
        mid_path = f"{checkpoint_dir}/checkpoint_{args.mid_checkpoint_step}.pt"

        if args.eigcov_reverse:
            # Delta = W_k - W_0
            zs_path = f"{checkpoint_dir}/zeroshot.pt"
            if not os.path.exists(zs_path):
                print(f"[skip] {zs_path} not found")
                continue
        else:
            # Delta = W_T - W_k
            if not os.path.exists(ft_path):
                print(f"[skip] {ft_path} not found")
                continue
        if not os.path.exists(mid_path):
            print(f"[skip] {mid_path} not found")
            continue

        print(f"Processing {dataset} ...")
        mid_sd = load_state_dict(mid_path)
        if args.eigcov_reverse:
            ref_sd = load_state_dict(zs_path)
        else:
            ref_sd = load_state_dict(ft_path)

        covs = {}
        for key in mid_sd:
            if mid_sd[key].dtype in (torch.int64, torch.uint8):
                continue
            if mid_sd[key].ndim != 2:
                continue
            if args.eigcov_reverse:
                delta = (mid_sd[key] - ref_sd[key]).float()  # W_k - W_0
            else:
                delta = (ref_sd[key] - mid_sd[key]).float()  # W_T - W_k
            cov_key = param_key_to_cov_key(key)
            covs[cov_key] = delta.T @ delta

        torch.save(covs, cov_path)
        print(f"  Saved {len(covs)} covariance matrices -> {cov_path}")

        del ref_sd, mid_sd
        torch.cuda.empty_cache()
