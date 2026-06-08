"""Per-layer RegMean loss of each merge method against the true covariances.

For each layer, this builds the multi-expert merged delta from each method's
effective covariance estimate, then plots the per-(task, layer) contribution to
the true RegMean objective:

    tr((delta_merge - delta_task) c_task (delta_merge - delta_task)^T)

Run: python scripts/vit/analysis/layerwise-rmloss-hist.py --model ViT-B-16 --group fft-8
"""

import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.append(REPO_ROOT)

from src.vit.experts import ViTExpert, optional_sidecar  # noqa: E402

pinv = lambda c: torch.linalg.pinv(c, hermitian=True)

DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]

configs = {
    "regmean": lambda d, c: c,
    "actmat": lambda d, c: d.T @ d,
    "mean": lambda d, c: torch.eye(c.shape[-1], device=c.device, dtype=c.dtype),
}


def rm_loss_contribution(delta_merge, delta_task, c_task):
    err = delta_merge - delta_task
    return (err @ c_task * err).sum().item()


def merge_with_cov_estimates(deltas, cov_estimates):
    return (deltas @ cov_estimates).sum(dim=0) @ pinv(cov_estimates.sum(dim=0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--group", default="fft-8")
    p.add_argument("--out", default="artifacts/analysis/layerwise-rmloss-hist.png")
    return p.parse_args()


def main():
    args = parse_args()
    experts_dir = osp.join(args.ckpt_dir, args.model, f"group-{args.group}", "experts")
    base = ViTExpert(weights_path=osp.join(experts_dir, DATASETS[0], "pretrained.pt"))
    experts = [
        ViTExpert(
            weights_path=osp.join(experts_dir, ds, "finetuned.pt"),
            covariance_path=optional_sidecar(
                osp.join(experts_dir, ds), "covariance.pt"
            ),
        )
        for ds in DATASETS
    ]

    rows = []  # one RegMean-loss contribution per (task, layer, method)
    for layer in base.get_layers():
        w0 = base.get_layer_params(layer)
        if w0.ndim != 2:
            continue

        deltas = []
        covs = []
        task_names = []
        for ds, exp in zip(DATASETS, experts):
            c, w = exp.get_layer_cov(layer), exp.get_layer_params(layer)
            if c is None or w.ndim != 2 or c.shape[-1] != w.shape[-1]:
                continue
            deltas.append((w - w0).float())
            covs.append(c.float())
            task_names.append(ds)

        if not deltas:
            continue

        deltas = torch.stack(deltas)
        covs = torch.stack(covs)
        for method, fn in configs.items():
            cov_estimates = torch.stack(
                [fn(delta, cov) for delta, cov in zip(deltas, covs)]
            )
            delta_merge = merge_with_cov_estimates(deltas, cov_estimates)
            for ds, delta, cov in zip(task_names, deltas, covs):
                rows.append(
                    {
                        "task": ds,
                        "layer": layer,
                        "method": method,
                        "rm_loss": rm_loss_contribution(delta_merge, delta, cov),
                    }
                )

    df = pd.DataFrame(rows)
    df["log10_rm_loss"] = torch.log10(
        torch.as_tensor(df["rm_loss"].to_numpy()) + 1
    ).numpy()
    g = sns.displot(
        df,
        x="log10_rm_loss",
        hue="method",
        kind="hist",
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        height=4,
        aspect=1.5,
    )
    g.set_axis_labels(r"$\log_{10}(1 + \mathrm{RegMean\ loss\ contribution})$", "density")
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    g.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
