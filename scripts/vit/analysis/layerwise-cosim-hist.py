"""Per-layer cosine alignment of each merge method's effective covariance
estimate to the true (collected) covariance c.

The summary table reports only the MEAN |cos(c, cov_est_fn(d, c))| per method,
which hides the spread (actmat's mean was 0.30 but its min 0.007). Here we plot
the full per-(task, layer) distribution as overlaid histograms, one hue per
method, to contrast HOW each estimate aligns with c across the network — not
just on average. |cos| (cov estimates are PSD, so cos >= 0 anyway).

Run: python scripts/vit/analysis/layerwise-cosim-hist.py --model ViT-B-16 --group fft-8
"""

import argparse
import math
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.append(REPO_ROOT)

from src.merging import _interp_cov  # noqa: E402
from src.vit.experts import ViTExpert, optional_sidecar  # noqa: E402

abs_cos = lambda a, b: ((a * b).sum() / (a.norm() * b.norm())).abs().item()

DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]

# method -> effective covariance estimate; same cov_est_fn(d, c) contract as the
# table. regmean is the trivial reference (|cos| == 1 everywhere).
configs = {
    "regmean": lambda d, c: c,
    "interp@0.1": lambda d, c: _interp_cov(c, 0.1 * math.pi),
    "interp@0.3": lambda d, c: _interp_cov(c, 0.3 * math.pi),
    "actmat": lambda d, c: d.T @ d,
    "mean": lambda d, c: torch.eye(c.shape[-1]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--group", default="fft-8")
    p.add_argument("--out", default="artifacts/analysis/layerwise-cosim-hist.png")
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

    rows = []  # one |cos| per (task, layer, method)
    for ds, exp in zip(DATASETS, experts):
        for layer in exp.get_layers():
            c, d = exp.get_layer_cov(layer), exp.get_layer_params(layer)
            if c is None or d.ndim != 2 or c.shape[-1] != d.shape[-1]:
                continue
            c, d = c.float(), (d - base.get_layer_params(layer)).float()
            for method, fn in configs.items():
                rows.append({"method": method, "abs_cos": abs_cos(c, fn(d, c))})
    df = pd.DataFrame(rows)

    g = sns.displot(
        df,
        x="abs_cos",
        hue="method",
        kind="hist",
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        height=4,
        aspect=1.5,
    )
    g.set_axis_labels(r"$|\cos(c,\ \hat c)|$", "density")
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    g.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
