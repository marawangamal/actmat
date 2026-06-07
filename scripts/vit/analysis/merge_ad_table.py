"""RegMean-family merge table: avg_top1 vs. how far each method's effective
covariance estimate sits from the true (collected) covariance.

Every RegMean-style method merges with SOME covariance estimate per layer:
    regmean        -> c            (the collected covariance, distance 0)
    mean           -> I            (covariance discarded)
    regmean_interp -> _interp_cov(c, ad*pi)
    actmat         -> dᵀd          (data-free second moment of the task delta)
We measure angle(c, cov_est_fn(d, c)) in units of pi (matching
generate_error_terms.py), pooled over all (task, layer) pairs, and pair it with
the avg_top1 that method achieved. Experts are loaded ONCE; each config just
re-indexes them in memory via cov_est_fn.

Run: python scripts/vit/analysis/merge_ad_table.py --model ViT-B-16 --group fft-8
"""

import argparse
import json
import math
import os.path as osp
import sys

import pandas as pd
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.append(REPO_ROOT)

from src.mergingv2 import _interp_cov  # noqa: E402
from src.vit.experts import ViTExpert, optional_sidecar  # noqa: E402

cos = lambda a, b: (a * b).sum() / (a.norm() * b.norm())
ad_fn = lambda a, b: math.acos(cos(a, b).clamp(-1, 1).item()) / math.pi  # units of pi

DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
ADS = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55]  # units of pi

configs = [
    {"name": "regmean", "results": "regmean", "cov_est_fn": lambda d, c: c},
    *[
        {
            "name": f"regmean_interp@{ad:g}",
            "results": f"regmean_interp-ad{ad:g}",
            "cov_est_fn": lambda d, c, ad=ad: _interp_cov(c, ad * math.pi),
        }
        for ad in ADS
    ],
    {"name": "actmat", "results": "actmat", "cov_est_fn": lambda d, c: d.T @ d},
    {
        "name": "mean",
        "results": "mean",
        "cov_est_fn": lambda d, c: torch.eye(c.shape[-1]),
    },
]


def compute_cov_estimate_stats(experts, base, cov_est_fn):
    """Angular distance (units of pi) from true cov c to cov_est_fn(d, c),
    pooled over all (expert, layer) pairs. d: (Do, Di), c: (Di, Di)."""
    ads = []
    for exp in experts:
        for layer in exp.get_layers():
            c, d = exp.get_layer_cov(layer), exp.get_layer_params(layer)
            if c is None or d.ndim != 2 or c.shape[-1] != d.shape[-1]:
                continue
            d = (d - base.get_layer_params(layer)).float()
            ads.append(ad_fn(c.float(), cov_est_fn(d, c.float())))
    ads = torch.tensor(ads)
    return {
        "ad_mean": ads.mean().item(),
        "ad_min": ads.min().item(),
        "ad_median": ads.median().item(),
        "ad_max": ads.max().item(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    p.add_argument("--results-dir", default="artifacts/results")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--group", default="fft-8")
    return p.parse_args()


def main():
    args = parse_args()
    experts_dir = osp.join(args.ckpt_dir, args.model, f"group-{args.group}", "experts")
    results_dir = osp.join(
        args.results_dir, args.model, f"group-{args.group}", "merged"
    )

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

    rows = []
    for cfg in configs:
        path = osp.join(results_dir, cfg["results"], "metrics.json")
        if not osp.exists(path):
            print(f"[skipped] {path}")
            continue
        avg_top1 = json.load(open(path))["model_config"]["avg_top1"] * 100
        stats = compute_cov_estimate_stats(experts, base, cfg["cov_est_fn"])
        rows.append({"method": cfg["name"], "avg_top1": avg_top1, **stats})
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
