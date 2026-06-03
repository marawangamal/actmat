"""Pairwise Frobenius-cosine heatmap of equal-dim input covariances.

Drops exact-duplicate covariances (e.g. self-attn k/v share q's input) keeping
one representative, then plots fcos(A,B)=<A,B>_F/(||A||||B||) for every kept
pair, ordered by forward-pass (dict) order. Title carries the layer count.

    python scripts/analysis/cov_pairwise_heatmap.py --cov <path> --tag <name> \
        --pipeline vision --dim 1024
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def short(k, pipeline):
    if pipeline == "vision":
        m = re.search(r"resblocks\.(\d+)\.(.+)$", k)
        return f"{int(m.group(1))}.{m.group(2)}" if m else k
    stack = "E" if ".encoder." in k else "D"
    m = re.search(r"block\.(\d+)\.layer\.\d+\.(.+)$", k)
    if not m:
        return k.split(".")[-1]
    comp = (m.group(2).replace("SelfAttention", "SA")
            .replace("EncDecAttention", "XA").replace("DenseReluDense", "FF"))
    return f"{stack}{int(m.group(1))}.{comp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pipeline", choices=["vision", "language"], default="vision")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--out-dir", default="artifacts/analysis/cov-layer-sim")
    args = ap.parse_args()

    cov = torch.load(args.cov, map_location="cpu", weights_only=False)
    covs = {k: v for k, v in cov.items()
            if not k.endswith("_n") and torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == args.dim}
    raw_n = len(covs)

    # drop exact duplicates (fingerprint then allclose), keep first in dict order
    groups = defaultdict(list)
    for k, v in covs.items():
        groups[(round(float(v.sum()), 2), round(float(v.float().norm()), 2))].append(k)
    drop = set()
    for ks in groups.values():
        reps = []
        for k in ks:
            if any(torch.allclose(covs[k], covs[r]) for r in reps):
                drop.add(k)
            else:
                reps.append(k)
    keep = [k for k in covs if k not in drop]
    N = len(keep)
    print(f"{args.tag}: {raw_n} {args.dim}-dim layers, {len(drop)} exact dups dropped -> N={N}")

    flat = [covs[k].float().flatten() for k in keep]
    norm = [f.norm() + 1e-12 for f in flat]
    S = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = float(flat[i] @ flat[j] / (norm[i] * norm[j]))

    labels = [short(k, args.pipeline) for k in keep]
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=90, fontsize=3.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=3.5)
    if args.pipeline == "language":
        n_enc = sum(1 for k in keep if ".encoder." in k)
        if 0 < n_enc < N:
            ax.axhline(n_enc - 0.5, color="r", lw=0.8); ax.axvline(n_enc - 0.5, color="r", lw=0.8)
    off = S[np.triu_indices(N, 1)]
    ax.set_title(f"{args.tag}: pairwise Frobenius cosine of {args.dim}-dim covariances "
                 f"(N={N}; {len(drop)} dups dropped; off-diag mean={off.mean():.3f})", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"{args.tag}_dim{args.dim}_pairwise.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out, "| off-diag mean fcos =", round(float(off.mean()), 4))


if __name__ == "__main__":
    main()
