"""Test the hypothesis: input-covariances of the SAME layer TYPE across blocks
are more similar to each other than to covariances of OTHER layer types.

For each expert we have one input second-moment matrix  C_L = Σ x xᵀ / n  per
linear layer L (collected by scripts/{vision,language}/covariance.py via the
split-MHA hooks in src/mhas.py — so attention q/k/v/o are separate Linears).

Similarity metric (dimension-restricted): Frobenius cosine
    fcos(A, B) = <A, B>_F / (||A||_F ||B||_F)
valid only between equal-dim matrices. We therefore compare within each matrix
size separately (e.g. ViT: the 1024-dim block {attn_in, o, c_fc} and the
4096-dim c_proj on its own).

A "layer type" is the key with its *block index* abstracted away. For ViT,
attn / attn.q / attn.k / attn.v are bit-identical in self-attention (shared
residual input) so they are collapsed into a single type ``attn_in``.

Outputs (to --out-dir, default artifacts/analysis/cov-layer-sim/):
  - <tag>_heatmap.png   : pairwise fcos, layers ordered by (type, block)
  - <tag>_summary.csv   : per-type within-type mean, vs between-type mean

Run from repo root:
    export PYTHONPATH="$PYTHONPATH:$PWD"
    python scripts/analysis/cov_layer_type_similarity.py \
        --cov artifacts/checkpoints/ViT-L-14/group-20/experts/SVHNVal/covariance.pt \
        --pipeline vision --tag vit-l-14_svhn
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def layer_type(key: str, pipeline: str) -> tuple[str, int]:
    """Return (type, block_index) for a covariance key, or (key, -1) if no block."""
    if pipeline == "vision":
        m = re.search(r"resblocks\.(\d+)\.(.+)$", key)
        if not m:
            return key, -1
        block, rest = int(m.group(1)), m.group(2)
        # self-attention q/k/v share the residual input and are bit-identical to
        # the module-level `attn` cov -> collapse into one type.
        if rest in ("attn", "attn.q", "attn.k", "attn.v"):
            rest = "attn_in"
        elif rest == "attn.o":
            rest = "attn_o"
        elif rest == "mlp.c_fc":
            rest = "mlp_c_fc"
        elif rest == "mlp.c_proj":
            rest = "mlp_c_proj"
        return rest, block
    else:  # language (T5)
        m = re.search(r"block\.(\d+)\.(.+)$", key)
        if not m:
            return key, -1
        block, rest = int(m.group(1)), m.group(2)
        stack = "enc" if ".encoder." in key else "dec"
        rest = rest.replace("layer.0.", "").replace("layer.1.", "").replace("layer.2.", "")
        return f"{stack}.{rest}", block


def load_covs(path: str):
    """Return {key: (C/n) float32 tensor} for matrix-valued (non _n) entries."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in raw.items():
        if k.endswith("_n") or not torch.is_tensor(v) or v.ndim != 2:
            continue
        n = raw.get(f"{k}_n", 1) or 1
        out[k] = (v.float() / n)
    return out


def fcos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True, help="path to covariance.pt")
    ap.add_argument("--pipeline", choices=["vision", "language"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="artifacts/analysis/cov-layer-sim")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    covs = load_covs(args.cov)
    # Dedupe ViT attn input copies: keep only the module-level `attn` key per block
    # (drops attn.q/.k/.v which are identical), keeping one representative.
    items = []  # (key, type, block, dim, tensor)
    seen_attn_in = set()
    for k, C in covs.items():
        t, b = layer_type(k, args.pipeline)
        if t == "attn_in":
            if b in seen_attn_in:
                continue
            if not k.endswith(".attn"):  # prefer the module-level key
                # only keep .attn; skip .attn.q/.k/.v
                if not k.endswith((".attn.q",)):
                    continue
            seen_attn_in.add(b)
        items.append((k, t, b, C.shape[0], C))

    # group by matrix dim
    by_dim = defaultdict(list)
    for it in items:
        by_dim[it[3]].append(it)

    summary_rows = []
    for dim, group in sorted(by_dim.items()):
        # order by (type, block)
        group = sorted(group, key=lambda x: (x[1], x[2]))
        labels = [f"{t}[{b}]" for (_, t, b, _, _) in group]
        types = [t for (_, t, _, _, _) in group]
        N = len(group)
        S = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i, N):
                s = fcos(group[i][4], group[j][4])
                S[i, j] = S[j, i] = s

        # within vs between
        within, between = [], []
        per_type_within = defaultdict(list)
        for i in range(N):
            for j in range(i + 1, N):
                if types[i] == types[j]:
                    within.append(S[i, j])
                    per_type_within[types[i]].append(S[i, j])
                else:
                    between.append(S[i, j])
        wm = float(np.mean(within)) if within else float("nan")
        bm = float(np.mean(between)) if between else float("nan")
        print(f"\n=== dim {dim}x{dim}  ({N} layers, types={sorted(set(types))}) ===")
        print(f"  within-type  mean fcos = {wm:.4f}  (n={len(within)})")
        print(f"  between-type mean fcos = {bm:.4f}  (n={len(between)})")
        print(f"  gap (within - between)  = {wm - bm:+.4f}")
        for t in sorted(per_type_within):
            vals = per_type_within[t]
            print(f"    {t:14s} within mean = {np.mean(vals):.4f}  (n={len(vals)})")
            summary_rows.append([args.tag, dim, t, f"{np.mean(vals):.4f}", len(vals), f"{bm:.4f}"])

        # heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(6, N * 0.12), max(5, N * 0.12)))
            im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
            # type boundaries
            bnds = [i for i in range(1, N) if types[i] != types[i - 1]]
            for x in bnds:
                ax.axhline(x - 0.5, color="w", lw=0.6)
                ax.axvline(x - 0.5, color="w", lw=0.6)
            # tick at center of each type block
            centers, tlabels, start = [], [], 0
            for i in range(1, N + 1):
                if i == N or types[i] != types[start]:
                    centers.append((start + i - 1) / 2)
                    tlabels.append(types[start])
                    start = i
            ax.set_xticks(centers); ax.set_xticklabels(tlabels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(centers); ax.set_yticklabels(tlabels, fontsize=8)
            ax.set_title(f"{args.tag}  dim={dim}  (Frobenius cosine)\nwithin={wm:.3f} between={bm:.3f}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            p = out_dir / f"{args.tag}_dim{dim}_heatmap.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"  saved {p}")
        except Exception as e:
            print(f"  (heatmap skipped: {e})")

    csv_path = out_dir / f"{args.tag}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "dim", "type", "within_mean_fcos", "n_within_pairs", "between_mean_fcos"])
        w.writerows(summary_rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
