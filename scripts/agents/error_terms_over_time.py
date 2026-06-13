"""Angular distance over training time between the three grad-cross terms.

`finetune.py --checkpoint-first --checkpoint-every N` snapshots the running
`gbar`/`sbar`/`stilde` stats as `{stat}_{step}.pt` next to each step checkpoint.
This lets us watch how the three terms drift apart *as training proceeds*,
instead of only at the final step (cf. scripts/analysis/generate_error_terms.py,
which needs a data-collected `covariance.pt` and only looks at the endpoint).

For each expert and each available step we compute the pairwise angular distance
(arccos of cosine similarity, normalised to [0, 1]) between the three objects the
error decomposition relates:

    G := gbar^T @ gbar   (outer second moment of the mean grad)
    sbar                 (grad-weighted input second moment)
    stilde               (input second moment scaled by mean grad-norm)

  cross : ad(G, sbar)
  corr  : ad(sbar, stilde)
  gst   : ad(G, stilde)

Each `.pt` is loaded ONCE per (expert, step) and indexed per layer, so we never
hold more than one step (~1.5 GB) in memory. Writes a long-form CSV and a
lineplot (step on x, angular distance on y, one line per term, aggregated over
layers and experts by seaborn) under artifacts/agents.
"""

import os
import os.path as osp
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(REPO_ROOT)

# ---- config ---- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experts-dir", default="artifacts/checkpoints/ViT-B-16/group-fft-8/experts"
)
parser.add_argument("--output-dir", default="artifacts/agents/error_terms_over_time")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

ad_fn = lambda c: np.arccos(float(c.clip(-1.0, 1.0))) / np.pi  # angular dist in [0,1]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of two flattened tensors. a, b: any matching shape."""
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


pair_to_terms = {  # pair label -> (left, right) keyed into the per-layer terms dict
    "cross": ("G", "sbar"),
    "corr": ("sbar", "stilde"),
    "gst": ("G", "stilde"),
}

# ---- load + measure ---- #
rows = []
for expert in tqdm(sorted(os.listdir(args.experts_dir)), desc="experts"):
    task_dir = osp.join(args.experts_dir, expert)
    steps = sorted(
        int(osp.basename(p)[len("gbar_") : -len(".pt")])
        for p in glob.glob(osp.join(task_dir, "gbar_*.pt"))
    )
    if not steps:
        print(f"[skipped] {task_dir} (no gbar_*.pt)")
        continue

    for step in steps:
        paths = {
            s: osp.join(task_dir, f"{s}_{step}.pt") for s in ("gbar", "sbar", "stilde")
        }
        if not all(osp.exists(p) for p in paths.values()):
            print(f"[skipped] {expert} step {step} (missing a stat file)")
            continue
        gbar = torch.load(paths["gbar"], map_location="cpu")
        sbar = torch.load(paths["sbar"], map_location="cpu")
        stilde = torch.load(paths["stilde"], map_location="cpu")

        for layer_idx, l in enumerate(gbar):
            terms = {"G": gbar[l].T @ gbar[l], "sbar": sbar[l], "stilde": stilde[l]}
            for pair, (a, b) in pair_to_terms.items():
                rows.append(
                    {
                        "dataset": expert,
                        "step": step,
                        "layer_name": l,
                        "layer_idx": layer_idx,
                        "angular_distance": ad_fn(
                            cosine_similarity(terms[a], terms[b])
                        ),
                        "type": pair,
                    }
                )

df = pd.DataFrame(rows)
csv_path = osp.join(args.output_dir, "error_terms_over_time.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} rows to {csv_path}")

# ---- plot ---- #
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(df, x="step", y="angular_distance", hue="type", marker="o", ax=ax)
ax.set_xlabel("Training step")
ax.set_ylabel("Angular distance")
ax.set_title("Grad-cross term distances over training (mean over layers, experts)")
fig.tight_layout()
png_path = osp.join(args.output_dir, "error_terms_over_time.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved figure to {png_path}")
