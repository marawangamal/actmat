"""Cross error term vs. training-set size, per dataset, across a few checkpoints.

The `cross` term is the angular distance ad(gbar^T@gbar, sbar) — the gap between
the outer product of the *mean* gradient and the mean of per-sample gradient
outer products (i.e. the cross-sample covariance the decomposition isolates).
`cross_0` is the --checkpoint-first single-batch estimate; the other `cross_{N}`
columns are the same term at later step checkpoints, to see whether the size
relationship (or lack of it) holds as the stats accumulate.

rows = one per dataset:
    num_samples : training samples the expert trained on
                  (len of the `{dataset}Val` train split, matching finetune.py)
    cross_{N}   : mean over layers of the `cross` angular distance at step N

cross values are reused from the long-form CSV produced by
error_terms_over_time.py, so no `.pt` reload is needed.
"""

import os
import os.path as osp
import sys
import argparse

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(REPO_ROOT)

from src.vision.datasets.registry import get_dataset

# ---- config ---- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--terms-csv",
    default="artifacts/agents/error_terms_over_time/error_terms_over_time.csv",
)
parser.add_argument("--data-location", default="artifacts/data/vision")
parser.add_argument("--output-dir", default="artifacts/agents/cross_vs_numsamples")
parser.add_argument(
    "--steps", default="0,200,400,800,1400", help="comma-separated step checkpoints"
)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
steps = [int(s) for s in args.steps.split(",")]

# len() never touches pixels, but the dataset classes still want a callable.
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ---- cross per dataset x step (mean over layers) ---- #
df = pd.read_csv(args.terms_csv)
cross = (
    df[(df.type == "cross") & (df.step.isin(steps))]
    .groupby(["dataset", "step"])["angular_distance"]
    .mean()
    .unstack("step")  # columns = steps
)

# ---- training-set size per dataset (matches finetune.py's {ds}Val split) ---- #
rows = []
for dataset in cross.index:
    ds = get_dataset(
        f"{dataset}Val", preprocess, location=args.data_location, num_workers=2
    )
    rows.append(
        {
            "dataset": dataset,
            "num_samples": len(ds.train_dataset),
            **{f"cross_{s}": cross.loc[dataset, s] for s in steps},
        }
    )
df = pd.DataFrame(rows).sort_values("num_samples")
csv_path = osp.join(args.output_dir, "cross_vs_numsamples.csv")
df.to_csv(csv_path, index=False)
print(df.round(4).to_string(index=False))
print(f"Saved {csv_path}")

# ---- scatter: one series per checkpoint ---- #
long = df.melt(
    id_vars=["dataset", "num_samples"], value_name="cross", var_name="step"
)
long["step"] = long["step"].str.removeprefix("cross_").astype(int)
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(long, x="num_samples", y="cross", hue="step", palette="viridis", s=70, ax=ax)
for _, r in df.iterrows():  # label each dataset once, at its step-0 point
    ax.annotate(r["dataset"], (r["num_samples"], r[f"cross_{steps[0]}"]),
                textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("Training samples")
ax.set_ylabel("cross error  ad(G, sbar)")
ax.set_title("Cross error vs. dataset size, across checkpoints")
fig.tight_layout()
png_path = osp.join(args.output_dir, "cross_vs_numsamples.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved figure to {png_path}")
