"""Fast variant of rm-loss.py.

Same math as scripts/analysis/rm-loss.py, but each ~850MB finetuned.pt is
unpickled exactly once (experts-outer) instead of once per layer. Per-expert
diffs + covariances are cached in fp32 (~9GB total). Finishes in ~1 min; run
it with enough RAM (e.g. a 32GB allocation).
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


def compute_rm_loss(w, d, c):
    """Compute the RegMean loss for linear layer at a given w

    Args:
        w: Shape: (Do, Di)
        d: Shape: (T, Do, Di)
        c: Shape: (T, Di, Di)
    """
    diff = w.unsqueeze(0) - d  # (T, Do, Di)
    loss = (diff @ c).mul_(diff).sum()
    return loss


def compute_rm_minimizer(w0, d, c):
    return w0 + ((d @ c).sum(dim=0) @ torch.linalg.pinv(c.sum(dim=0)))


rows = []
model = "t5-base"
group = "main"
experts = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]
w_0 = torch.load(
    REPO_ROOT / f"artifacts/checkpoints/{model}/pretrained.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
).state_dict()

# Unpickle each expert ONCE, caching only tracked-layer diffs + covariance.
diffs = defaultdict(list)  # layer -> [Δ_t, ...]
covs = defaultdict(list)  # layer -> [C_t, ...]
for e in tqdm(experts, desc="load experts"):
    expert_dir = REPO_ROOT / f"artifacts/checkpoints/{model}/group-{group}/experts/{e}"
    c_dict = torch.load(expert_dir / "covariance.pt", map_location="cpu")
    w_t = torch.load(
        expert_dir / "finetuned.pt",
        weights_only=False,
        map_location=torch.device("cpu"),
    ).state_dict()
    for l in w_0:
        lc = l.replace(".weight", "")
        if lc not in c_dict:
            continue
        diffs[l].append(w_t[l] - w_0[l])
        covs[l].append(c_dict[lc])
    del w_t, c_dict

for l in tqdm(w_0, desc="layers"):
    if l not in diffs:
        continue
    d = torch.stack(diffs[l])  # (T, Do, Di)
    c = torch.stack(covs[l])  # (T, Di, Di)
    c_methods = zip(["actmat", "regmean"], [d.transpose(-2, -1) @ d, c])

    for method, c_hat in c_methods:
        w_star = compute_rm_minimizer(w_0[l], d, c_hat)
        loss = compute_rm_loss(w_star, d, c_hat)
        rows.append(
            {
                "model": model,
                "group": group,
                "method": method,
                "metric_type": "rm_loss",
                "metric": loss.item(),
                "layer": l,
            }
        )


import pandas as pd
import seaborn as sns

df = pd.DataFrame(rows)
print(df)

g = sns.relplot(data=df, x="layer", y="metric", hue="method")
g.set_xticklabels(rotation=90)
out = REPO_ROOT / "artifacts" / "results-analysis" / "rm_loss.png"
out.parent.mkdir(parents=True, exist_ok=True)
g.savefig(out, bbox_inches="tight")
print(f"saved plot -> {out}")
