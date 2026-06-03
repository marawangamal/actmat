import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


# compute regmean losses and put them under results/**/rm_metrics.json
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

for l in tqdm(w_0, desc="layers"):
    lc = l.replace(".weight", "")
    d, c = list(), list()
    for e in experts:
        c_dict = torch.load(
            REPO_ROOT
            / f"artifacts/checkpoints/{model}/group-{group}/experts/{e}/covariance.pt",
            map_location=torch.device("cpu"),
        )
        if lc not in c_dict:
            break
        w_t = torch.load(
            REPO_ROOT
            / f"artifacts/checkpoints/{model}/group-{group}/experts/{e}/finetuned.pt",
            weights_only=False,
            map_location=torch.device("cpu"),
        ).state_dict()[l]
        d.append(w_t - w_0[l])
        c.append(c_dict[lc])

    if not d:
        tqdm.write(f"[skipped] {lc}")
        continue

    d, c = torch.stack(d), torch.stack(c)
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
