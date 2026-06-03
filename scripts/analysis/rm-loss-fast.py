"""Fast variant of rm-loss.py (compute only; plotting is rm-loss-fast-plot.py).

Same math as scripts/analysis/rm-loss.py, but each ~850MB finetuned.pt is
unpickled exactly once (experts-outer) instead of once per layer. Per-expert
diffs + covariances are cached in fp32 (~9GB total). Finishes in ~1 min; run
it with enough RAM (e.g. a 32GB allocation).

Writes per-(model, layer, method) RegMean-loss rows to
artifacts/analysis/rm-loss/rm_loss_vl.json. Mirrors the OLMo split
(rm-loss-olmo.py + rm-loss-olmo-plot.py).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

tr_abt = lambda a, b: (a * b).sum()


def compute_rm_loss_v1(w, d, c):
    """Compute the RegMean loss for linear layer at a given w

    Args:
        w: Shape: (Do, Di)
        d: Shape: (T, Do, Di)
        c: Shape: (T, Di, Di)
    """
    diff = w.unsqueeze(0) - d  # (T, Do, Di)
    loss = (diff @ c).mul_(diff).sum()
    return loss


def compute_rm_loss_v2(w, w_t, c_t):
    """Compute the RegMean loss for linear layer at a given w

    Args:
        w: Shape: (Do, Di)
        w_t: Shape: (T, Do, Di)
        c: Shape: (T, Di, Di)
    """
    w_test = w.unsqueeze(0)
    # loss_1 = torch.trace((w_test @ c_t @ w_test.transpose(-2, -1)).sum(0))
    loss_1 = tr_abt(w_test @ c_t, w_test)
    # loss_2 = torch.trace((w_t @ c_t @ w_t.transpose(-2, -1)).sum(0))
    loss_2 = tr_abt(w_t @ c_t, w_t)
    # loss_3 = torch.trace((w_t @ c_t @ w_test.transpose(-2, -1)).sum(0))
    loss_3 = tr_abt(w_t @ c_t, w_test)
    return loss_1 + loss_2 - 2 * loss_3


def compute_rm_minimizer(w0, d, c):
    return w0 + ((d @ c).sum(dim=0) @ torch.linalg.pinv(c.sum(dim=0)))


def load_state_dict(path):
    sd = torch.load(
        path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    if not isinstance(sd, dict):
        return sd.state_dict()
    return sd


rows = []
# model = "t5-base"
# group = "main"
# experts = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

vit_experts = [
    d + "Val"
    for d in ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
]
t5_experts = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]
configs = [
    {
        "model": "ViT-B-16",
        "experts": vit_experts,
        "group": "8",
        "lc_fn": lambda l: "image_encoder." + l.replace(".weight", ""),
    },
    {
        "model": "t5-base",
        "experts": t5_experts,
        "group": "main",
        "lc_fn": lambda l: l.replace(".weight", ""),
    },
]
for cfg in configs:
    model, experts, group, lc_fn = (
        cfg["model"],
        cfg["experts"],
        cfg["group"],
        cfg["lc_fn"],
    )
    w_0 = torch.load(
        REPO_ROOT / f"artifacts/checkpoints/{model}/pretrained.pt",
        weights_only=False,
        map_location=torch.device("cpu"),
    ).state_dict()

    # Unpickle each expert ONCE, caching only tracked-layer diffs + covariance.
    diffs = defaultdict(list)  # layer -> [Δ_t, ...]
    covs = defaultdict(list)  # layer -> [C_t, ...]
    for e in tqdm(experts, desc="load experts"):
        expert_dir = (
            REPO_ROOT / f"artifacts/checkpoints/{model}/group-{group}/experts/{e}"
        )
        c_dict = torch.load(expert_dir / "covariance.pt", map_location="cpu")
        w_t = load_state_dict(expert_dir / "finetuned.pt")
        for l in w_0:
            # lc = l.replace(".weight", "")
            lc = lc_fn(l)
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
        c_eye = torch.stack(
            [torch.eye(c.size(-2), c.size(-1)) for _ in range(c.size(0))]
        )
        c_methods = zip(
            ["actmat", "regmean", "identity"], [d.transpose(-2, -1) @ d, c, c_eye]
        )

        for method, c_hat in c_methods:
            w_star = compute_rm_minimizer(w_0[l], d, c_hat)
            # loss = compute_rm_loss_v2(w_star, d + w_0[l].unsqueeze(0), c_hat)
            loss = compute_rm_loss_v2(w_star - w_0[l], d, c)
            rows.append(
                {
                    "model": model,
                    "group": group,
                    "method": method,
                    "metric_type": "rm_loss",
                    "metric": loss.item(),
                    "layer": l,
                    "Di": w_star.shape[-1],
                    "Do": w_star.shape[-2],
                }
            )

out = REPO_ROOT / "artifacts/analysis/rm-loss/rm_loss_vl.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} rows -> {out}")
