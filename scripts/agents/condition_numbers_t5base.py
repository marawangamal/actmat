"""Per-layer condition numbers for the two input-side estimators on t5-base.

For each Linear layer L (parameter ``…layer….weight`` with shape (Do, Di)):

  * RegMean estimator:   C_regmean[L]_t  =  Eₓ[x xᵀ]  collected on task t's data
                                            (loaded from covariance_old.pt;
                                             this is the Apr-30 train-set cov)
  * ACTMat estimator:    C_actmat[L]_t   =  d_tᵀ d_t   where d_t = θ_ft_t − θ_pre

Both are (Di, Di) PSD. We report cond = σ₁ / σ_k where k = min(Do, Di) — the
maximum possible rank of ACTMat's dᵀd. That makes the actmat side a real-valued
cond on its non-null subspace, and the regmean side a cond at a matched index in
the spectrum (apples-to-apples across layers regardless of matrix size).

Output columns per layer (CSV):
    layer, shape,
    rm_qasc, rm_wiki_qa, rm_quartz, rm_paws, rm_story_cloze, rm_winogrande, rm_wsc, rm_SUM,
    am_qasc, am_wiki_qa, am_quartz, am_paws, am_story_cloze, am_winogrande, am_wsc, am_SUM

where rm_SUM = cond(Σ_t C_regmean_t) and am_SUM = cond(Σ_t C_actmat_t).

Run from repo root:
    export PYTHONPATH="$PYTHONPATH:$PWD"
    python scripts/agents/condition_numbers_t5base.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.language.modeling import T5Wrapper  # noqa: F401 — needed to unpickle


TASKS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]


def cond_at_rank_k(M: torch.Tensor, k: int) -> float:
    """σ₁ / σ_k for a symmetric PSD matrix M (PSD ⇒ singular values = eigvals).

    k is 1-indexed (k=1 is the largest σ; k=D is the smallest). If the k-th
    largest σ is ≤ 0 (numerical zero / true rank deficiency past position k),
    returns +inf. No clipping or threshold.
    """
    M = 0.5 * (M + M.transpose(-1, -2))  # symmetrize against fp drift
    evs = torch.linalg.eigvalsh(M.to(torch.float64))  # ascending
    D = evs.shape[0]
    k = min(max(k, 1), D)
    sigma_1 = float(evs[-1])
    sigma_k = float(evs[-k])
    if sigma_1 <= 0 or sigma_k <= 0:
        return float("inf")
    return sigma_1 / sigma_k


def get_state_dict(obj):
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t5-base")
    ap.add_argument(
        "--cov-file",
        default="covariance_old.pt",
        help="Filename inside each dataset checkpoint dir.",
    )
    ap.add_argument(
        "--out",
        default="artifacts/agents/condition-numbers/t5-base-cond.csv",
    )
    args = ap.parse_args()

    ckpt_root = project_root / "artifacts" / "checkpoints" / args.model
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained from {ckpt_root}/pretrained.pt")
    pre = get_state_dict(torch.load(
        ckpt_root / "pretrained.pt", map_location="cpu", weights_only=False
    ))

    # Candidate 2D Linear weights — same filter as combine_task_vectors.
    weight_keys = [
        k for k, v in pre.items()
        if v.ndim == 2 and max(v.shape) < 20_000
    ]
    print(f"Found {len(weight_keys)} candidate 2D weights in pretrained")

    # Establish the intersection with covariance_old.pt keys (use task 0 as
    # a probe — every task uses the same module set).
    probe_cov_path = ckpt_root / TASKS[0] / args.cov_file
    probe_cov = torch.load(probe_cov_path, map_location="cpu", weights_only=False)
    cov_keys = {k for k in probe_cov if not k.endswith("_n")}
    del probe_cov

    def to_cov_key(param_key: str) -> str:
        return param_key.removesuffix(".weight")

    layer_keys = [k for k in weight_keys if to_cov_key(k) in cov_keys]
    skipped = [k for k in weight_keys if to_cov_key(k) not in cov_keys]
    print(f"Layers w/ matching cov entry: {len(layer_keys)}")
    if skipped:
        print(f"Skipped (no cov entry): {len(skipped)}")
        for k in skipped[:5]:
            print(f"  - {k}")
        if len(skipped) > 5:
            print(f"  ...({len(skipped) - 5} more)")

    # Per-task per-layer condition numbers.
    cond_rm: dict[str, list[float]] = {k: [] for k in layer_keys}
    cond_am: dict[str, list[float]] = {k: [] for k in layer_keys}
    # Running sums for cond-of-sum.
    sum_rm: dict[str, torch.Tensor] = {}
    sum_am: dict[str, torch.Tensor] = {}

    t0 = time.time()
    for ti, task in enumerate(TASKS):
        task_dir = ckpt_root / task
        print(f"\n[{ti+1}/{len(TASKS)}] task={task}")

        # ---- RegMean cov for this task ----
        cov_path = task_dir / args.cov_file
        print(f"  load {cov_path.name}")
        cov = torch.load(cov_path, map_location="cpu", weights_only=False)

        for k in layer_keys:
            ck = to_cov_key(k)
            M = cov[ck].to(torch.float32)
            rank_k = min(pre[k].shape)
            cond_rm[k].append(cond_at_rank_k(M, rank_k))
            sum_rm[k] = sum_rm.get(k, torch.zeros_like(M)) + M
        del cov
        gc.collect()

        # ---- ACTMat d.T @ d for this task ----
        ft_path = task_dir / "finetuned.pt"
        print(f"  load {ft_path.name}")
        ft = get_state_dict(torch.load(
            ft_path, map_location="cpu", weights_only=False
        ))

        for k in layer_keys:
            d = (ft[k] - pre[k]).to(torch.float32)   # (Do, Di)
            G = d.T @ d                              # (Di, Di)
            rank_k = min(pre[k].shape)
            cond_am[k].append(cond_at_rank_k(G, rank_k))
            sum_am[k] = sum_am.get(k, torch.zeros_like(G)) + G
        del ft
        gc.collect()

        print(f"  elapsed {(time.time()-t0)/60:.1f} min")

    # Cond of summed estimators.
    print("\nComputing cond of summed estimators...")
    cond_rm_sum = {
        k: cond_at_rank_k(sum_rm[k], min(pre[k].shape)) for k in layer_keys
    }
    cond_am_sum = {
        k: cond_at_rank_k(sum_am[k], min(pre[k].shape)) for k in layer_keys
    }

    # Write CSV.
    cols = (
        ["layer", "shape"]
        + [f"rm_{t}" for t in TASKS] + ["rm_SUM"]
        + [f"am_{t}" for t in TASKS] + ["am_SUM"]
    )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in layer_keys:
            shape = tuple(pre[k].shape)
            row = (
                [k, f"{shape[0]}x{shape[1]}"]
                + [f"{c:.4e}" for c in cond_rm[k]] + [f"{cond_rm_sum[k]:.4e}"]
                + [f"{c:.4e}" for c in cond_am[k]] + [f"{cond_am_sum[k]:.4e}"]
            )
            w.writerow(row)
    print(f"\nWrote {out_path}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
