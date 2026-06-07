"""Evaluate per-task fine-tuned baselines on GLUE for normalized score comparisons.

For each of the 8 GLUE tasks, applies *that task's* delta to the pretrained
body (equivalent to loading the original lu-vae fine-tune) and evaluates with
the per-task primary metric.

This is a thin wrapper around the same eval machinery as
`eval_task_addition.py`, but it skips the merging step entirely — the body is
just (pretrained + delta_t) for the single task t.

Writes to: artifacts/results/roberta-base-ft-baselines/metrics.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.args import parse_arguments
from src.nlg.task_vectors import (
    ParamFolderTaskVector,
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)
from src.utils import resolve_run_dir

# Reuse the eval helpers from the merge script
from scripts.roberta.eval_task_addition import (
    GLUE_TASKS,
    GLUE_PRIMARY_METRIC,
    load_task_model,
    eval_task,
)


def build_task_body(args, task: str) -> dict:
    """Apply *just this task's* delta to the pretrained body → original FT body."""
    save = args.save
    tv = ParamFolderTaskVector(checkpoint_dir=f"{save}/{task}")

    pretrained_dir = Path(save) / "pretrained"
    pre_manifest = _load_manifest(pretrained_dir)
    body_sd = {}
    for key in tqdm(tv.lazy_keys(), desc=f"build {task}", leave=False):
        delta = tv.get_vector_element(key)
        pre_t = _load_single_tensor(
            _build_param_file_path(pretrained_dir, pre_manifest, key)
        )
        body_sd[key] = (pre_t.float() + delta).to(pre_t.dtype)
    return body_sd


def main():
    args = parse_arguments()
    args.save = resolve_run_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_datasets = args.eval_datasets or list(GLUE_TASKS)
    results_file = Path(f"{args.results_dir}/{args.model}-ft-baselines/metrics.json")
    if results_file.exists() and not args.overwrite:
        print(f"Skipping: {results_file} already exists (use --overwrite to rerun)")
        return

    print(f"Evaluating per-task FT baselines on {eval_datasets}")

    tasks = []
    primary_scores = []
    for task in eval_datasets:
        task_dir = Path(args.save) / task
        print(f"\n>>> {task}")
        body_sd = build_task_body(args, task)
        model = load_task_model(task_dir, body_sd, device)
        raw = eval_task(task, task_dir, model, args.batch_size, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        primary_key = GLUE_PRIMARY_METRIC[task]
        primary = float(raw[primary_key])
        primary_scores.append(primary)
        tasks.append(
            {
                "alias": task,
                "metrics": {**{k: float(v) for k, v in raw.items()}, "primary_score": primary},
                "task_config": {"primary_metric": primary_key},
            }
        )
        print(f"    {task}: {primary_key}={primary:.4f}")

    avg = float(np.mean(primary_scores))
    print(f"\nAverage FT baseline across {len(eval_datasets)} tasks: {avg:.4f}")

    metrics_json = {
        "all_primary_scores": [
            f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
        ],
        "average_primary_score": avg,
        "tasks": tasks,
        "model_config": {
            "model": args.model,
            "merge_func": "ft-baseline",
            "eval_datasets": eval_datasets,
            "seed": args.seed,
        },
    }
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(metrics_json, indent=2))
    print(f"Saved {results_file}")


if __name__ == "__main__":
    main()
