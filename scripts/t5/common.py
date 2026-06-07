import json
import os
import os.path as osp

import numpy as np

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]


def parse_csv(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def make_tasks(scores):
    return [
        {
            "alias": name,
            "metrics": {"top1": score, "primary_score": score},
            "task_config": {"primary_metric": "top1"},
        }
        for name, score in scores.items()
        if isinstance(score, (int, float, np.floating))
    ]


def write_metrics(output_dir, tasks, model_config):
    os.makedirs(output_dir, exist_ok=True)
    metrics_json = {
        "all_primary_scores": [
            f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
        ],
        "tasks": tasks,
        "model_config": model_config,
    }
    metrics_path = osp.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Results saved to {metrics_path}")


def evaluate_model(model, eval_datasets, args):
    from src.language.eval import eval_single_dataset

    scores = {}
    for dataset in eval_datasets:
        scores[f"{dataset}:top1"] = eval_single_dataset(
            args.eval_split, model, model.tokenizer, dataset, args
        )["top1"]
    scores["avg_top1"] = sum(scores[f"{d}:top1"] for d in eval_datasets) / len(
        eval_datasets
    )
    return scores
