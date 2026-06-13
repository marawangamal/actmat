import argparse
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from scripts.t5.common import (  # noqa: E402
    T5_DATASETS,
    evaluate_model,
    make_tasks,
    parse_csv,
    write_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experts-dir", required=True)
    parser.add_argument("--eval-datasets", type=parse_csv, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", default="finetuned.pt")
    parser.add_argument("--eval-split", default="test", choices=["test", "validation"])
    parser.add_argument("--data-location", default="data")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "huggingface"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_path = osp.join(args.output_dir, "metrics.json")
    if osp.exists(metrics_path) and not args.overwrite:
        print(f"Skipping: {metrics_path} already exists (use --overwrite to rerun)")
        raise SystemExit(0)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_datasets = args.eval_datasets or list(T5_DATASETS)
    scores = {}
    for dataset in eval_datasets:
        expert_dir = osp.join(args.experts_dir, dataset)
        model = torch.load(
            osp.join(expert_dir, args.checkpoint_name),
            map_location="cpu",
            weights_only=False,
        )
        dataset_scores = evaluate_model(model, [dataset], args)
        scores[f"{dataset}:top1"] = dataset_scores[f"{dataset}:top1"]

    scores["avg_top1"] = sum(scores[f"{d}:top1"] for d in eval_datasets) / len(
        eval_datasets
    )
    write_metrics(
        args.output_dir,
        make_tasks(scores),
        {
            "model": args.model,
            "checkpoint_name": args.checkpoint_name,
            "eval_split": args.eval_split,
            "avg_top1": scores["avg_top1"],
        },
    )
