import argparse
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from scripts.vit.common import (  # noqa: E402
    DEFAULT_VIT_DATASETS,
    eval_dataset_with_head,
    make_eval_namespace,
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
    parser.add_argument("--data-location", default="data/vision")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "models"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_path = osp.join(args.output_dir, "metrics.json")
    if osp.exists(metrics_path) and not args.overwrite:
        print(f"Skipping: {metrics_path} already exists (use --overwrite to rerun)")
        raise SystemExit(0)

    eval_datasets = args.eval_datasets or list(DEFAULT_VIT_DATASETS)
    eval_args = make_eval_namespace(args)
    scores = {}
    for dataset in eval_datasets:
        expert_dir = osp.join(args.experts_dir, dataset)
        image_encoder = torch.load(
            osp.join(expert_dir, args.checkpoint_name),
            map_location="cpu",
            weights_only=False,
        )
        head_path = osp.join(expert_dir, "head.pt")
        scores[f"{dataset}:top1"] = eval_dataset_with_head(
            image_encoder, dataset, head_path, eval_args
        )["top1"]

    scores["avg_top1"] = sum(scores[f"{d}:top1"] for d in eval_datasets) / len(eval_datasets)
    write_metrics(
        args.output_dir,
        make_tasks(scores),
        {
            "model": args.model,
            "checkpoint_name": args.checkpoint_name,
            "avg_top1": scores["avg_top1"],
        },
    )
