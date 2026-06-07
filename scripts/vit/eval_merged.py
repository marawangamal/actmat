import argparse
import json
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from scripts.vit.common import (  # noqa: E402
    DEFAULT_VIT_DATASETS,
    eval_dataset_with_head,
    make_tasks,
    parse_csv,
    write_metrics,
)
from src.core.merge import merge_experts  # noqa: E402
from src.vit.experts import ViTExpert, optional_sidecar  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experts-dir", required=True)
    parser.add_argument("--eval-datasets", type=parse_csv, default=None)
    parser.add_argument("--merge-method", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", default="finetuned.pt")
    parser.add_argument("--covariance-name", default="covariance.pt")
    parser.add_argument("--fisher-name", default="fisher.pt")
    parser.add_argument("--ignore-keep-pt", default=None)
    parser.add_argument(
        "--merge-kwargs",
        type=json.loads,
        default={},
        help='JSON dict of extra kwargs forwarded to the merge function, e.g. \'{"angular_distance": 0.3}\'.',
    )
    parser.add_argument("--ignore-mean", default=None)
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
    base_expert_dir = osp.join(args.experts_dir, eval_datasets[0])
    base_model_path = osp.join(base_expert_dir, "pretrained.pt")

    base = ViTExpert(weights_path=base_model_path)
    experts = []
    for dataset in eval_datasets:
        expert_dir = osp.join(args.experts_dir, dataset)
        experts.append(
            ViTExpert(
                weights_path=osp.join(expert_dir, args.checkpoint_name),
                covariance_path=optional_sidecar(expert_dir, args.covariance_name),
                fisher_path=optional_sidecar(expert_dir, args.fisher_name),
            )
        )

    merged = ViTExpert()
    merge_experts(
        base,
        experts,
        merged,
        args.merge_method,
        ignore_keep_pt=args.ignore_keep_pt,
        ignore_mean=args.ignore_mean,
        merge_kwargs=args.merge_kwargs,
    )

    image_encoder = torch.load(base_model_path, map_location="cpu", weights_only=False)
    image_encoder.load_state_dict(merged.model_state_dict, strict=False)

    scores = {}
    for dataset in eval_datasets:
        head_path = osp.join(args.experts_dir, dataset, "head.pt")
        scores[f"{dataset}:top1"] = eval_dataset_with_head(
            image_encoder, dataset, head_path, args
        )["top1"]
    scores["avg_top1"] = sum(scores[f"{d}:top1"] for d in eval_datasets) / len(
        eval_datasets
    )

    write_metrics(
        args.output_dir,
        make_tasks(scores),
        {
            "model": args.model,
            "merge_method": args.merge_method,
            "merge_kwargs": args.merge_kwargs,
            "checkpoint_name": args.checkpoint_name,
            "avg_top1": scores["avg_top1"],
        },
    )
