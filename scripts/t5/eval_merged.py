import argparse
import json
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
from src.core.merge import merge_experts  # noqa: E402
from src.t5.experts import T5Expert, optional_sidecar  # noqa: E402


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
    parser.add_argument("--ignore-mean", default=None)
    parser.add_argument(
        "--merge-kwargs",
        type=json.loads,
        default={},
        help='JSON dict of extra kwargs forwarded to the merge function.',
    )
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
    base_expert_dir = osp.join(args.experts_dir, eval_datasets[0])
    base_model_path = osp.join(base_expert_dir, "pretrained.pt")

    base = T5Expert(weights_path=base_model_path)
    experts = []
    for dataset in eval_datasets:
        expert_dir = osp.join(args.experts_dir, dataset)
        experts.append(
            T5Expert(
                weights_path=osp.join(expert_dir, args.checkpoint_name),
                covariance_path=optional_sidecar(expert_dir, args.covariance_name),
                fisher_path=optional_sidecar(expert_dir, args.fisher_name),
            )
        )

    merged = T5Expert()
    merge_experts(
        base,
        experts,
        merged,
        args.merge_method,
        ignore_keep_pt=args.ignore_keep_pt,
        ignore_mean=args.ignore_mean,
        merge_kwargs=args.merge_kwargs,
    )

    model = torch.load(base_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(merged.model_state_dict, strict=False)

    scores = evaluate_model(model, eval_datasets, args)
    write_metrics(
        args.output_dir,
        make_tasks(scores),
        {
            "model": args.model,
            "merge_method": args.merge_method,
            "merge_kwargs": args.merge_kwargs,
            "checkpoint_name": args.checkpoint_name,
            "eval_split": args.eval_split,
            "avg_top1": scores["avg_top1"],
        },
    )
