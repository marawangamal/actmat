import argparse
import json
import os.path as osp

import torch

from src.core.merge import merge_experts
from src.hf.experts import HFExpert
from src.utils import sanitize_hf_id


def resolve(name_or_path):
    # hf id or local path -> local dir
    if osp.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # hf ids or local paths can be used
    parser.add_argument("--base-model-name-or-path", required=True)
    parser.add_argument("--chat-template-name-or-path", required=True)
    parser.add_argument("--expert-model-names-or-paths", nargs="+", required=True)
    parser.add_argument("--merge-method", default="sum")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ignore-keep-pt", default=None)
    parser.add_argument("--ignore-mean", default=None)
    parser.add_argument(
        "--expert-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of extra kwargs forwarded to each HFExpert.",
    )
    # Covariance-based methods (e.g. regmean) read per-expert stats sidecars from
    # <expert-stats-dir>/<expert-id>/{covariance,fisher}.pt (expert-id = the model
    # basename).
    parser.add_argument("--expert-stats-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    _MERGE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    base_model_path = resolve(args.base_model_name_or_path)
    chat_template_path = resolve(args.chat_template_name_or_path)

    # build experts list
    base_hf_dir = HFExpert(base_model_path, **args.expert_kwargs)
    expert_hf_dirs = []
    for m in args.expert_model_names_or_paths:
        covariance_path = None
        fisher_path = None
        if args.expert_stats_dir is not None:
            expert_stats_path = osp.join(args.expert_stats_dir, sanitize_hf_id(m))
            covariance_path = osp.join(expert_stats_path, "covariance.pt")
            fisher_path = osp.join(expert_stats_path, "fisher.pt")
        expert_hf_dirs.append(
            HFExpert(
                resolve(m),
                covariance_path=covariance_path,
                fisher_path=fisher_path,
                **args.expert_kwargs,
            )
        )

    merged_model_hf_dir = HFExpert(
        args.output_dir,
        chat_template_path,
        chat_template_path,
        **args.expert_kwargs,
    )

    merge_experts(
        base_hf_dir,
        expert_hf_dirs,
        merged_model_hf_dir,
        args.merge_method,
        ignore_keep_pt=args.ignore_keep_pt,
        ignore_mean=args.ignore_mean,
        device=_MERGE_DEVICE,
    )
