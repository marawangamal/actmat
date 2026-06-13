import argparse
import json
import os.path as osp
import re

import torch

from src import mergingv2
from src.hf2.experts import HFExpert
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
    for layer_name in base_hf_dir.get_layers():
        metadata = base_hf_dir.get_layer_metadata(layer_name)
        w_list = []
        stat_fetcher_maps = []
        w_0 = base_hf_dir.get_layer_params(layer_name)

        if args.ignore_keep_pt and re.search(args.ignore_keep_pt, layer_name):
            w_merged = w_0
        else:
            for expert_hf_dir in expert_hf_dirs:
                w_t = expert_hf_dir.get_layer_params(layer_name)
                w_list.append(w_t)
                stat_fetcher_maps.append(expert_hf_dir.get_stat_fetcher_map(layer_name))

            if w_0.ndim != 2 or (
                args.ignore_mean and re.search(args.ignore_mean, layer_name)
            ):
                print(
                    f"[IGNORE-MEAN] forcing mean merge for layer: {layer_name}",
                    flush=True,
                )
                # fallback to mean merging
                w_merged = torch.stack(w_list).mean(0)
            else:
                # ** merge ***
                w0 = w_0.to(_MERGE_DEVICE).float()
                d = torch.stack([w.to(_MERGE_DEVICE).float() - w0 for w in w_list])
                merge_fn = getattr(mergingv2, "merge_" + args.merge_method)
                merged_delta = merge_fn(d=d, stat_fetcher_maps=stat_fetcher_maps)
                w_merged = (w0 + merged_delta).to(w_0.dtype).cpu()

        merged_model_hf_dir.save_layer_params(w_merged, layer_name, metadata=metadata)

    # save index
    merged_model_hf_dir.flush()  # saves index
