import argparse
import json
import os
import os.path as osp
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src import mergingv2
from src.utils import sanitize_hf_id


class HFExpert:
    def __init__(
        self,
        model_dir,
        chat_template_path=None,
        tokenizer_path=None,
        covariance_path=None,
        fisher_path=None,
    ):
        self.model_dir = model_dir
        self.covariance_path = covariance_path
        self.fisher_path = fisher_path
        self.chat_template_path = chat_template_path
        self.tokenizer_path = tokenizer_path
        self._cov = None
        self._fish = None

        # create dir if not exists
        os.makedirs(model_dir, exist_ok=True)

        # set weight map
        index_path = osp.join(model_dir, "model.safetensors.index.json")
        single_path = osp.join(model_dir, "model.safetensors")
        if osp.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        elif osp.exists(single_path):
            with safe_open(single_path, framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        else:
            self.weight_map = {}
        self.total_size = 0

        # copy tokenizer files + config, then chat template
        if tokenizer_path is not None:
            tokenizer_files = (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            )
            for name in tokenizer_files:
                src = osp.join(tokenizer_path, name)
                if osp.isfile(src):
                    shutil.copy2(src, osp.join(model_dir, name))
        if chat_template_path is not None:
            chat_template = osp.join(chat_template_path, "chat_template.jinja")
            if osp.exists(chat_template):
                shutil.copy2(chat_template, osp.join(model_dir, "chat_template.jinja"))

    @property
    def cov(self):
        if self._cov is None and self.covariance_path is not None:
            self._cov = torch.load(
                self.covariance_path, map_location="cpu", mmap=True, weights_only=True
            )
        return self._cov

    @property
    def fish(self):
        if self._fish is None and self.fisher_path is not None:
            self._fish = torch.load(
                self.fisher_path, map_location="cpu", mmap=True, weights_only=True
            )
        return self._fish

    def _param_key_to_cov_key(self, layer):
        return layer.replace(".weight", "")

    def get_layer_params(self, layer_name):
        # loads layer from hf dir with pottentially many shards
        shard_name = self.weight_map[layer_name]
        with safe_open(
            osp.join(self.model_dir, shard_name), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(layer_name)

    def make_stat_fetcher_map(self, layer_name):
        return {
            "covariance": lambda: self.get_layer_cov(layer_name),
            "fisher": lambda: self.get_layer_fish(layer_name),
        }

    def get_layers(self):
        return self.weight_map.keys()

    def get_layer_cov(self, layer):
        if self.cov is None:
            return None
        return self.cov.get(self._param_key_to_cov_key(layer))

    def get_layer_fish(self, layer):
        if self.fish is None:
            return None
        return self.fish.get(self._param_key_to_cov_key(layer))

    def save_layer_params(self, tensor, shard_filename, layer_name):
        # saves into hf compatible safetensors layer_file at key layer_name
        shard_path = osp.join(self.model_dir, shard_filename)
        tensors = {}
        if osp.exists(shard_path):
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}
        tensors[layer_name] = tensor.contiguous()
        save_file(tensors, shard_path, metadata={"format": "pt"})
        self.weight_map[layer_name] = shard_filename
        self.total_size += tensor.numel() * tensor.element_size()

    def flush(self):
        # saves index
        index_path = osp.join(self.model_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": self.total_size},
                    "weight_map": self.weight_map,
                },
                f,
                indent=2,
            )


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
    base_hf_dir = HFExpert(base_model_path)
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
            )
        )

    layer_to_file = base_hf_dir.weight_map
    merged_model_hf_dir = HFExpert(
        args.output_dir, chat_template_path, chat_template_path
    )
    for layer_name in layer_to_file:
        shard_filename = layer_to_file[layer_name]
        w_list = []
        stat_fetcher_maps = []
        w_0 = base_hf_dir.get_layer_params(layer_name)

        if args.ignore_keep_pt and re.search(args.ignore_keep_pt, layer_name):
            w_merged = w_0
        else:
            for expert_hf_dir in expert_hf_dirs:
                w_t = expert_hf_dir.get_layer_params(layer_name)
                w_list.append(w_t)
                stat_fetcher_maps.append(
                    expert_hf_dir.make_stat_fetcher_map(layer_name)
                )

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

        merged_model_hf_dir.save_layer_params(w_merged, shard_filename, layer_name)

    # save index
    merged_model_hf_dir.flush()  # saves index
