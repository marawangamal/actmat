import argparse
import json
import os
import os.path as osp
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src import merging


class HFDir:
    def __init__(self, model_dir, chat_template_path=None, tokenizer_path=None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        # maps each layer name -> its shard filename (sharded or single-file)
        index_path = osp.join(model_dir, "model.safetensors.index.json")
        if osp.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
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

    def load_layer_params(self, layer_name):
        # loads layer from hf dir with pottentially many shards
        shard_name = self.weight_map[layer_name]
        with safe_open(
            osp.join(self.model_dir, shard_name), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(layer_name)

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


def apply_merge(w_list, w_0, method="sum"):
    deltas = torch.stack([w.float() - w_0.float() for w in w_list])
    if deltas[0].ndim == 2:
        merged_delta = getattr(merging, "merge_" + method)(deltas)
    else:
        merged_delta = deltas.mean(dim=0)
    return (w_0.float() + merged_delta).to(w_0.dtype)


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
    return parser.parse_args()


args = parse_args()

base_model_path = resolve(args.base_model_name_or_path)
chat_template_path = resolve(args.chat_template_name_or_path)

base_hf_dir = HFDir(base_model_path)
expert_hf_dirs = [HFDir(resolve(m)) for m in args.expert_model_names_or_paths]

layer_to_file = base_hf_dir.weight_map
merged_model_hf_dir = HFDir(args.output_dir, chat_template_path, chat_template_path)
for layer_name in layer_to_file:
    shard_filename = layer_to_file[layer_name]
    w_list = []
    w_0 = base_hf_dir.load_layer_params(layer_name)

    if args.ignore_keep_pt and re.search(args.ignore_keep_pt, layer_name):
        w_merged = w_0
    else:
        for expert_hf_dir in expert_hf_dirs:
            w_t = expert_hf_dir.load_layer_params(layer_name)
            w_list.append(w_t)

        if args.ignore_mean and re.search(args.ignore_mean, layer_name):
            w_merged = apply_merge(w_list, w_0, method="mean")
        else:
            w_merged = apply_merge(w_list, w_0, method=args.merge_method)

    merged_model_hf_dir.save_layer_params(w_merged, shard_filename, layer_name)

# save index
merged_model_hf_dir.flush()  # saves index
