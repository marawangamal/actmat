import argparse
import json
import os
import os.path as osp
import re

from safetensors import safe_open
from safetensors.torch import save_file


class HFDir:
    def __init__(model_dir, chat_template_path, tokenizer_path):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        index_path = osp.join(self.model_dir, "model.safetensors.index.json")

        # 1. make the index if not exists
        if not osp.exists(index_path):
            # make the index
            pass

        # 2. copy over tok, chat etc. from chat_template_path, tokenizer_path
        # ...

    def load_layer_params(layer_name):
        # loads layer from hf dir with pottentially many shards
        index_path = osp.join(self.model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_name = weight_map[layer_name]
        # Finish this
        # tensor = ...
        return tensor

    def save_layer_params(self, tensor, shard_filename, layer_name):
        # saves into hf compatible safetensors layer_file at key layer_name
        shard_path = osp.join(self.model_dir, shard_filename)
        tensors = {}
        if osp.exists(shard_path):
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}
        tensors[layer_name] = tensor.contiguous()
        save_file(tensors, shard_path, metadata={"format": "pt"})


def load_layer_index(model_dir):
    # maps each layer name -> its shard filename (sharded or single-file)
    index_path = osp.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def fmt(inp):
    return inp.rstrip("/").replace("/", "_")


def parse_args():
    parser = argparse.ArgumentParser()
    # hf ids or local paths can be used
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--expert-models", nargs="+", required=True)
    parser.add_argument("--merge-method", default="sum")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--ignore-keep-pt", default=None)
    parser.add_argument("--ignore-mean", default=None)
    return parser.parse_args()


args = parse_args()

layer_to_file = load_layer_index(args.base_model)
base_model_path = osp.join(hf_home, fmt(args.base_model))
chat_template_path = osp.join(hf_home, fmt(args.chat_template))
merged_model_path = osp.join(args.save_dir, fmt(args.base_model), args.merge_method)
merged_model_hf_dir = HFDir(merged_model_path, chat_template_path, chat_template_path)
for layer_name in layer_to_file:
    shard_filename = layer_to_file[layer_name]
    w_list = []
    w_0 = load_layer_params(args.base_model, layer_name)

    if args.ignore_keep_pt and re.search(args.ignore_keep_pt, layer_name):
        w_merged = w_0
    else:
        for model_name_or_path in args.expert_models:
            w_t = merged_model_hf_dir.load_layer_params(layer_name)
            w_list.append(wt)

        if args.ignore_mean and re.search(args.ignore_mean, layer_name):
            w_merged = apply_merge(w_list, w_0, method="mean")
        else:
            w_merged = apply_merge(w_list, w_0, method=args.merge_method)

    merged_model_hf_dir.save_layer_params(w_merged, shard_filename, layer_name)
    # add to index
    merged_model_index[layer_name] = shard_filename

# save index
merged_model_hf_dir.flush()  # saves index
