import json
import os
import os.path as osp
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.core.experts import Expert


class HFExpert(Expert):
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
        self._cov = None
        self._fish = None
        self.total_size = 0

        os.makedirs(model_dir, exist_ok=True)

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
            src = osp.join(chat_template_path, "chat_template.jinja")
            if osp.exists(src):
                shutil.copy2(src, osp.join(model_dir, "chat_template.jinja"))

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

    def _param_key_to_cov_key(self, layer_name):
        return layer_name.replace(".weight", "")

    def get_layers(self):
        return self.weight_map.keys()

    def get_layer_metadata(self, layer_name):
        return self.weight_map[layer_name]

    def get_layer_params(self, layer_name):
        shard_name = self.weight_map[layer_name]
        with safe_open(
            osp.join(self.model_dir, shard_name), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(layer_name)

    def get_layer_cov(self, layer_name):
        if self.cov is None:
            return None
        return self.cov.get(self._param_key_to_cov_key(layer_name))

    def get_layer_fish(self, layer_name):
        if self.fish is None:
            return None
        return self.fish.get(self._param_key_to_cov_key(layer_name))

    def save_layer_params(self, tensor, layer_name, metadata=None):
        shard_filename = metadata or "model.safetensors"
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
        with open(osp.join(self.model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": self.total_size},
                    "weight_map": self.weight_map,
                },
                f,
                indent=2,
            )

