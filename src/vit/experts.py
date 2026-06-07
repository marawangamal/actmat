import os
import os.path as osp

import torch

from src import mhas
from src.core.experts import Expert


class ViTExpert(Expert):
    def __init__(self, weights_path=None, covariance_path=None, fisher_path=None):
        self.weights_path = weights_path
        self.covariance_path = covariance_path
        self.fisher_path = fisher_path
        self._cov = None
        self._fish = None

        if weights_path is None:
            self.state_dict = {}
        else:
            obj = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = obj.state_dict() if hasattr(obj, "state_dict") else obj
            self.state_dict = mhas.copy_from_pytorch_state_dict(state_dict)

    @property
    def cov(self):
        if self._cov is None and self.covariance_path is not None:
            self._cov = torch.load(
                self.covariance_path, map_location="cpu", weights_only=False
            )
        return self._cov

    @property
    def fish(self):
        if self._fish is None and self.fisher_path is not None:
            self._fish = torch.load(
                self.fisher_path, map_location="cpu", weights_only=False
            )
        return self._fish

    @property
    def model_state_dict(self):
        return mhas.copy_to_pytorch_state_dict(self.state_dict)

    def _param_key_to_cov_key(self, layer_name):
        return "image_encoder." + layer_name.replace(".weight", "")

    def get_layers(self):
        return self.state_dict.keys()

    def get_layer_params(self, layer_name):
        return self.state_dict[layer_name]

    def get_layer_cov(self, layer_name):
        if self.cov is None:
            return None
        return self.cov.get(self._param_key_to_cov_key(layer_name))

    def get_layer_fish(self, layer_name):
        if self.fish is None:
            return None
        return self.fish.get(self._param_key_to_cov_key(layer_name))

    def save_layer_params(self, tensor, layer_name, metadata=None):
        self.state_dict[layer_name] = tensor.contiguous()


def optional_sidecar(expert_dir, filename):
    path = osp.join(expert_dir, filename)
    return path if os.path.exists(path) else None
