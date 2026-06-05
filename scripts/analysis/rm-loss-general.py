"""Unified RegMean-loss analysis across checkpoint formats (compute only).

Two `Expert` classes over the two on-disk layouts in this repo, both loading a
layer's weights *efficiently* (no full-model instantiation, no per-call rescans):

  * BasicExpert -- pickled `.pt` checkpoint (vision/language). The `.pt` is read
                   ONCE into its state_dict and indexed per layer; `covariance.pt`
                   is a plain {layer: C} dict.
  * HFExpert    -- HF safetensors model (local HF dir or repo id). Weights are read
                   one layer at a time from the sharded safetensors via the index's
                   weight-map; `covariance.pt` is one big pickle, mmap'd.

The base model is just an Expert with no covariance. For every cov-tracked square
layer we form deltas d = w_t - w_0, build each method's merged weight w*, and score
the RegMean loss of (w* - w_0) against the REAL covariance. Writes per-(model,
layer, method) rows to artifacts/analysis/rm-loss/rm_loss_general.csv.
"""

import glob
import json
import os
import os.path as osp
import sys

import pandas as pd
import torch
from safetensors import safe_open
from tqdm import tqdm

rootdir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(rootdir)

from src.mhas import copy_from_pytorch_state_dict  # noqa: E402  (needs rootdir on path)

tr_abt = lambda a, b: (a * b).sum()
pinv = lambda c: torch.linalg.pinv(c, hermitian=True)  # c is symmetric PSD


# --------------------------------------------------------------------------- #
# loss + merge methods                                                        #
# --------------------------------------------------------------------------- #
def compute_rm_loss(w_test, w_t, c_t):
    """Compute the RegMean loss for linear layer at a given w

    Args:
        w_test: (Do, Di)      merged delta (w* - w_0)
        w_t:    (T, Do, Di)   per-expert deltas
        c_t:    (T, Di, Di)   per-expert (real) covariances
    """
    w_test = w_test.unsqueeze(0)
    loss_1 = tr_abt(w_test @ c_t, w_test)
    loss_2 = tr_abt(w_t @ c_t, w_t)
    loss_3 = tr_abt(w_t @ c_t, w_test)
    return loss_1 + loss_2 - 2 * loss_3


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # mean over experts of the per-expert cosine (each expert weighted equally,
    # not dominated by the largest-norm one). a, b: (T, Di, Di)
    a, b = a.flatten(1), b.flatten(1)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean()


def cosine_similarity_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # cosine over the whole flattened matrices (one scalar). a, b: (Di, Di)
    return (a.flatten() @ b.flatten()) / (a.norm() * b.norm())


def l2_error_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # mean over experts of the Frobenius distance ||a_i - b_i||_F. a, b: (T, Di, Di)
    return (a - b).flatten(1).norm(dim=1).mean()


def l2_error_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Frobenius distance ||a - b||_F over the whole matrices. a, b: (Di, Di)
    return (a - b).norm()


def rel_l2_error_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # mean over experts of ||a_i - b_i||_F / ||b_i||_F. a, b: (T, Di, Di)
    a, b = a.flatten(1), b.flatten(1)
    return ((a - b).norm(dim=1) / b.norm(dim=1)).mean()


def rel_l2_error_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ||a - b||_F / ||b||_F over the whole matrices. a, b: (Di, Di)
    return (a - b).norm() / b.norm()


def merge_actmat(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    c = d.transpose(-2, -1) @ d
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_actmat_identity_inv(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    T, _, _ = d.shape
    c = d.transpose(-2, -1) @ d
    return w0 + (((d @ c).sum(dim=0)) * 1 / T)


def merge_regmean(w0: torch.Tensor, d: torch.Tensor, c: torch.Tensor, **kwargs):
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_regmean_random_psd(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    # ablation: RegMean with a RANDOM PSD per expert (same shape as the real cov) --
    # tests whether the *specific* covariance matters or any PSD weighting would do.
    T, _, Di = d.shape
    a = torch.randn(T, Di, Di, device=d.device, dtype=d.dtype)
    c = a @ a.transpose(-2, -1)  # (T, Di, Di), symmetric PSD
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_mean(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    return w0 + d.mean(0)


merge_configs = [
    {
        "method_name": "actmat",
        "metrics": [
            {
                "metric_type": "rm_loss",
                "metric_func": lambda w0, d, c: compute_rm_loss(
                    merge_actmat(w0, d, c) - w0, d, c
                ),
            },
            {
                "metric_type": "cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_batch(
                    d.transpose(-2, -1) @ d, c
                ),
            },
            {  # cosine of the inverse-sums: (Σ ĉ)^+  vs  (Σ c)^+  -- what the merge actually inverts
                "metric_type": "inv_cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_full(
                    pinv((d.transpose(-2, -1) @ d).sum(0)), pinv(c.sum(0))
                ),
            },
            {  # raw Frobenius distances: ||ĉ - c||  and  ||(Σ ĉ)^+ - (Σ c)^+||
                "metric_type": "l2_error",
                "metric_func": lambda w0, d, c: l2_error_batch(d.transpose(-2, -1) @ d, c),
            },
            {
                "metric_type": "inv_l2_error",
                "metric_func": lambda w0, d, c: l2_error_full(
                    pinv((d.transpose(-2, -1) @ d).sum(0)), pinv(c.sum(0))
                ),
            },
            {  # relative Frobenius: ||ĉ - c|| / ||c||  and  ||(Σĉ)^+ - (Σc)^+|| / ||(Σc)^+||
                "metric_type": "rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_batch(d.transpose(-2, -1) @ d, c),
            },
            {
                "metric_type": "inv_rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_full(
                    pinv((d.transpose(-2, -1) @ d).sum(0)), pinv(c.sum(0))
                ),
            },
        ],
    },
    {
        "method_name": "actmat-identity-inv",
        "metrics": [
            {
                "metric_type": "rm_loss",
                "metric_func": lambda w0, d, c: compute_rm_loss(
                    merge_actmat_identity_inv(w0, d, c) - w0, d, c
                ),
            },
            {  # estimate = dᵀd (== actmat)
                "metric_type": "cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_batch(
                    d.transpose(-2, -1) @ d, c
                ),
            },
            {  # inverse it actually applies = (1/T) I (== identity); cosine is scale-free
                "metric_type": "inv_cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype), pinv(c.sum(0))
                ),
            },
            {  # estimate = dᵀd (== actmat); inverse it applies = (1/T) I
                "metric_type": "l2_error",
                "metric_func": lambda w0, d, c: l2_error_batch(d.transpose(-2, -1) @ d, c),
            },
            {
                "metric_type": "inv_l2_error",
                "metric_func": lambda w0, d, c: l2_error_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype) / c.size(0), pinv(c.sum(0))
                ),
            },
            {
                "metric_type": "rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_batch(d.transpose(-2, -1) @ d, c),
            },
            {
                "metric_type": "inv_rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype) / c.size(0), pinv(c.sum(0))
                ),
            },
        ],
    },
    {
        "method_name": "regmean",
        "metrics": [
            {
                "metric_type": "rm_loss",
                "metric_func": lambda w0, d, c: compute_rm_loss(
                    merge_regmean(w0, d, c) - w0, d, c
                ),
            },
            {  # estimate IS the true cov -> both ceilings == 1 (sanity reference)
                "metric_type": "cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_batch(c, c),
            },
            {
                "metric_type": "inv_cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_full(
                    pinv(c.sum(0)), pinv(c.sum(0))
                ),
            },
            {  # estimate IS the true cov -> both floors == 0
                "metric_type": "l2_error",
                "metric_func": lambda w0, d, c: l2_error_batch(c, c),
            },
            {
                "metric_type": "inv_l2_error",
                "metric_func": lambda w0, d, c: l2_error_full(pinv(c.sum(0)), pinv(c.sum(0))),
            },
            {
                "metric_type": "rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_batch(c, c),
            },
            {
                "metric_type": "inv_rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_full(pinv(c.sum(0)), pinv(c.sum(0))),
            },
        ],
    },
    {
        "method_name": "identity",
        "metrics": [
            {
                "metric_type": "rm_loss",
                "metric_func": lambda w0, d, c: compute_rm_loss(
                    merge_mean(w0, d, c) - w0, d, c
                ),
            },
            {
                "metric_type": "cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_batch(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype).expand(c.size(0), -1, -1),
                    c,
                ),
            },
            {  # (Σ ĉ)^+ = (1/T) I; cosine is scale-free, so compare I vs (Σ c)^+
                "metric_type": "inv_cosine_similarity",
                "metric_func": lambda w0, d, c: cosine_similarity_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype), pinv(c.sum(0))
                ),
            },
            {
                "metric_type": "l2_error",
                "metric_func": lambda w0, d, c: l2_error_batch(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype).expand(c.size(0), -1, -1), c
                ),
            },
            {  # inverse = (1/T) I
                "metric_type": "inv_l2_error",
                "metric_func": lambda w0, d, c: l2_error_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype) / c.size(0), pinv(c.sum(0))
                ),
            },
            {
                "metric_type": "rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_batch(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype).expand(c.size(0), -1, -1), c
                ),
            },
            {
                "metric_type": "inv_rel_l2_error",
                "metric_func": lambda w0, d, c: rel_l2_error_full(
                    torch.eye(c.size(-1), device=c.device, dtype=c.dtype) / c.size(0), pinv(c.sum(0))
                ),
            },
        ],
    },
]


# --------------------------------------------------------------------------- #
# Experts                                                                     #
# --------------------------------------------------------------------------- #
def _is_square(v):
    return torch.is_tensor(v) and v.ndim == 2 and v.size(0) == v.size(1)


class Expert:
    def get_layers(self):
        raise NotImplementedError

    def get_layer_params(self, layer):
        raise NotImplementedError

    def get_layer_cov(self, layer):
        raise NotImplementedError


class BasicExpert(Expert):
    """Pickled `.pt` checkpoint (+ optional plain {layer: C} covariance.pt).

    The `.pt` is unpickled once to its state_dict; layers are then dict lookups.
    """

    def __init__(self, weights_pt, cov_pt=None):
        self.weights_pt = weights_pt
        self.cov_pt = cov_pt
        self._sd = None
        self._cov = None

    @property
    def sd(self):
        if self._sd is None:
            obj = torch.load(self.weights_pt, weights_only=False, map_location="cpu")
            self._sd = obj if isinstance(obj, dict) else obj.state_dict()
        return self._sd

    @property
    def cov(self):
        if self._cov is None and self.cov_pt is not None:
            self._cov = torch.load(self.cov_pt, map_location="cpu")
        return self._cov

    def _param_key_to_cov_key(self, layer):
        return layer.replace(".weight", "")

    def get_layers(self):
        return self.sd.keys()

    def get_layer_params(self, layer):
        return self.sd[layer]

    def get_layer_cov(self, layer):
        return self.cov.get(self._param_key_to_cov_key(layer))


class ViTExpert(BasicExpert):
    """OpenCLIP ViT `.pt` checkpoint. Covariance was collected on the split-QKV MHA
    (src/mhas.py), so the packed in_proj_weight/out_proj are remapped to q/k/v/o.weight
    to match; cov keys also carry an 'image_encoder.' prefix.
    """

    @property
    def sd(self):
        # split packed MHA weights -> q/k/v/o.weight so keys line up with the split-MHA cov
        if self._sd is None:
            self._sd = copy_from_pytorch_state_dict(super().sd)
        return self._sd

    def _param_key_to_cov_key(self, layer):
        return "image_encoder." + layer.replace(".weight", "")


class HFExpert(Expert):
    """HF safetensors model -- local HF dir or repo id (newest cache snapshot).

    Weights are read one layer at a time from the sharded safetensors; `layer` keys
    are exactly the model.safetensors.index.json weight-map keys. Optional mmap'd
    covariance.pt is indexed by the cov key (param key minus ".weight").
    """

    def __init__(self, model_name_or_path, covariance_path=None):
        self.covariance_path = covariance_path
        self._cov = None

        if osp.isdir(model_name_or_path):  # local HF dir
            self.rootdir = model_name_or_path
        else:  # bare HF repo id -> newest cache snapshot
            hub = osp.join(
                os.environ.get("HF_HOME", osp.expanduser("~/.cache/huggingface")), "hub"
            )
            snaps = sorted(
                glob.glob(
                    osp.join(
                        hub,
                        "models--" + model_name_or_path.replace("/", "--"),
                        "snapshots",
                        "*",
                    )
                )
            )
            if not snaps:
                raise FileNotFoundError(
                    f"No HF snapshot for {model_name_or_path} under {hub}"
                )
            self.rootdir = snaps[-1]

        index = json.load(open(osp.join(self.rootdir, "model.safetensors.index.json")))
        self.weight_map = index["weight_map"]  # param_key -> shard filename

    @property
    def cov(self):
        if self._cov is None and self.covariance_path is not None:
            self._cov = torch.load(
                self.covariance_path, map_location="cpu", mmap=True, weights_only=True
            )
        return self._cov

    def _param_key_to_cov_key(self, layer):
        return layer.replace(".weight", "")

    def get_layers(self):
        return self.weight_map.keys()

    def get_layer_params(self, layer):
        path = osp.join(self.rootdir, self.weight_map[layer])
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(layer)

    def get_layer_cov(self, layer):
        return self.cov.get(self._param_key_to_cov_key(layer))


# --------------------------------------------------------------------------- #
rows = []
configs = [
    {
        "model": "t5-base",
        "type": "basic",
        "base": "artifacts/checkpoints/t5-base/pretrained.pt",
        "experts-path": "artifacts/checkpoints/t5-base/group-main/experts",
    },
    {
        "model": "t5-large",
        "type": "basic",
        "base": "artifacts/checkpoints/t5-large/pretrained.pt",
        "experts-path": "artifacts/checkpoints/t5-large/group-main/experts",
    },
    {
        "model": "ViT-B-16",
        "type": "vit",
        "base": "artifacts/checkpoints/ViT-B-16/pretrained.pt",
        "experts-path": "artifacts/checkpoints/ViT-B-16/group-8/experts",
    },
    {
        "model": "ViT-B-32",
        "type": "vit",
        "base": "artifacts/checkpoints/ViT-B-32/pretrained.pt",
        "experts-path": "artifacts/checkpoints/ViT-B-32/group-8/experts",
    },
    {
        "model": "ViT-L-14",
        "type": "vit",
        "base": "artifacts/checkpoints/ViT-L-14/pretrained.pt",
        "experts-path": "artifacts/checkpoints/ViT-L-14/group-8/experts",
    },
    {
        "model": "Olmo-3-7b",
        "type": "hf",
        "base": "allenai/Olmo-3-1025-7B",
        "experts-path": "artifacts/checkpoints/Olmo-3-7b/group-rl-zero/experts",
        "expert_to_model_name_or_path": {  # expert dir name -> HF model_name_or_path (weights); cov stays local
            "Code": "allenai/Olmo-3-7B-RL-Zero-Code",
            "IF": "allenai/Olmo-3-7B-RL-Zero-IF",
            "Math": "allenai/Olmo-3-7B-RL-Zero-Math",
        },
    },
]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}")

for cfg in configs:
    model = cfg["model"]
    edir = osp.join(rootdir, cfg["experts-path"])
    expert_dirs = sorted(
        osp.join(edir, n)
        for n in os.listdir(edir)
        if osp.exists(osp.join(edir, n, "covariance.pt"))
    )

    if cfg["type"] == "hf":
        base = HFExpert(cfg["base"])
        experts = [
            HFExpert(
                cfg["expert_to_model_name_or_path"][osp.basename(d)],
                osp.join(d, "covariance.pt"),
            )
            for d in expert_dirs
        ]
    else:  # basic / vit -- pickled .pt (ViTExpert only differs in the cov-key mapping)
        cls = ViTExpert if cfg["type"] == "vit" else BasicExpert
        base = cls(osp.join(rootdir, cfg["base"]))
        experts = [
            cls(osp.join(d, "finetuned.pt"), osp.join(d, "covariance.pt"))
            for d in expert_dirs
        ]

    print(
        f"\n=== {model}: {len(experts)} experts ({', '.join(osp.basename(d) for d in expert_dirs)}) ==="
    )
    # cov-tracked layers come from the experts (the base has no covariance)
    # layers = sorted(set.intersection(*(e.get_layers() for e in experts)))
    # layers = [l for l in layers if base.has(l)]
    layers = base.get_layers()
    layer_idx = 0
    for l in tqdm(layers, desc=model):
        w_0 = base.get_layer_params(l).float().to(dev)  # (Do, Di)
        if not w_0.ndim == 2 or experts[0].get_layer_cov(l) is None:
            continue
        w_list = []
        c_list = []
        for expert in experts:
            w_t = expert.get_layer_params(l).float().to(dev)
            c_t = expert.get_layer_cov(l).float().to(dev)
            w_list.append(w_t)
            c_list.append(c_t)
        d = torch.stack(w_list) - w_0  # (T, Do, Di)
        c = torch.stack(c_list)  # (T, Di, Di)
        for cfg_m in merge_configs:
            for metric in cfg_m["metrics"]:
                rows.append(
                    {
                        "model": model,
                        "method": cfg_m["method_name"],
                        "layer": l,
                        "layer_idx": layer_idx,
                        "metric_type": metric["metric_type"],
                        "metric": metric["metric_func"](w_0, d, c).item(),
                        "Di": w_0.shape[-1],
                        "Do": w_0.shape[-2],
                    }
                )
        layer_idx += 1
        del w_0, d, c

out = osp.join(rootdir, "artifacts/analysis/rm-loss/rm_loss_general.csv")
os.makedirs(osp.dirname(out), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(f"\nwrote {len(df)} rows -> {out}")
