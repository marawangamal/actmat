"""Unified RegMean-loss analysis across checkpoint formats (compute only).

Two `Expert` classes over the two on-disk layouts in this repo, both loading a
layer's weights *efficiently* (no full-model instantiation, no per-call rescans):

  * BasicExpert -- pickled `.pt` checkpoint (vision/language). The `.pt` is read
                   ONCE into its state_dict and indexed per layer; `covariance.pt`
                   is a plain {layer: C} dict.
  * HFExpert    -- safetensors, per-layer lazy reads. Works over an OLMo param-folder
                   (one file per weight) or a bare HF repo id (cache-snapshot shards,
                   used for the base). `covariance.pt` is one big pickle, mmap'd.

The base model is just an Expert with no covariance. For every cov-tracked square
layer we form deltas d = w_t - w_0, build each method's merged weight w*, and score
the RegMean loss of (w* - w_0) against the REAL covariance. Writes per-(model,
layer, method) rows to artifacts/analysis/rm-loss/rm_loss_general.json.
"""

import glob
import json
import os
import os.path as osp
import sys

import torch
from safetensors import safe_open
from tqdm import tqdm

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(REPO_ROOT)

tr_abt = lambda a, b: (a * b).sum()
pinv = lambda c: torch.linalg.pinv(c, hermitian=True)  # c is symmetric PSD


# --------------------------------------------------------------------------- #
# loss + merges  (operate on deltas d = w_t - w_0; merges return full w*)      #
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


def merge_actmat(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    c = d.transpose(-2, -1) @ d
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_regmean(w0: torch.Tensor, d: torch.Tensor, c: torch.Tensor, **kwargs):
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_mean(w0: torch.Tensor, d: torch.Tensor, c=None, **kwargs):
    return w0 + d.mean(0)


merge_configs = [
    ("actmat", merge_actmat),
    ("regmean", merge_regmean),
    ("identity", merge_mean),
]


# --------------------------------------------------------------------------- #
# experts — `layer` is the cov key == param name minus ".weight".             #
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
        return self.cov[layer]


class HFExpert(Expert):
    """safetensors weights, read one layer at a time (+ optional mmap'd covariance.pt)."""

    def __init__(self, model_name_or_path, covariance_path=None):
        self.covariance_path = covariance_path
        self._cov = None
        self.rootdir = None
        self.model_index = {}  # real_key -> (file_path, internal_key)

        # bare HF repo id -> newest cache snapshot
        hub = osp.join(os.environ.get("HF_HOME"), "hub")
        snaps = sorted(
            glob.glob(
                osp.join(
                    hub,
                    "models--" + weights_src.replace("/", "--"),
                    "snapshots",
                    "*",
                )
            )
        )
        if not snaps:
            raise FileNotFoundError(f"No HF snapshot for {weights_src} under {hub}")

        self.rootdir = open(osp.join(snaps[-1], "model.safetensors.index.json"))
        self.weight_map = json.load(self.rootdir)

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
        return self.sd.keys()

    def get_layer_params(self, layer):
        path = osp.join(self.rootdir, self.weight_map, layer)
        res = safe_open(path, framework="pt", device="cpu")
        return res.get_tensor(internal_key)

    def get_layer_cov(self, layer):
        return self.cov[_param_key_to_cov_key(layer)]


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
        "model": "Olmo-3-7b",
        "type": "hf",
        "base": "allenai/Olmo-3-1025-7B",
        "experts-path": "artifacts/checkpoints/Olmo-3-7b/group-rl-zero/experts",
    },
]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}")

for cfg in configs:
    model = cfg["model"]
    edir = osp.join(REPO_ROOT, cfg["experts-path"])
    expert_dirs = sorted(
        osp.join(edir, n)
        for n in os.listdir(edir)
        if osp.exists(osp.join(edir, n, "covariance.pt"))
    )

    if cfg["type"] == "basic":
        base = BasicExpert(osp.join(REPO_ROOT, cfg["base"]))
        experts = [
            BasicExpert(osp.join(d, "finetuned.pt"), osp.join(d, "covariance.pt"))
            for d in expert_dirs
        ]
    else:
        base = HFExpert(cfg["base"])
        experts = [
            HFExpert(osp.join(d, "finetuned"), osp.join(d, "covariance.pt"))
            for d in expert_dirs
        ]

    print(
        f"\n=== {model}: {len(experts)} experts ({', '.join(osp.basename(d) for d in expert_dirs)}) ==="
    )
    # cov-tracked layers come from the experts (the base has no covariance)
    # layers = sorted(set.intersection(*(e.get_layers() for e in experts)))
    # layers = [l for l in layers if base.has(l)]
    layers = base.get_layers()

    for l in tqdm(layers, desc=model):
        w_0 = base.get_layer_params(l).float().to(dev)  # (Do, Di)
        if not w_0.ndim == 2:
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
        for merge_method, merge_func in merge_configs:
            w_star = merge_func(w_0, d, c)
            loss = compute_rm_loss(w_star - w_0, d, c)
            rows.append(
                {
                    "model": model,
                    "method": merge_method,
                    "layer": l,
                    "metric_type": "rm_loss",
                    "metric": loss.item(),
                    "Di": w_0.shape[-1],
                    "Do": w_0.shape[-2],
                }
            )
        del w_0, d, c

out = osp.join(REPO_ROOT, "artifacts/analysis/rm-loss/rm_loss_general.json")
os.makedirs(osp.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote {len(rows)} rows -> {out}")
