"""RegMean-loss analysis for Olmo-3-7b (streaming, low-memory).

Same math as scripts/analysis/rm-loss.py, but adapted to the OLMo artifact
layout, which makes the t5/ViT "cache-everything" fast variant infeasible
(3 experts x 28.5GB covariance = ~85GB just for covs):

  * covariance.pt is a single ~28.5GB monolithic pickle per expert  -> open ONCE
    with mmap=True (peak RSS ~0.6GB) and index per-layer inside the loop.
  * finetuned/ is a param-folder (one safetensors file per weight) -> load the
    single layer weight on demand; no need to keep full models in RAM.
  * the model-level pretrained/ folder is gone (dangling symlink), so w_0 comes
    from the HF base model allenai/Olmo-3-1025-7B (sharded safetensors in cache).

Peak RAM is bounded by the largest layer (mlp.down_proj, 11008x11008) across the
3 experts plus a few fp32 working copies -> ~6-10GB. A 32GB allocation is plenty.
"""

import json
import sys
from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

tr_abt = lambda a, b: (a * b).sum()


def compute_rm_loss_v2(w, w_t, c_t):
    """RegMean loss at delta `w` given per-expert deltas `w_t` and covs `c_t`.

    Args:
        w:   (Do, Di)        merged delta (w_star - w_0)
        w_t: (T, Do, Di)     per-expert deltas
        c_t: (T, Di, Di)     per-expert (real) covariances
    """
    w_test = w.unsqueeze(0)
    loss_1 = tr_abt(w_test @ c_t, w_test)
    loss_2 = tr_abt(w_t @ c_t, w_t)
    loss_3 = tr_abt(w_t @ c_t, w_test)
    return loss_1 + loss_2 - 2 * loss_3


def compute_rm_minimizer(w0, d, c):
    # c.sum(0) is symmetric PSD (sum of covariances / of dᵀd) -> eigh-based pinv:
    # faster than the default SVD path and numerically apt for Hermitian inputs.
    return w0 + ((d @ c).sum(dim=0) @ torch.linalg.pinv(c.sum(dim=0), hermitian=True))


def identity_minimizer(w0, d):
    # c_t == I  =>  pinv(sum_t I) = (1/T) I  =>  w* = w0 + mean_t d  (no SVD)
    return w0 + d.mean(dim=0)


# --- base-model (w_0) lazy reader: HF cache, sharded safetensors -------------
BASE_REPO = "allenai/Olmo-3-1025-7B"


def _base_reader():
    hub = Path(__import__("os").environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snaps = sorted(glob(str(hub / f"models--{BASE_REPO.replace('/', '--')}/snapshots/*/")))
    if not snaps:
        raise FileNotFoundError(f"No HF snapshot for {BASE_REPO} under {hub}")
    snap = Path(snaps[-1])
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(key):  # key includes ".weight"
        fname = idx[key]
        if fname not in handles:
            handles[fname] = safe_open(str(snap / fname), framework="pt", device="cpu")
        return handles[fname].get_tensor(key)

    return get, set(idx)


# --- finetuned (w_t) lazy reader: param-folder safetensors -------------------
def _ft_reader(expert_dir: Path):
    manifest = json.loads((expert_dir / "finetuned/param_manifest.json").read_text())["params"]
    params_dir = expert_dir / "finetuned/params"

    def get(key):  # key includes ".weight"
        return next(iter(__import__("safetensors.torch", fromlist=["load_file"]).load_file(
            str(params_dir / manifest[key]["file"])).values()))

    return get


def main():
    model, group = "Olmo-3-7b", "rl-zero"
    experts = ["Code", "IF", "Math"]
    ckpt = REPO_ROOT / f"artifacts/checkpoints/{model}/group-{group}/experts"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    base_get, base_keys = _base_reader()
    ft_get = {e: _ft_reader(ckpt / e) for e in experts}
    # mmap the big covariance pickles ONCE (peak RSS ~0.6GB each, lazy-paged).
    covs = {
        e: torch.load(ckpt / e / "covariance.pt", map_location="cpu", mmap=True, weights_only=True)
        for e in experts
    }

    # tracked layers = square-matrix cov keys present in every expert
    # (cov key == param key minus ".weight"; skip the paired `_n` count ints).
    def square_keys(sd):
        return {k for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 2 and v.size(0) == v.size(1)}

    tracked = sorted(set.intersection(*[square_keys(covs[e]) for e in experts]))
    rows = []
    for lc in tqdm(tracked, desc="layers"):
        wkey = lc + ".weight"
        if wkey not in base_keys:
            tqdm.write(f"[skip: not in base] {lc}")
            continue
        w0 = base_get(wkey).float().to(dev)                                   # (Do, Di)
        d = torch.stack([(ft_get[e](wkey).float().to(dev) - w0) for e in experts])  # (T, Do, Di)
        c = torch.stack([covs[e][lc].float().to(dev) for e in experts])       # (T, Di, Di)

        minimizers = [
            ("actmat", lambda: compute_rm_minimizer(w0, d, d.transpose(-2, -1) @ d)),
            ("regmean", lambda: compute_rm_minimizer(w0, d, c)),
            ("identity", lambda: identity_minimizer(w0, d)),
        ]
        for method, mk in minimizers:
            w_star = mk()
            loss = compute_rm_loss_v2(w_star - w0, d, c)              # always vs REAL cov
            rows.append({
                "model": model, "group": group, "method": method,
                "metric_type": "rm_loss", "metric": loss.item(),
                "layer": lc, "Di": w0.shape[-1], "Do": w0.shape[-2],
            })
        del w0, d, c

    out = REPO_ROOT / "artifacts/analysis/rm-loss/rm_loss_olmo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
