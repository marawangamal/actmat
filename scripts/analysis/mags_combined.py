"""Whole-model task-vector magnitudes across vision / language / OLMo experts.

Writes one row per (model, dataset) to artifacts/analysis/magnitudes/magnitudes_combined.csv.
The combined magnitude is the global Frobenius norm of all 2-D layer deltas:
sqrt(sum_l ||delta_l||_F^2). CPU-only; lazy task-vector loading keeps peak
RSS bounded to a single parameter at a time.
"""

import os
import os.path as osp

import pandas as pd
import torch
from tqdm import tqdm

from src import mhap, mhas  # noqa: F401
from src.nlg.task_vectors import ParamFolderTaskVector
from src.vision.task_vectors import NonLinearTaskVector


ROOT = "artifacts/checkpoints"
OUT = "artifacts/analysis/magnitudes/magnitudes_sgd_compare.csv"
SKIP_SUBSTRINGS = ("lm_head", "embed")

# ViT-B-16 task-vector magnitude comparison across optimizer/LR/WD recipes.
# Each config has its own checkpoint `root`; `variant` labels the recipe in the
# output CSV so the three sets are comparable side by side.
_VISION_8 = [
    d + "Val"
    for d in ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
]

# OLMo-3-7B polyglot language experts (param-folder checkpoints). Each
# `<lang>/` dir holds a `finetuned/` param folder plus a `pretrained` symlink to
# the shared base, so ParamFolderTaskVector reads them with no extra setup.
_POLYGLOT_LANGS = ["ar", "cs", "de", "es"]

# WizardLM (Llama-2-13B) domain experts, same param-folder layout as polyglot:
# each `<expert>/` dir has a `finetuned/` param folder plus a `pretrained` symlink.
_WIZARDLM_EXPERTS = ["Math", "Code", "LM"]

CONFIGS = [
    {
        "model": "Olmo-3-7b-polyglot",
        "variant": "polyglot-sft",
        "root": "artifacts/checkpoints",
        "datasets": _POLYGLOT_LANGS,
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "wizardlm",
        "variant": "wizardlm-13b",
        "root": "artifacts/checkpoints",
        "datasets": _WIZARDLM_EXPERTS,
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "ViT-B-16",
        "variant": "adamw-1e-5-wd0.1",
        "root": "artifacts/checkpoints",
        "datasets": _VISION_8,
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "ViT-B-16",
        "variant": "sgd-1e-4-wd0.1",
        "root": "artifacts/checkpoints-sgd",
        "datasets": _VISION_8,
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "ViT-B-16",
        "variant": "sgd-1e-5-wd0",
        "root": "artifacts/checkpoints-sgd-lr1e5-wd0",
        "datasets": _VISION_8,
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        # Only Cars exists for this recipe (single-dataset run, not the full 8).
        "model": "ViT-B-16",
        "variant": "sgd-1e-5-wd0.1",
        "root": "artifacts/checkpoints-sgd-lr1e-5",
        "datasets": _VISION_8,
        "task_vector_cls": NonLinearTaskVector,
    },
]


def combined_magnitude(tv):
    sq_sum = 0.0
    for layer_key in tqdm(tv.lazy_keys(), leave=False):
        if any(s in layer_key for s in SKIP_SUBSTRINGS):
            continue
        d_t = tv.get_vector_element(layer_key)
        if d_t.ndim != 2:
            continue
        sq_sum += torch.linalg.norm(d_t).item() ** 2
    return sq_sum ** 0.5


def main():
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    done = set()
    if osp.exists(OUT):
        existing = pd.read_csv(OUT, usecols=["model", "variant", "dataset"])
        done = set(map(tuple, existing.drop_duplicates().to_records(index=False)))
    all_rows = []
    write_header = not osp.exists(OUT)
    for config in CONFIGS:
        model = config["model"]
        variant = config.get("variant", "")
        root = config.get("root", ROOT)
        cls = config["task_vector_cls"]
        for dataset in tqdm(config["datasets"], desc=f"{model}/{variant}"):
            if (model, variant, dataset) in done:
                print(f"[skip] {model}/{variant}/{dataset} already in CSV")
                continue
            tv_dir = osp.join(root, model, dataset)
            if not osp.isdir(tv_dir):
                print(f"[skip] missing checkpoint dir: {tv_dir}")
                continue
            # Skip experts still training. Param-folder checkpoints expose a
            # `finetuned/` directory; .pt checkpoints expose a `finetuned.pt` file.
            if cls is ParamFolderTaskVector:
                ft_marker = osp.join(tv_dir, "finetuned", "param_manifest.json")
            else:
                ft_marker = osp.join(tv_dir, "finetuned.pt")
            if not osp.exists(ft_marker):
                print(f"[skip] no finetuned checkpoint yet: {tv_dir}")
                continue
            tv = cls(tv_dir)
            mag = combined_magnitude(tv)
            row = {"model": model, "variant": variant, "dataset": dataset, "magnitude": mag}
            all_rows.append(row)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=write_header, index=False)
            write_header = False
            del tv

    print(f"wrote {len(all_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
