"""Per-layer task-vector magnitudes across vision / language / OLMo experts.

Writes one row per (model, dataset, layer) to artifacts/csvs/magnitudes.csv.
CPU-only; uses lazy task-vector loading so peak RSS stays bounded to a
single parameter at a time.
"""

import os
import os.path as osp

import pandas as pd
import torch
from tqdm import tqdm

from src import mhap, mhas  # noqa: F401  (kept for parity with original snippet)
from src.nlg.task_vectors import ParamFolderTaskVector
from src.vision.task_vectors import NonLinearTaskVector


ROOT = "artifacts/checkpoints"
OUT = "artifacts/csvs/magnitudes.csv"
SKIP_SUBSTRINGS = ("lm_head", "embed")

CONFIGS = [
    {
        "model": "ViT-B-16",
        "datasets": [d + "Val" for d in ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"]],
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "ViT-L-14",
        "datasets": [d + "Val" for d in ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "SVHN"]],
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "t5-base",
        "datasets": ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"],
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "t5-large",
        "datasets": ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"],
        "task_vector_cls": NonLinearTaskVector,
    },
    {
        "model": "Olmo-3-7b",
        "datasets": ["Math", "Code", "IF"],
        "task_vector_cls": ParamFolderTaskVector,
    },
]


def collect_rows(model, dataset, tv):
    out = []
    layer_idx = 0
    for layer_key in tqdm(tv.lazy_keys(), desc=dataset, leave=False):
        if any(s in layer_key for s in SKIP_SUBSTRINGS):
            continue
        d_t = tv.get_vector_element(layer_key)
        if d_t.ndim != 2:
            continue
        mag = torch.linalg.norm(d_t).item()
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "magnitude": mag,
                "layer_idx": layer_idx,
                "layer_key": layer_key,
            }
        )
        layer_idx += 1
    return out


def main():
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    all_rows = []
    write_header = not osp.exists(OUT)
    for config in CONFIGS:
        model = config["model"]
        cls = config["task_vector_cls"]
        for dataset in tqdm(config["datasets"], desc=model):
            tv_dir = osp.join(ROOT, model, dataset)
            if not osp.isdir(tv_dir):
                print(f"[skip] missing checkpoint dir: {tv_dir}")
                continue
            tv = cls(tv_dir)
            rows = collect_rows(model, dataset, tv)
            all_rows.extend(rows)
            # Incremental append: survive an OOM on a later config.
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=write_header, index=False)
            write_header = False
            del tv

    print(f"wrote {len(all_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
