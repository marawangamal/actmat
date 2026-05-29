"""Whole-model task-vector magnitudes across vision / language / OLMo experts.

Writes one row per (model, dataset) to artifacts/csvs/magnitudes_combined.csv.
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
OUT = "artifacts/csvs/magnitudes_combined.csv"
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
        "model": "roberta-base",
        "datasets": ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "roberta-large",
        "datasets": ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "Olmo-3-7b",
        "datasets": ["Math", "Code", "IF"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "Olmo-3-7b-polyglot",
        "datasets": ["ar", "cs", "de", "es"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "gemma-2-2b-it",
        "datasets": ["instruction", "math", "coding", "multilingual"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "gemma-2-9b-it",
        "datasets": ["instruction", "math", "coding", "multilingual"],
        "task_vector_cls": ParamFolderTaskVector,
    },
    {
        "model": "wizardlm",
        "datasets": ["LM", "Math", "Code"],
        "task_vector_cls": ParamFolderTaskVector,
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
        existing = pd.read_csv(OUT, usecols=["model", "dataset"])
        done = set(map(tuple, existing.drop_duplicates().to_records(index=False)))
    all_rows = []
    write_header = not osp.exists(OUT)
    for config in CONFIGS:
        model = config["model"]
        cls = config["task_vector_cls"]
        for dataset in tqdm(config["datasets"], desc=model):
            if (model, dataset) in done:
                print(f"[skip] {model}/{dataset} already in CSV")
                continue
            tv_dir = osp.join(ROOT, model, dataset)
            if not osp.isdir(tv_dir):
                print(f"[skip] missing checkpoint dir: {tv_dir}")
                continue
            tv = cls(tv_dir)
            mag = combined_magnitude(tv)
            row = {"model": model, "dataset": dataset, "magnitude": mag}
            all_rows.append(row)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=write_header, index=False)
            write_header = False
            del tv

    print(f"wrote {len(all_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
