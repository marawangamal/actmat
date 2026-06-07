import json
import os
import os.path as osp
from types import SimpleNamespace

import numpy as np
import torch
import tqdm

from src import utils
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.modeling import ClassificationHead, ImageClassifier


DEFAULT_VIT_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]


def parse_csv(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def make_eval_namespace(args):
    return SimpleNamespace(
        model=args.model,
        data_location=args.data_location,
        cache_dir=args.cache_dir,
        feature_cache_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_split=getattr(args, "eval_split", "test"),
        eval_max_batches=getattr(args, "eval_max_batches", None),
    )


def load_head(head_path):
    if not osp.exists(head_path):
        raise FileNotFoundError(f"Missing classification head: {head_path}")
    return ClassificationHead.load(head_path)


def eval_dataset_with_head(image_encoder, dataset_name, head_path, args):
    classification_head = load_head(head_path)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    use_train = getattr(args, "eval_split", "test") == "train"
    dataloader = get_dataloader(dataset, is_train=use_train, args=args, image_encoder=None)
    device = args.device
    max_batches = getattr(args, "eval_max_batches", None)

    with torch.no_grad():
        correct, n = 0.0, 0.0
        for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

    top1 = correct / n if n > 0 else 0.0
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return {"top1": top1}


def write_metrics(output_dir, tasks, model_config):
    os.makedirs(output_dir, exist_ok=True)
    metrics_json = {
        "all_primary_scores": [
            f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
        ],
        "tasks": tasks,
        "model_config": model_config,
    }
    metrics_path = osp.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Results saved to {metrics_path}")


def make_tasks(scores):
    return [
        {
            "alias": name,
            "metrics": {"top1": score, "primary_score": score},
            "task_config": {"primary_metric": "top1"},
        }
        for name, score in scores.items()
        if isinstance(score, (int, float, np.floating))
    ]
