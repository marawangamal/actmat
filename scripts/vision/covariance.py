"""Collect activation covariance statistics for one vision checkpoint."""

import argparse
import gc
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

torch = None
build_classification_head = None
ClassificationHead = None
ImageClassifier = None
ImageEncoder = None
get_dataset = None
mhap = None
mhas = None
OnlineCovariance = None
register_hooks = None


def load_project_imports():
    global torch
    global build_classification_head
    global ClassificationHead
    global ImageClassifier
    global ImageEncoder
    global get_dataset
    global mhap
    global mhas
    global OnlineCovariance
    global register_hooks

    import torch as torch_module
    from src.vision.heads import build_classification_head as build_head
    from src.vision.modeling import (
        ClassificationHead as Head,
        ImageClassifier as Classifier,
        ImageEncoder as Encoder,
    )
    from src.vision.datasets.registry import get_dataset as get_vision_dataset
    from src import mhap as mhap_module, mhas as mhas_module
    from src.covariance import OnlineCovariance as OnlineCovarianceClass
    from src.covariance import register_hooks as register_cov_hooks

    torch = torch_module
    build_classification_head = build_head
    ClassificationHead = Head
    ImageClassifier = Classifier
    ImageEncoder = Encoder
    get_dataset = get_vision_dataset
    mhap = mhap_module
    mhas = mhas_module
    OnlineCovariance = OnlineCovarianceClass
    register_hooks = register_cov_hooks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect covariance statistics for a single vision checkpoint."
    )
    parser.add_argument("--model", required=True, help="OpenCLIP model name.")
    parser.add_argument(
        "--finetuned-path",
        required=True,
        help="Path to finetuned.pt or checkpoint_<step>.pt.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Exact output path for the covariance .pt file.",
    )
    parser.add_argument(
        "--data-location",
        default="data/vision",
        help="Root directory for vision datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("SCRATCH", os.path.expanduser("~/.cache")), "models"
        ),
        help="OpenCLIP cache directory.",
    )
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cov-split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--cov-num-batches",
        type=lambda x: [int(v) for v in x.split(",")],
        default=[10],
    )
    parser.add_argument("--cov-batch-size", type=int, default=32)
    parser.add_argument("--cov-type", choices=["cov", "sm"], default="sm")
    parser.add_argument(
        "--cov-estimator", choices=["sampled", "full", "avg"], default="full"
    )
    parser.add_argument("--mha", choices=["packed", "split"], default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def dataset_from_checkpoint(path):
    dataset = Path(path).parent.name
    if not dataset.endswith("Val"):
        dataset = f"{dataset}Val"
    return dataset


def get_direct_classification_head(args, dataset_name):
    checkpoint_dir = Path(args.finetuned_path).parent
    candidates = [
        checkpoint_dir / "head.pt",
        checkpoint_dir.parent / f"head_{dataset_name}.pt",
    ]
    for filename in candidates:
        if filename.exists():
            return ClassificationHead.load(str(filename))

    print(
        f"Did not find classification head for {args.model} on {dataset_name}; "
        f"building and saving {candidates[0]}."
    )
    model = ImageEncoder(args, keep_lang=True).model
    classification_head = build_classification_head(
        model,
        dataset_name,
        None,
        args.data_location,
        args.device,
    )
    os.makedirs(candidates[0].parent, exist_ok=True)
    classification_head.save(str(candidates[0]))
    return classification_head


def compute_covs(encoder, dataset_name, args, on_end=None):
    classification_head = get_direct_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.eval()
    model.to(args.model_device)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.cov_batch_size,
        num_workers=args.num_workers,
    )
    split = args.cov_split
    max_num_batches = max(args.cov_num_batches)
    loader = dataset.train_loader if split == "train" else dataset.test_loader
    dataset_size = len(loader.dataset)
    print(f"    {dataset_size} samples (split={split})")

    cobjs, handles = register_hooks(
        model,
        cov_device=args.cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        extra_module_types=(
            mhap.MultiHeadAttentionPacked,
            mhas.MultiHeadAttentionSplit,
        ),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    total_batches = len(loader) if max_num_batches is None else None
    n_batches = 0
    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc="Computing covariance",
            total=total_batches,
        ):
            if max_num_batches is not None and n_batches >= max_num_batches:
                break
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            _ = loss_fn(model(images), labels)
            n_batches += 1
    print(f"    Used {n_batches} batches (max={max_num_batches})")

    del model, dataset, loader, handles
    gc.collect()

    if on_end is not None:
        on_end(cobjs)


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Skipping cached covariance: {args.output_path}")
        sys.exit(0)

    load_project_imports()

    dataset_name = dataset_from_checkpoint(args.finetuned_path)
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")
    args.device = args.model_device

    print(f"\nCollecting covariance for {dataset_name}")
    print(f"  checkpoint: {args.finetuned_path}")
    print(f"  output:     {args.output_path}")
    encoder = torch.load(args.finetuned_path, map_location="cpu", weights_only=False)

    if args.mha is not None:
        swap_fn = {
            "packed": mhap.swap_mha,
            "split": mhas.swap_mha,
        }[args.mha]
        encoder = swap_fn(encoder)

    def on_end(cobjs):
        saveable = {}
        for lname in list(cobjs):
            if isinstance(cobjs[lname], OnlineCovariance):
                saveable[lname] = cobjs[lname].cov.cpu()
                saveable[f"{lname}_n"] = cobjs[lname].n
            else:
                saveable[lname] = cobjs[lname]
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(saveable, args.output_path)
        print(f"  Saved to {args.output_path}")

    compute_covs(encoder, dataset_name, args, on_end=on_end)
    del encoder
    gc.collect()
