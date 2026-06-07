import argparse
import gc
import os
import os.path as osp
import sys

from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

torch = None
ClassificationHead = None
ImageClassifier = None
get_dataset = None
mhas = None
OnlineCovariance = None
register_hooks = None


def load_project_imports():
    global torch
    global ClassificationHead
    global ImageClassifier
    global get_dataset
    global mhas
    global OnlineCovariance
    global register_hooks

    import torch as torch_module
    from src import mhas as mhas_module
    from src.covariance import OnlineCovariance as OnlineCovarianceClass
    from src.covariance import register_hooks as register_cov_hooks
    from src.vision.datasets.registry import get_dataset as get_vision_dataset
    from src.vision.modeling import ClassificationHead as Head
    from src.vision.modeling import ImageClassifier as Classifier

    torch = torch_module
    ClassificationHead = Head
    ImageClassifier = Classifier
    get_dataset = get_vision_dataset
    mhas = mhas_module
    OnlineCovariance = OnlineCovarianceClass
    register_hooks = register_cov_hooks


def parse_int_list(value):
    return [int(v) for v in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-name", default="finetuned.pt")
    parser.add_argument("--data-location", default="data/vision")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "models"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cov-split", default="train", choices=["train", "test"])
    parser.add_argument("--cov-num-batches", type=parse_int_list, default=[10])
    parser.add_argument("--cov-batch-size", type=int, default=32)
    parser.add_argument("--cov-type", choices=["cov", "sm"], default="sm")
    parser.add_argument("--cov-estimator", choices=["sampled", "full", "avg"], default="full")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def compute_covs(encoder, args, on_end=None):
    head_path = osp.join(args.expert_dir, "head.pt")
    classification_head = ClassificationHead.load(head_path)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.eval()
    model.to(args.model_device)

    dataset = get_dataset(
        args.dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.cov_batch_size,
        num_workers=args.num_workers,
    )
    loader = dataset.train_loader if args.cov_split == "train" else dataset.test_loader
    max_num_batches = max(args.cov_num_batches)
    print(f"    {len(loader.dataset)} samples (split={args.cov_split})")

    cobjs, handles = register_hooks(
        model,
        cov_device=args.cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        extra_module_types=(mhas.MultiHeadAttentionSplit,),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    n_batches = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Computing covariance"):
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
    if osp.exists(args.output_path) and not args.overwrite:
        print(f"Skipping cached covariance: {args.output_path}")
        raise SystemExit(0)

    load_project_imports()
    args.dataset_name = args.dataset if args.dataset.endswith("Val") else f"{args.dataset}Val"
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")
    args.device = args.model_device

    checkpoint_path = osp.join(args.expert_dir, args.checkpoint_name)
    print(f"\nCollecting covariance for {args.dataset_name}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output:     {args.output_path}")

    encoder = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = mhas.swap_mha(encoder)

    def on_end(cobjs):
        saveable = {}
        for lname in list(cobjs):
            if isinstance(cobjs[lname], OnlineCovariance):
                saveable[lname] = cobjs[lname].cov.cpu()
                saveable[f"{lname}_n"] = cobjs[lname].n
            else:
                saveable[lname] = cobjs[lname]
        output_dir = osp.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(saveable, args.output_path)
        print(f"  Saved to {args.output_path}")

    compute_covs(encoder, args, on_end=on_end)
    del encoder
    gc.collect()
