import argparse
import gc
import os
import os.path as osp
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from src.covariance import OnlineCovariance, register_hooks  # noqa: E402
from src.language.datasets.batcher import Batcher  # noqa: E402
from src.language.datasets.dataset_readers import get_datasetReader  # noqa: E402
from src.language.datasets.pytorch_dataset import PytorchDataset  # noqa: E402


def parse_int_list(value):
    return [int(v) for v in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-name", default="finetuned.pt")
    parser.add_argument("--data-location", default="data")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "huggingface"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cov-split", default="train", choices=["train", "test"])
    parser.add_argument("--cov-num-batches", type=parse_int_list, default=[10])
    parser.add_argument("--cov-batch-size", type=int, default=32)
    parser.add_argument("--cov-type", choices=["cov", "sm"], default="sm")
    parser.add_argument(
        "--cov-estimator", choices=["sampled", "full", "avg"], default="full"
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def compute_covs(model, args, on_end=None):
    model.eval()
    model.to(args.model_device)
    tokenizer = model.tokenizer

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }
    dataset_reader = get_datasetReader(args.dataset, dataset_kwargs)
    create_dataset = lambda dataset: PytorchDataset(
        dataset, tokenizer, str(args.model_device)
    )
    batcher = Batcher(
        dataset_reader,
        create_dataset,
        train_batchSize=args.cov_batch_size,
        eval_batchSize=args.cov_batch_size,
        world_size=None,
        device=None,
    )
    data_iter = batcher.get_splitOfBatches(
        args.cov_split, template_idx=0, is_evaluation=False
    )

    mask_ref = [None, None]
    cobjs, handles = register_hooks(
        model,
        cov_device=args.cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        mask_ref=mask_ref,
        batch_first=True,
    )

    n_batches = 0
    max_batches = max(args.cov_num_batches)
    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Computing covariance", total=max_batches):
            if max_batches is not None and n_batches >= max_batches:
                break
            mask_ref[0] = batch.get("input_mask")
            mask_ref[1] = batch.get("target_mask")
            model(batch)
            n_batches += 1
    print(f"    Used {n_batches} batches (max={max_batches})")

    for handle in handles:
        handle.remove()

    if on_end is not None:
        on_end(cobjs)

    model.cpu()
    del batcher, handles
    gc.collect()


if __name__ == "__main__":
    args = parse_args()
    if osp.exists(args.output_path) and not args.overwrite:
        print(f"Skipping cached covariance: {args.output_path}")
        raise SystemExit(0)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")

    checkpoint_path = osp.join(args.expert_dir, args.checkpoint_name)
    print(f"\nCollecting covariance for {args.dataset}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output:     {args.output_path}")
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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

    compute_covs(model, args, on_end=on_end)
