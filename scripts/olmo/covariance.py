"""Collect per-layer activation covariances for Olmo-3-7B on Dolci RL-Zero datasets.

Runs forward passes (no training) while forward hooks from src/covariance.py
capture activation statistics. Each capability produces a covariance NPZ file
that can be used by RegMean or other covariance-based merge methods.

Usage:
    export PYTHONPATH="$PYTHONPATH:$PWD"

    # Single capability
    python scripts/olmo/covariance.py --capability math --hf-cache-dir $SCRATCH/huggingface

    # All capabilities
    python scripts/olmo/covariance.py --capability all --hf-cache-dir $SCRATCH/huggingface
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.covariance import register_hooks

PRETRAINED_MODEL = "allenai/Olmo-3-1025-7B"
MAX_SEQ_LEN = 4096

CAPABILITY_DATASETS = {
    "math": "allenai/Dolci-RL-Zero-Math-7B",
    "code": "allenai/Dolci-RL-Zero-Code-7B",
    "if": "allenai/Dolci-RL-Zero-IF-7B",
}

ANSWER_COLUMN = {
    "math": "ground_truth",
    "code": "solution",
    "if": "ground_truth",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect activation covariances for Olmo-3-7B"
    )
    p.add_argument(
        "--capability",
        type=str,
        required=True,
        choices=list(CAPABILITY_DATASETS.keys()) + ["all"],
    )
    p.add_argument("--output-dir", type=str, default="checkpoints/olmo")
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--cov-type",
        type=str,
        default="sm",
        choices=["sm", "cov"],
    )
    p.add_argument(
        "--cov-estimator",
        type=str,
        default="full",
        choices=["avg", "sampled", "full"],
    )
    p.add_argument("--cov-num-batches", type=int, default=10)
    p.add_argument("--cov-batch-size", type=int, default=32)
    return p.parse_args()


def _prepare_messages(ds, capability):
    """Construct a uniform `messages` column from capability-specific schemas."""
    answer_col = ANSWER_COLUMN[capability]

    def _to_messages(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex[answer_col]},
            ]
        }

    ds = ds.map(_to_messages, desc=f"Preparing messages for {capability}")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    return ds


def collect_covariance(capability, args):
    print(f"\n{'='*60}")
    print(f"Collecting covariance: {capability}")
    print(f"{'='*60}")

    run_dir = os.path.join(args.output_dir, f"Olmo-3-7B-{capability}")

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    # Load dataset
    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.hf_cache_dir)
    ds = _prepare_messages(ds, capability)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, cache_dir=args.hf_cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_fn(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False)
            for msgs in examples["messages"]
        ]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors=None,
        )

    ds = ds.map(
        tokenize_fn, batched=True, remove_columns=["messages"], desc="Tokenizing"
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(ds, batch_size=args.cov_batch_size, shuffle=False)

    # Load model
    print("Loading model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        cache_dir=args.hf_cache_dir,
    )
    model.to(device)
    model.eval()

    # Register forward hooks
    cobjs, handles = register_hooks(
        model,
        cov_device="cpu",
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        batch_first=True,
    )

    # Forward pass loop
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Computing covariance",
            total=min(args.cov_num_batches, len(dataloader)),
        ):
            if n_batches >= args.cov_num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            n_batches += 1

    print(f"  Processed {n_batches} batches")

    # Remove hooks and save
    for h in handles:
        h.remove()

    saveable = {}
    for name, cobj in cobjs.items():
        saveable[name] = cobj.cov.cpu().numpy()
        saveable[f"{name}_n"] = cobj.n

    os.makedirs(run_dir, exist_ok=True)
    cov_path = os.path.join(run_dir, f"covariance_{capability}.npz")
    np.savez(cov_path, **saveable)
    print(f"Saved covariances ({len(cobjs)} layers) to {cov_path}")


def main():
    args = parse_args()

    if args.capability == "all":
        for cap in CAPABILITY_DATASETS:
            collect_covariance(cap, args)
    else:
        collect_covariance(args.capability, args)


if __name__ == "__main__":
    main()
