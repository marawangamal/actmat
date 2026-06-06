"""Collect per-layer activation covariances for Olmo-3-7B on Dolci RL-Zero datasets.

Runs forward passes (no training) through each capability's **finetuned expert**
while forward hooks from src/covariance.py capture activation statistics.
Activations are taken from the expert (not the pretrained base) so RegMean /
ActMat see the same inputs the expert's linear layers receive at deployment.

Each capability produces a covariance file saved inside its checkpoint
directory, where ParamFolderTaskVector auto-discovers it for merging.

Usage:
    export PYTHONPATH="$PYTHONPATH:$PWD"

    # Single capability
    python scripts/olmo/covariance.py --capability math --save artifacts/checkpoints/olmo

    # All capabilities
    python scripts/olmo/covariance.py --capability all --save artifacts/checkpoints/olmo
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.args import parse_arguments
from src.covariance import register_hooks
from src.nlg.task_vectors import (
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)

MAX_SEQ_LEN = 256

CAPABILITY_DATASETS = {
    "Math": "allenai/Dolci-RL-Zero-Math-7B",
    "Code": "allenai/Dolci-RL-Zero-Code-7B",
    "IF": "allenai/Dolci-RL-Zero-IF-7B",
}


def _load_expert_model(finetuned_dir: Path, device: torch.device):
    """Materialize the expert from the param-folder layout in ``finetuned_dir``.

    Init the model normally (not on meta) so non-persistent buffers like
    ``rotary_emb.inv_freq`` get computed correctly, then copy each param into
    place from its safetensors file. Peak CPU memory stays at ~one model copy.
    """
    manifest = _load_manifest(finetuned_dir)
    config = AutoConfig.from_pretrained(str(finetuned_dir))
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    state = model.state_dict()
    for key in tqdm(list(manifest["params"]), desc="Loading expert params"):
        if key not in state:
            raise KeyError(f"Param '{key}' in manifest but not in model state_dict")
        t = _load_single_tensor(
            _build_param_file_path(finetuned_dir, manifest, key)
        )
        state[key].copy_(t.to(state[key].dtype))
        del t
    model.to(device)
    model.eval()
    return model


def collect_covariance(capability, args):
    print(f"\n{'='*60}")
    print(f"Collecting covariance: {capability}")
    print(f"{'='*60}")

    run_dir = Path(args.save) / capability
    cov_path = run_dir / "covariance.pt"
    finetuned_dir = run_dir / "finetuned"

    if cov_path.exists() and not args.overwrite:
        print(f"  Skipping {capability} — {cov_path} already exists")
        return

    if not finetuned_dir.exists():
        raise FileNotFoundError(
            f"Expert checkpoint not found: {finetuned_dir}. "
            "Run scripts/olmo/save_model_param_folder.py first."
        )

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    # Load dataset
    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.cache_dir)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    # Load this expert's own tokenizer — chat templates can differ across capabilities.
    tokenizer = AutoTokenizer.from_pretrained(str(finetuned_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize only the prompts (no assistant answers needed for covariance)
    def tokenize_fn(examples):
        messages = [
            [{"role": "user", "content": prompt}] for prompt in examples["prompt"]
        ]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages
        ]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors=None,
        )

    ds = ds.map(
        tokenize_fn, batched=True, remove_columns=ds.column_names, desc="Tokenizing"
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(ds, batch_size=args.cov_batch_size, shuffle=False)

    # Load the *expert* (finetuned) model so hooks see post-finetune activations.
    print(f"Loading expert from {finetuned_dir} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_expert_model(finetuned_dir, device)

    # Register forward hooks
    cobjs, handles = register_hooks(
        model,
        cov_device="cpu",
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        batch_first=True,
    )

    # Forward pass loop
    max_num_batches = max(args.cov_num_batches)
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Computing covariance",
            total=min(max_num_batches, len(dataloader)),
        ):
            if n_batches >= max_num_batches:
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
        saveable[name] = cobj.cov.cpu()
        saveable[f"{name}_n"] = cobj.n

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saveable, cov_path)
    print(f"Saved covariances ({len(cobjs)} layers) to {cov_path}")


def main():
    args = parse_arguments()
    if args.save is None:
        args.save = "artifacts/checkpoints/Olmo-3-7b"

    if args.capability == "all":
        for cap in CAPABILITY_DATASETS:
            collect_covariance(cap, args)
    else:
        collect_covariance(args.capability, args)


if __name__ == "__main__":
    main()
