#!/usr/bin/env python3
"""Compute per-layer Gram matrices of masked input activations for a causal LM.

This script uses ``trl.SFTTrainer`` to build the tokenized training dataloader.
For each linear-like leaf module, it collects the Gram matrix of the module's
input activations, masking out padded positions using the current batch's
``attention_mask``.

Example:
  python scripts/nlg/compute_input_grams.py \
    --model /path/to/model \
    --dataset allenai/c4 \
    --dataset-config en \
    --split train \
    --text-field text \
    --max-samples 256 \
    --batch-size 4 \
    --max-length 512 \
    --output-dir results/grams/c4_train
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    Conv1D = ()

try:
    from trl import SFTTrainer
except Exception as exc:  # pragma: no cover
    SFTTrainer = None
    TRL_IMPORT_ERROR = exc
else:
    TRL_IMPORT_ERROR = None


COMMON_TEXT_FIELDS = ("text", "content", "prompt", "completion", "document")
EXEMPT_MATRIX_WEIGHT_TYPES = (nn.Embedding,)


class GramAccumulator:
    def __init__(self, dim: int, device: torch.device):
        self.dim = dim
        self.device = device
        self.gram = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        self.num_vectors = 0

    def add(self, x: torch.Tensor) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        x = x.to(self.device, dtype=torch.float64)
        self.gram += x.transpose(0, 1) @ x
        self.num_vectors += x.shape[0]

    def as_numpy(self) -> np.ndarray:
        return self.gram.cpu().numpy()

    def mean_gram_numpy(self) -> Optional[np.ndarray]:
        if self.num_vectors == 0:
            return None
        return (self.gram / self.num_vectors).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute masked Gram matrices of linear-layer input activations."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--dataset", required=True, help="Dataset name or local dataset path.")
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config/subset name passed to datasets.load_dataset.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use.")
    parser.add_argument(
        "--text-field",
        default=None,
        help="Text field to tokenize. If omitted, tries common names like 'text'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of dataset examples.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes for SFTTrainer.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model load dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model forward passes.",
    )
    parser.add_argument(
        "--gram-device",
        default="cpu",
        help="Device where Gram matrices are accumulated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to HF loading.",
    )
    parser.add_argument(
        "--ignore-unexpected-matrix-weight",
        action="store_true",
        help="Skip the unexpected matrix-weight module check instead of raising.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def resolve_text_field(dataset: Dataset, requested_field: Optional[str]) -> str:
    if requested_field is not None:
        if requested_field not in dataset.column_names:
            raise ValueError(
                f"text field '{requested_field}' not found in dataset columns {dataset.column_names}"
            )
        return requested_field

    for field in COMMON_TEXT_FIELDS:
        if field in dataset.column_names:
            return field

    raise ValueError(
        "Could not infer text field. Pass --text-field explicitly. "
        f"Available columns: {dataset.column_names}"
    )


def is_linear_like_module(module: nn.Module) -> bool:
    linear_types: Tuple[type, ...] = (nn.Linear,) + ((Conv1D,) if Conv1D else ())
    return isinstance(module, linear_types)


def is_exempt_matrix_weight_module(module: nn.Module) -> bool:
    if isinstance(module, EXEMPT_MATRIX_WEIGHT_TYPES):
        return True
    module_name = module.__class__.__name__.lower()
    return "embedding" in module_name or "norm" in module_name


def has_direct_matrix_weight(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.nn.Parameter) and weight.ndim == 2


def apply_sequence_mask(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return x
    if x.ndim != 3:
        return x

    mask = attention_mask
    if mask.ndim != 2:
        raise ValueError(f"Expected attention mask with ndim=2, got shape {tuple(mask.shape)}")

    if x.shape[0] == mask.shape[0] and x.shape[1] == mask.shape[1]:
        selected = x[mask.to(device=x.device, dtype=torch.bool)]
        return selected

    if x.shape[0] == mask.shape[1] and x.shape[1] == mask.shape[0]:
        selected = x.permute(1, 0, 2)[mask.to(device=x.device, dtype=torch.bool)]
        return selected

    return x


def register_input_gram_hooks(
    model: nn.Module,
    gram_device: torch.device,
    mask_ref: List[Optional[torch.Tensor]],
) -> Tuple[Dict[str, GramAccumulator], List[torch.utils.hooks.RemovableHandle], Set[str]]:
    accumulators: Dict[str, GramAccumulator] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    hooked: Set[str] = set()

    for name, module in model.named_modules():
        if not is_linear_like_module(module):
            continue

        hooked.add(name)

        def make_hook(module_name: str):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                if not isinstance(x, torch.Tensor):
                    return
                if x.shape[-1] != mod.weight.shape[1]:
                    raise ValueError(
                        f"Input dim mismatch for {module_name}: "
                        f"got {x.shape[-1]}, expected {mod.weight.shape[1]}"
                    )

                x = apply_sequence_mask(x.detach(), mask_ref[0])
                if module_name not in accumulators:
                    accumulators[module_name] = GramAccumulator(x.shape[-1], gram_device)
                accumulators[module_name].add(x)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    return accumulators, handles, hooked


def validate_hook_coverage(
    model: nn.Module,
    hooked_module_names: Set[str],
    ignore_unexpected: bool,
) -> List[str]:
    exempt_modules = []
    unexpected_modules = []

    for name, module in model.named_modules():
        if list(module.children()):
            continue
        if not has_direct_matrix_weight(module):
            continue
        if name in hooked_module_names:
            continue
        if is_exempt_matrix_weight_module(module):
            exempt_modules.append(name)
            continue
        unexpected_modules.append(f"{name} ({module.__class__.__name__})")

    if unexpected_modules and not ignore_unexpected:
        raise RuntimeError(
            "Found leaf modules with 2D weights that are neither hooked nor exempt: "
            + ", ".join(unexpected_modules)
        )

    return exempt_modules


def build_sft_trainer(model, tokenizer, dataset: Dataset, text_field: str, args) -> SFTTrainer:
    if SFTTrainer is None:
        raise RuntimeError(
            "trl.SFTTrainer is unavailable. Install `trl` in the runtime environment."
        ) from TRL_IMPORT_ERROR

    tmp_dir = tempfile.mkdtemp(prefix="compute_input_grams_")
    training_args = TrainingArguments(
        output_dir=tmp_dir,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
    }

    trainer_signature = SFTTrainer.__init__.__code__.co_varnames
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_signature:
        trainer_kwargs["dataset_text_field"] = text_field
    if "max_seq_length" in trainer_signature:
        trainer_kwargs["max_seq_length"] = args.max_length
    if "dataset_kwargs" in trainer_signature:
        trainer_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": False}

    return SFTTrainer(**trainer_kwargs)


def load_text_dataset(args) -> Tuple[Dataset, str]:
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    text_field = resolve_text_field(dataset, args.text_field)
    return dataset, text_field


def save_results(
    output_dir: Path,
    args,
    text_field: str,
    num_examples: int,
    num_batches: int,
    accumulators: Dict[str, GramAccumulator],
    exempt_modules: Sequence[str],
) -> None:
    gram_arrays = {}
    for name, accumulator in accumulators.items():
        safe_name = name.replace(".", "__")
        gram_arrays[f"{safe_name}__gram"] = accumulator.as_numpy()
        gram_arrays[f"{safe_name}__num_vectors"] = np.array(accumulator.num_vectors, dtype=np.int64)
        mean_gram = accumulator.mean_gram_numpy()
        if mean_gram is not None:
            gram_arrays[f"{safe_name}__mean_gram"] = mean_gram

    np.savez(output_dir / "input_grams.npz", **gram_arrays)

    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_field": text_field,
        "num_examples": num_examples,
        "num_batches": num_batches,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "hooked_linear_modules": sorted(accumulators.keys()),
        "exempt_matrix_weight_modules": list(exempt_modules),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = resolve_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset, text_field = load_text_dataset(args)
    trainer = build_sft_trainer(model, tokenizer, raw_dataset, text_field, args)
    dataloader = trainer.get_train_dataloader()

    gram_device = torch.device(args.gram_device)
    model_device = torch.device(args.device)
    model.to(model_device)
    model.eval()

    mask_ref: List[Optional[torch.Tensor]] = [None]
    accumulators, handles, hooked_module_names = register_input_gram_hooks(
        model, gram_device, mask_ref
    )
    exempt_modules = validate_hook_coverage(
        model,
        hooked_module_names,
        ignore_unexpected=args.ignore_unexpected_matrix_weight,
    )

    num_batches = 0
    num_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model_device) for k, v in batch.items() if torch.is_tensor(v)}
            mask_ref[0] = batch.get("attention_mask")
            model(**batch)
            num_batches += 1
            if "input_ids" in batch:
                num_examples += batch["input_ids"].shape[0]

    for handle in handles:
        handle.remove()

    save_results(
        output_dir=output_dir,
        args=args,
        text_field=text_field,
        num_examples=num_examples,
        num_batches=num_batches,
        accumulators=accumulators,
        exempt_modules=exempt_modules,
    )

    print(f"Saved Gram matrices to: {output_dir / 'input_grams.npz'}")
    print(f"Saved metadata to: {output_dir / 'metadata.json'}")
    print(f"Hooked linear modules: {len(accumulators)}")
    print(f"Exempt matrix-weight modules: {len(exempt_modules)}")
    print(f"Processed examples: {num_examples}")


if __name__ == "__main__":
    main()
