"""Fine-tune Llama-3.1-8B on individual capability subsets of the Tulu 3 SFT mixture.

Each capability is trained separately to produce a specialist checkpoint,
which can later be merged using scripts/nlg/merge.py.

Uses allenai/tulu-3-sft-mixture (939k examples with 'messages' + 'source' columns),
filtered by source to isolate each capability.

Usage:
     python scripts/nlg/get_covariance.py --capability math --model pmahdavi/Llama-3.1-8B-coding --output-dir ../olmes/models/pmahdavi/Llama-3.1-8B-coding  --max-samples 320 
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    Conv1D = ()

PRETRAINED_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET_ID = "allenai/tulu-3-sft-mixture"
MAX_SEQ_LEN = 4096
CAPABILITY_SOURCES = {
    "math": [
        "ai2-adapt-dev/personahub_math_v5_regen_149960",
        "allenai/tulu-3-sft-personas-math-grade",
        "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
        "ai2-adapt-dev/numinamath_tir_math_decontaminated",
        "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    ],
    "coding": [
        "ai2-adapt-dev/personahub_code_v2_34999",
        "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    ],
    "general": [
        "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
        "ai2-adapt-dev/oasst1_converted",
        "ai2-adapt-dev/no_robots_converted",
    ],
    "knowledge": [
        "ai2-adapt-dev/flan_v2_converted",
        "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
        "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    ],
    "precise_if": [
        "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    ],
}

EXEMPT_MATRIX_WEIGHT_TYPES = (nn.Embedding,)


def _safe_file_stem(param_name: str) -> str:
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def _atomic_write_json(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def add_to_gram_matrix(gram: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.to(device=gram.device, dtype=torch.float64)
    gram += x.transpose(0, 1) @ x
    return gram


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
    if attention_mask is None or x.ndim != 3:
        return x

    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention mask with ndim=2, got shape {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(device=x.device, dtype=torch.bool)
    if x.shape[0] == mask.shape[0] and x.shape[1] == mask.shape[1]:
        return x[mask]
    return x


def register_input_gram_hooks(
    model: nn.Module,
    gram_device: torch.device,
    mask_ref: List[Optional[torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle], Set[str]]:
    accumulators: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    hooked_module_names: Set[str] = set()

    for name, module in model.named_modules():
        if not is_linear_like_module(module):
            continue

        hooked_module_names.add(name)
        accumulators[name] = torch.zeros(
            (module.weight.shape[1], module.weight.shape[1]),
            dtype=torch.float64,
            device=gram_device,
        )

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
                    raise KeyError(f"Missing preallocated Gram accumulator for {module_name}")
                accumulators[module_name] = add_to_gram_matrix(accumulators[module_name], x)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    return accumulators, handles, hooked_module_names


def validate_input_gram_hook_coverage(
    model: nn.Module,
    hooked_module_names: Set[str],
    ignore_unexpected: bool = False,
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


def save_input_gram_results(
    output_dir: str | Path,
    accumulators: Dict[str, torch.Tensor],
    metadata: Optional[Dict] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grams_dir = output_path / "params"
    grams_dir.mkdir(parents=True, exist_ok=True)

    num_tokens = metadata["num_tokens"]
    manifest = {
        "format": "pt",
        "model_id": "input_grams",
        "params": {},
    }
    for name, gram_matrix in accumulators.items():
        mean_gram = (gram_matrix / num_tokens).to(torch.float32).detach().cpu().contiguous()
        filename = f"{_safe_file_stem(name)}.pt"
        file_path = grams_dir / filename
        torch.save(mean_gram, file_path)
        manifest["params"][name] = {
            "file": filename,
            "shape": list(mean_gram.shape),
            "dtype": str(mean_gram.dtype),
        }

    _atomic_write_json(output_path / "param_manifest.json", manifest)

    if metadata is not None:
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def compute_input_grams(
    model: nn.Module,
    dataloader,
    gram_device: str | torch.device = "cuda",
    max_batches: Optional[int] = None,
    ignore_unexpected_matrix_weight: bool = False,
    progress_desc: str = "Computing covariance",
):
    gram_device = torch.device(gram_device)
    mask_ref: List[Optional[torch.Tensor]] = [None]

    accumulators, handles, hooked_module_names = register_input_gram_hooks(
        model=model,
        gram_device=gram_device,
        mask_ref=mask_ref,
    )
    exempt_modules = validate_input_gram_hook_coverage(
        model=model,
        hooked_module_names=hooked_module_names,
        ignore_unexpected=ignore_unexpected_matrix_weight,
    )

    model.eval()
    num_examples = 0
    num_tokens = 0
    model_device = next(model.parameters()).device

    total_batches = None
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, total=total_batches, desc=progress_desc)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {k: v.to(model_device) for k, v in batch.items() if torch.is_tensor(v)}
            attention_mask = batch.get("attention_mask")
            mask_ref[0] = attention_mask
            model(**batch)
            if "input_ids" in batch:
                num_examples += batch["input_ids"].shape[0]
            if attention_mask is not None:
                num_tokens += int(attention_mask.sum().item())
            elif "input_ids" in batch:
                num_tokens += int(batch["input_ids"].numel())

    for handle in handles:
        handle.remove()

    metadata = {
        "num_examples": num_examples,
        "num_tokens": num_tokens,
        "hooked_linear_modules": sorted(accumulators.keys()),
        "exempt_matrix_weight_modules": exempt_modules,
    }
    return accumulators, metadata


def format_example(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_train_dataloader(dataset, tokenizer, batch_size: int):
    def tokenize_batch(batch):
        texts = [format_example(messages, tokenizer) for messages in batch["messages"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def collate_fn(features):
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Get gram matrix Llama-3.1-8B on Tulu 3 capability subsets"
    )
    p.add_argument(
        "--capability",
        type=str,
        required=True,
        choices=list(CAPABILITY_SOURCES.keys()) + ["all"],
    )
    p.add_argument("--output-dir", type=str, default="checkpoints/nlg")
    p.add_argument(
        "--model",
        type=str,
        default=PRETRAINED_MODEL,
        help="Model checkpoint or HF model id to continue SFT from.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=False)
    return p.parse_args()


def get_covariance(capability, args):
    print(f"\n{'='*60}")
    print(f"Training capability: {capability}")
    print(f"{'='*60}")

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    # Load and filter dataset
    sources = CAPABILITY_SOURCES[capability]
    print(f"  Sources: {sources}")
    ds = load_dataset(DATASET_ID, split="train", cache_dir=args.hf_cache_dir)
    ds = ds.filter(lambda ex: ex["source"] in sources, desc=f"Filtering {capability}")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    print(f"  {len(ds)} examples")

    # Load tokenizer and model from the requested checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.hf_cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        cache_dir=args.hf_cache_dir,
    )

    run_dir = os.path.join(args.output_dir, f"Llama-3.1-8B-{capability}")

    dataloader = build_train_dataloader(ds, tokenizer, args.batch_size)

    accumulators, metadata = compute_input_grams(
        model=model,
        dataloader=dataloader,
        gram_device="cuda",
        progress_desc=f"Computing covariance ({capability})",
    )
    save_input_gram_results(
        output_dir=run_dir,
        accumulators=accumulators,
        metadata=metadata,
    )

    print(f"Done: {capability}")


def main():
    args = parse_args()

    if args.capability == "all":
        for cap in CAPABILITY_SOURCES:
            get_covariance(cap, args)
    else:
        get_covariance(args.capability, args)


if __name__ == "__main__":
    main()
