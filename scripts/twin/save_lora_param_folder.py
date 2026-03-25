#!/usr/bin/env python3
"""Convert a LoRA adapter + base model into param-folder format.

Loads the base model, applies the LoRA adapter, merges weights, then exports
each parameter as an individual safetensors file with a manifest.

Usage:
    python scripts/twin/save_lora_param_folder.py \
        --base-model Qwen/Qwen-14B \
        --adapter lu-vae/qwen-bbq-merged \
        --output-dir checkpoints/twin/qwen-bbq-merged

    # All 4 adapters + base:
    python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-bbq-merged --output-dir checkpoints/twin/qwen-bbq-merged
    python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-cnn-merged --output-dir checkpoints/twin/qwen-cnn-merged
    python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-mmlu-merged --output-dir checkpoints/twin/qwen-mmlu-merged
    python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-truthfulqa-merged --output-dir checkpoints/twin/qwen-truthfulqa-merged
    # Base model (no adapter):
    python scripts/nlg/save_model_param_folder.py --model Qwen/Qwen-14B --output-dir checkpoints/twin/Qwen-14B
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _safe_file_stem(param_name: str) -> str:
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def _parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA adapter + base model into param-folder format."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HF model ID for the base model (e.g. Qwen/Qwen-14B).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="HF adapter repo or local path (e.g. lu-vae/qwen-bbq-merged).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output param_folder directory.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory.",
    )
    args = parser.parse_args()

    import os
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    output_dir = Path(args.output_dir).expanduser().resolve()
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=_parse_dtype(args.dtype),
        device_map="cpu",
        cache_dir=args.hf_cache_dir,
    )

    # Load and merge LoRA adapter
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, cache_dir=args.hf_cache_dir)
    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    # Save config + tokenizer
    config = AutoConfig.from_pretrained(
        args.base_model, trust_remote_code=True, cache_dir=args.hf_cache_dir
    )
    config.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, cache_dir=args.hf_cache_dir
    )
    tokenizer.save_pretrained(str(output_dir))

    # Serialize each parameter as an individual file
    use_safetensors = False
    try:
        from safetensors.torch import save_file as save_safetensors_file

        use_safetensors = True
    except Exception:
        save_safetensors_file = None

    manifest = {
        "format": "safetensors" if use_safetensors else "pt",
        "model_id": args.base_model,
        "params": {},
    }

    print("Exporting parameters ...")
    with torch.no_grad():
        for param_name, tensor in model.state_dict().items():
            tensor_cpu = tensor.detach().cpu().contiguous()
            stem = _safe_file_stem(param_name)
            if use_safetensors:
                filename = f"{stem}.safetensors"
                save_safetensors_file(
                    {"tensor": tensor_cpu}, str(params_dir / filename)
                )
            else:
                filename = f"{stem}.pt"
                torch.save(tensor_cpu, params_dir / filename)
            manifest["params"][param_name] = {
                "file": filename,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype),
            }

    manifest_path = output_dir / "param_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved param_folder to {output_dir}")


if __name__ == "__main__":
    main()
