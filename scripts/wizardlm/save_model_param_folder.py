#!/usr/bin/env python3
"""Convert an HF causal-LM checkpoint to the param_folder layout consumed by
ParamFolderTaskVector — one safetensors file per parameter, plus a manifest.

Streams tensors directly from the on-disk shards (safetensors or
pytorch_model.bin) so peak RAM stays bounded by the largest single tensor,
not the whole model. Required for 13B-scale checkpoints on commodity nodes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoConfig, AutoTokenizer

_DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _safe_file_stem(param_name: str) -> str:
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def _write_tensor(
    key: str,
    tensor: torch.Tensor,
    target_dtype,
    params_dir: Path,
    manifest_params: dict,
) -> None:
    tensor = tensor.contiguous()
    if target_dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(target_dtype)
    stem = _safe_file_stem(key)
    filename = f"{stem}.safetensors"
    save_safetensors_file({"tensor": tensor}, str(params_dir / filename))
    manifest_params[key] = {
        "file": filename,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _stream_safetensors(shards, target_dtype, params_dir, manifest_params):
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                _write_tensor(
                    key, f.get_tensor(key), target_dtype, params_dir, manifest_params
                )


def _stream_pytorch_bin(shards, target_dtype, params_dir, manifest_params):
    for shard in shards:
        sd = torch.load(str(shard), map_location="cpu", weights_only=True)
        for key, tensor in sd.items():
            _write_tensor(key, tensor, target_dtype, params_dir, manifest_params)
        del sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=list(_DTYPE_MAP),
        help="Cast every floating-point tensor to this dtype before writing.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    local = Path(
        snapshot_download(
            args.model,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "pytorch_model.bin.index.json",
                "config.json",
                "generation_config.json",
                "tokenizer*",
                "*.model",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )

    config = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    config.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    target_dtype = _DTYPE_MAP[args.dtype]
    manifest_params: dict[str, dict] = {}

    st_shards = sorted(local.glob("*.safetensors"))
    if st_shards:
        _stream_safetensors(st_shards, target_dtype, params_dir, manifest_params)
    else:
        bin_shards = sorted(local.glob("pytorch_model*.bin"))
        if not bin_shards:
            raise FileNotFoundError(
                f"No safetensors or pytorch_model*.bin shards found in {local}"
            )
        _stream_pytorch_bin(bin_shards, target_dtype, params_dir, manifest_params)

    manifest = {
        "format": "safetensors",
        "model_id": args.model,
        "params": manifest_params,
    }
    (output_dir / "param_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved param_folder to {output_dir} ({len(manifest_params)} tensors)")


if __name__ == "__main__":
    main()
