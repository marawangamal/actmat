"""Merge WizardLM-13B param-folder task vectors into a HuggingFace checkpoint.

Reproduces the DARE paper Fig. 1 (right) merging of WizardLM-13B,
WizardMath-13B and llama-2-13b-code-alpaca. Uses ParamFolderTaskVector
for lazy, memory-bounded merging — only one parameter is loaded per model
at a time.

Usage:
    python scripts/wizardlm/merge.py \\
        --save artifacts/checkpoints/wizardlm \\
        --merge-func dare \\
        --merge-kwargs '{"drop_rate": 0.5, "base_merge": "sum"}' \\
        --output-dir artifacts/checkpoints/wizardlm-merged-dare
"""

import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.args import parse_arguments
from src.merging import combine_task_vectors
from src.nlg.task_vectors import (
    ParamFolderTaskVector,
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)

WIZARDLM_TASKS = ["LM", "Math", "Code"]


def merge(args):
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    save_root = Path(args.save).expanduser().resolve()
    task_dirs = [save_root / t for t in WIZARDLM_TASKS]
    output_dir = Path(args.output_dir).expanduser().resolve()

    for td in task_dirs:
        if not (td / "pretrained").exists() or not (td / "finetuned").exists():
            raise FileNotFoundError(
                f"{td} must contain pretrained/ and finetuned/ subdirectories. "
                "Run scripts/wizardlm/download_models.sh first."
            )

    pretrained_dir = (task_dirs[0] / "pretrained").resolve()
    print(f"Tasks          : {WIZARDLM_TASKS}")
    print(f"Merge function : {args.merge_func}")
    print(f"Output dir     : {output_dir}")
    print("=" * 80)

    task_vectors = [ParamFolderTaskVector(checkpoint_dir=str(td)) for td in task_dirs]
    merged_tv = combine_task_vectors(
        task_vectors,
        args.merge_func,
        ignore_keys=args.ignore_keys,
        **args.merge_kwargs,
    )

    # Apply deltas to pretrained tensors lazily to stay memory-bounded.
    pre_manifest = _load_manifest(pretrained_dir)
    merged_vector = merged_tv.vector
    del merged_tv
    final_sd = {}
    for key in tqdm(list(merged_vector.keys()), desc="Applying deltas"):
        delta = merged_vector.pop(key)
        pre_t = _load_single_tensor(
            _build_param_file_path(pretrained_dir, pre_manifest, key)
        )
        final_sd[key] = (pre_t.float() + delta).to(pre_t.dtype)

    # Fill any keys skipped by combine_task_vectors (e.g. --ignore-keys
    # embed_tokens/lm_head when vocab shapes differ) with pretrained values.
    for key in pre_manifest["params"]:
        if key in final_sd:
            continue
        final_sd[key] = _load_single_tensor(
            _build_param_file_path(pretrained_dir, pre_manifest, key)
        )

    config = AutoConfig.from_pretrained(str(pretrained_dir))
    # CPU-init in bf16 (not meta) so non-overridden buffers (e.g.
    # rotary_emb.inv_freq, which is non-persistent and may not be present in
    # the persisted state_dict) stay valid for save_pretrained.
    # strict=False because final_sd may include legacy persistent keys that
    # the current transformers release no longer registers (e.g. inv_freq
    # was made non-persistent in newer transformers).
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(final_sd, assign=True, strict=False)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print(f"  first 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  first 5 unexpected: {unexpected[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir} ...")
    model.save_pretrained(output_dir)

    tokenizer_dir = (
        Path(args.tokenizer_dir) if args.tokenizer_dir else task_dirs[0] / "finetuned"
    )
    print(f"Copying tokenizer from {tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    args = parse_arguments()
    if args.save is None:
        args.save = "artifacts/checkpoints/wizardlm"
    merge(args)
