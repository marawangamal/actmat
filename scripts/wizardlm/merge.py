"""Merge WizardLM-13B param-folder task vectors into a HuggingFace checkpoint.

Reproduces the DARE paper Fig. 1 (right) merging of WizardLM-13B,
WizardMath-13B and llama-2-13b-code-alpaca. Uses ParamFolderTaskVector
for lazy, memory-bounded merging — only one parameter is loaded per model
at a time.

Usage:
    python scripts/wizardlm/merge.py \\
        --save artifacts/checkpoints/Llama-2-13b-wizardlm \\
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
        if not (td / "zeroshot").exists() or not (td / "finetuned").exists():
            raise FileNotFoundError(
                f"{td} must contain zeroshot/ and finetuned/ subdirectories. "
                "Run scripts/wizardlm/download_models.sh first."
            )

    pretrained_dir = (task_dirs[0] / "zeroshot").resolve()
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

    config = AutoConfig.from_pretrained(str(pretrained_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(final_sd, assign=True)

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
        args.save = "artifacts/checkpoints/Llama-2-13b-wizardlm"
    merge(args)
