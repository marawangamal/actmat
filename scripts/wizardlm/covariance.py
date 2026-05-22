"""Collect per-layer activation covariances for WizardLM-13B / WizardMath-13B /
llama-2-13b-code-alpaca on each expert's training data.

Runs forward passes (no training) while forward hooks from src/covariance.py
capture activation statistics. Each capability produces a covariance .pt file
saved inside its task directory, where ParamFolderTaskVector auto-discovers it.

Following the vision/language convention (and unlike OLMo, where pretrained
and RL-Zero activations come from the same distribution), the **finetuned**
expert is loaded for covariance — its activation statistics are what
RegMean/ACTMat actually want.

Usage:
    export PYTHONPATH="$PYTHONPATH:$PWD"

    # Single capability
    python scripts/wizardlm/covariance.py --capability lm \\
        --save artifacts/checkpoints/Llama-2-13b-wizardlm

    # All capabilities
    python scripts/wizardlm/covariance.py --capability all \\
        --save artifacts/checkpoints/Llama-2-13b-wizardlm
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.args import parse_arguments
from src.covariance import register_hooks

PRETRAINED_TOKENIZER = "meta-llama/Llama-2-13b-hf"
MAX_SEQ_LEN = 512

# Capability -> (HF dataset id, dataset key in this script's dispatch).
# Note: the WizardMath team did not publish their training set, so we use
# TIGER-Lab/MathInstruct as a covariance proxy (instruction-tuning math data
# of similar provenance). This only affects stats-based mergers
# (RegMean/Fisher/ACTMat); sum/mean/ties/dare/tsv/isoc are unaffected.
CAPABILITY_DATASETS = {
    "LM": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
    "Math": "TIGER-Lab/MathInstruct",
    "Code": "sahil2801/CodeAlpaca-20k",
}

# Normalize CLI capability tokens (lowercase) to the directory names used by
# scripts/wizardlm/download_models.sh.
CAPABILITY_ALIASES = {"lm": "LM", "math": "Math", "code": "Code"}


def _format_prompt(example: dict, capability: str) -> str:
    """Build the user prompt for one example from the expert's training set.

    Wrapped in the Llama-2 [INST] ... [/INST] template since the base tokenizer
    has no chat template.
    """
    if capability == "LM":
        # WizardLM_evol_instruct_V2_196k: list of {"from": "human"/"gpt", "value": ...}
        msgs = example["conversations"]
        instr = next((m["value"] for m in msgs if m.get("from") == "human"), "")
    elif capability == "Math":
        # TIGER-Lab/MathInstruct: {"instruction": ..., "output": ...}
        instr = example["instruction"]
    elif capability == "Code":
        # sahil2801/CodeAlpaca-20k: {"instruction": ..., "input": ..., "output": ...}
        instr = example["instruction"]
        if example.get("input"):
            instr = f"{instr}\n\n{example['input']}"
    else:
        raise ValueError(f"Unknown capability: {capability}")
    return f"[INST] {instr.strip()} [/INST]"


def collect_covariance(capability, args):
    print(f"\n{'=' * 60}")
    print(f"Collecting covariance: {capability}")
    print(f"{'=' * 60}")

    run_dir = os.path.join(args.save, capability)
    cov_path = os.path.join(run_dir, "covariance.pt")

    if os.path.exists(cov_path) and not args.overwrite:
        print(f"  Skipping {capability} — {cov_path} already exists")
        return

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.cache_dir)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_TOKENIZER, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        keys = list(examples.keys())
        n = len(examples[keys[0]])
        rows = [{k: examples[k][i] for k in keys} for i in range(n)]
        texts = [_format_prompt(r, capability) for r in rows]
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

    # Load the FINETUNED expert (vision/language convention — not OLMo's base).
    ft_dir = os.path.join(run_dir, "finetuned")
    print(f"Loading finetuned model from {ft_dir} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        ft_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    cobjs, handles = register_hooks(
        model,
        cov_device="cpu",
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        batch_first=True,
    )

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

    for h in handles:
        h.remove()

    saveable = {}
    for name, cobj in cobjs.items():
        saveable[name] = cobj.cov.cpu()
        saveable[f"{name}_n"] = cobj.n

    os.makedirs(run_dir, exist_ok=True)
    torch.save(saveable, cov_path)
    print(f"Saved covariances ({len(cobjs)} layers) to {cov_path}")


def main():
    args = parse_arguments()
    if args.save is None:
        args.save = "artifacts/checkpoints/Llama-2-13b-wizardlm"

    cap = (args.capability or "all").lower()
    if cap == "all":
        caps = list(CAPABILITY_DATASETS.keys())
    else:
        if cap not in CAPABILITY_ALIASES:
            raise ValueError(
                f"Unknown --capability {args.capability!r}; expected one of "
                f"{list(CAPABILITY_ALIASES.keys())} or 'all'."
            )
        caps = [CAPABILITY_ALIASES[cap]]

    for cap_name in caps:
        collect_covariance(cap_name, args)


if __name__ == "__main__":
    main()
