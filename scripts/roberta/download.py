"""Download RoBERTa-base GLUE experts from Twin-Merging's HF release and stage
them as param-folder checkpoints (the layout consumed by ParamFolderTaskVector).

Source: https://huggingface.co/lu-vae/roberta-glue
Paper:  Lu et al., "Twin-Merging" (arXiv:2406.15479)

Layout written to disk (default: artifacts/checkpoints/roberta-base/):

  pretrained/                          # shared roberta-base body param-folder
    param_manifest.json
    params/<key>__<hash>.safetensors   # one file per tensor (197 of them)
  <task>/
    pretrained/  -> symlink → ../pretrained
    finetuned/
      param_manifest.json
      params/<key>__<hash>.safetensors
    classifier_head.pt                 # per-task SeqCls head (output dims differ)
    num_labels.json                    # {"num_labels": int, "task": str}
    config.json, tokenizer.json, ...   # tokenizer / config files

Body vs. head split: each task fine-tunes the same RoBERTa body but uses a
task-specific classifier head with different output dims. We merge only the
body — heads are kept separately and re-attached at eval time.

Usage:
  python scripts/roberta/download.py
  python scripts/roberta/download.py --output-dir /path/to/dir
  python scripts/roberta/download.py --tasks cola,sst2  # subset
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file as save_safetensors_file
from transformers import AutoModel, AutoConfig

GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
HF_REPO = "lu-vae/roberta-glue"  # used only for --variant base when no --snapshot-dir given

# Per-variant defaults: HF base-model id + the per-task subdirectory name used by the
# Twin-Merging / WUDI release ("roberta-{base,large}_lr1e-05").
VARIANT_DEFAULTS = {
    "base":  ("FacebookAI/roberta-base",  "roberta-base_lr1e-05"),
    "large": ("FacebookAI/roberta-large", "roberta-large_lr1e-05"),
}

# Non-weight files we copy verbatim from each task's snapshot dir.
SIDE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def split_body_and_head(state_dict):
    """Split a SeqCls state_dict into body (roberta.*) and head (classifier.*)."""
    body, head = {}, {}
    for k, v in state_dict.items():
        if k.startswith("roberta."):
            body[k] = v
        elif k.startswith("classifier."):
            head[k] = v
        else:
            print(f"  [warn] unrecognized key prefix, dropping: {k}")
    return body, head


def load_full_state_dict(src_dir: Path) -> dict:
    """Load the full SeqCls state dict from either safetensors or pytorch_model.bin."""
    safetensors_path = src_dir / "model.safetensors"
    bin_path = src_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        return load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {src_dir}")


def _safe_file_stem(param_name: str) -> str:
    """Match the naming convention used by scripts/wizardlm/save_model_param_folder.py."""
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def write_param_folder(folder: Path, model_id: str, state_dict: dict):
    """Write {folder}/params/*.safetensors + {folder}/param_manifest.json."""
    params_dir = folder / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    manifest_params: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        tensor = tensor.detach().cpu().contiguous()
        stem = _safe_file_stem(key)
        filename = f"{stem}.safetensors"
        save_safetensors_file({"tensor": tensor}, str(params_dir / filename))
        manifest_params[key] = {
            "file": filename,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    manifest = {
        "format": "safetensors",
        "model_id": model_id,
        "params": manifest_params,
    }
    (folder / "param_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def stage_task(task: str, snapshot_root: Path, output_root: Path, subdir: str):
    """Convert one task's HF snapshot into a param-folder task dir.

    Returns the body's state dict (so the caller can reconcile key sets across
    tasks before saving the shared pretrained body).
    """
    src = snapshot_root / task / subdir
    dst = output_root / task
    dst.mkdir(parents=True, exist_ok=True)

    full_sd = load_full_state_dict(src)
    body, head = split_body_and_head(full_sd)

    # Write the finetuned body as a param-folder
    write_param_folder(dst / "finetuned", model_id=f"{HF_REPO}#{task}", state_dict=body)
    print(f"  [{task}] wrote finetuned/ param-folder ({len(body)} tensors)")

    # Prefer the classifier head as packaged separately by the authors when present;
    # fall back to whatever we sliced out of model.safetensors / pytorch_model.bin.
    sep_head_path = src / "classifier_head.pt"
    if sep_head_path.exists():
        shutil.copyfile(sep_head_path, dst / "classifier_head.pt")
        print(f"  [{task}] copied classifier_head.pt from snapshot")
    else:
        torch.save(head, dst / "classifier_head.pt")
        print(f"  [{task}] saved classifier head ({len(head)} tensors)")

    # Copy tokenizer + config side-files
    for fname in SIDE_FILES:
        sp = src / fname
        if sp.exists():
            shutil.copyfile(sp, dst / fname)

    # Record num_labels for later re-instantiation of the SeqCls head
    cfg = AutoConfig.from_pretrained(str(src))
    (dst / "num_labels.json").write_text(
        json.dumps({"task": task, "num_labels": cfg.num_labels}, indent=2)
    )
    print(f"  [{task}] num_labels = {cfg.num_labels}")

    return body


def save_pretrained_folder(folder: Path, cache_dir: str, key_filter: set, base_model: str):
    """Save the pretrained roberta body as a param-folder, restricted to key_filter.

    `key_filter` is the set of body keys that appear in the fine-tuned checkpoints
    (which lack the pooler — its slot is taken by the SeqCls classifier head).
    Restricting here keeps the pretrained and finetuned key sets aligned for
    `_TaskVector` arithmetic.
    """
    if (folder / "param_manifest.json").exists():
        print(f">>> Pretrained body already at {folder} — skipping")
        return
    print(f">>> Downloading pretrained body: {base_model}")
    model = AutoModel.from_pretrained(base_model, cache_dir=cache_dir)
    # AutoModel returns a RobertaModel; its state_dict keys are bare (no 'roberta.' prefix).
    body = {f"roberta.{k}": v for k, v in model.state_dict().items()}
    dropped = sorted(k for k in body if k not in key_filter)
    body = {k: v for k, v in body.items() if k in key_filter}
    if dropped:
        print(f"    Dropped {len(dropped)} keys not in finetuned body: {dropped}")
    write_param_folder(folder, model_id=base_model, state_dict=body)
    print(f"    Wrote {len(body)} pretrained tensors to {folder}")


def link_pretrained(task_dir: Path, shared_pretrained: Path):
    """Create / refresh the pretrained -> ../pretrained symlink inside a task dir."""
    link = task_dir / "pretrained"
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)
    link.symlink_to(os.path.relpath(shared_pretrained, task_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_DEFAULTS),
        default="base",
        help="roberta variant — controls --base-model and --subdir defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination root. Default: artifacts/checkpoints/roberta-{variant}.",
    )
    parser.add_argument(
        "--tasks",
        type=lambda x: x.split(","),
        default=None,
        help=f"Comma-separated subset of {GLUE_TASKS}. Default: all.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME"),
        help="HF cache dir. Defaults to $HF_HOME.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Skip the HF download and read from this existing snapshot dir.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HF id for the pretrained body. Default: variant-specific.",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Per-task subdirectory name in the snapshot. Default: variant-specific.",
    )
    args = parser.parse_args()

    base_default, subdir_default = VARIANT_DEFAULTS[args.variant]
    base_model = args.base_model or base_default
    subdir = args.subdir or subdir_default
    output_root = Path(args.output_dir or f"artifacts/checkpoints/roberta-{args.variant}")

    tasks = args.tasks or GLUE_TASKS
    bad = [t for t in tasks if t not in GLUE_TASKS]
    if bad:
        print(f"ERROR: unknown tasks {bad}; choose from {GLUE_TASKS}", file=sys.stderr)
        sys.exit(2)

    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot the HF repo (cached after the first call) unless overridden
    if args.snapshot_dir is None:
        if args.variant != "base":
            print(
                f"ERROR: --variant={args.variant} has no HF release; "
                "pass --snapshot-dir pointing at a local Drive download.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f">>> Downloading {HF_REPO}")
        snapshot_root = Path(
            snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=[f"{t}/**" for t in tasks],
                cache_dir=args.cache_dir,
            )
        )
    else:
        snapshot_root = Path(args.snapshot_dir)
    print(f"    Snapshot at {snapshot_root}")
    print(f"    Per-task subdir: {subdir}")
    print(f"    Pretrained body: {base_model}")
    print(f"    Output: {output_root}")

    # 2. Stage each task — gives us the canonical body key set
    body_key_sets = {}
    for task in tasks:
        print(f">>> Staging {task}")
        body = stage_task(task, snapshot_root, output_root, subdir)
        body_key_sets[task] = set(body.keys())

    canonical_keys = next(iter(body_key_sets.values()))
    for task, keys in body_key_sets.items():
        if keys != canonical_keys:
            extra = sorted(keys - canonical_keys)
            missing = sorted(canonical_keys - keys)
            print(f"  [warn] {task} body keys diverge: +{extra} -{missing}")

    # 3. Save shared pretrained body, then symlink it into each task dir
    shared_pretrained = output_root / "pretrained"
    save_pretrained_folder(shared_pretrained, args.cache_dir, key_filter=canonical_keys, base_model=base_model)
    for task in tasks:
        link_pretrained(output_root / task, shared_pretrained)

    print("\nDone. Layout:")
    print(f"  {output_root}/")
    print(f"    pretrained/  (param-folder, shared)")
    for task in tasks:
        print(f"    {task}/  (pretrained -> ../pretrained, finetuned/, classifier_head.pt)")


if __name__ == "__main__":
    main()
