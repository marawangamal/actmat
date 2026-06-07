"""Evaluate merged RoBERTa-base task vectors on the GLUE benchmark.

Mirrors Twin-Merging's discriminative eval protocol (Lu et al., arXiv:2406.15479):
  - HF `datasets.load_dataset("glue", task)` validation split (validation_matched for MNLI)
  - max_length=128, truncation=True
  - Per-task metric via `evaluate.load("glue", task)`:
        cola -> matthews_correlation
        sst2 / mnli / qnli / rte -> accuracy
        mrpc / qqp -> mean(accuracy, f1)
        stsb -> mean(pearson, spearmanr)

Eval-time model assembly:
  1. Merge ParamFolderTaskVectors via combine_task_vectors -> body delta.
  2. Apply delta to pretrained roberta-base body params.
  3. For each task, instantiate RobertaForSequenceClassification with the
     task's num_labels, load the merged body, and overwrite `model.classifier`
     with the task's saved `classifier_head.pt` (a pickled
     RobertaClassificationHead).

Example:
  python scripts/roberta/eval_task_addition.py \\
      --model=roberta-base --merge-func=sum

Results land in `artifacts/results/{model}-{merge_func}/metrics.json`.
"""

import json
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
)

from src.args import parse_arguments
from src.merging import combine_task_vectors
from src.nlg.task_vectors import (
    ParamFolderTaskVector,
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)
from src.utils import resolve_run_dir

GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]

# Default freeze targets: matches WUDI's `--exclude-param ".*bias.*" ".*LayerNorm.*"
# ".*embeddings.*"` and keeps the merged delta scoped to encoder linear weights.
# Override with --freeze-keys (use --freeze-keys '' to merge everything).
DEFAULT_FREEZE_KEYS = ["bias", "LayerNorm", "embeddings"]

# Twin-Merging's text-field map (arXiv:2406.15479 eval.py)
GLUE_TEXT_FIELDS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
}

# Primary metric per task (Twin-Merging eval.py glue_data_metrics_map)
GLUE_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte":  "accuracy",
    "mrpc": "averaged_scores",
    "qqp":  "averaged_scores",
    "stsb": "averaged_scores",
}


def build_merged_body(args, eval_datasets):
    """Run combine_task_vectors and add the merged delta to the pretrained body.

    Returns a state_dict (roberta.* keys) for the merged body.
    """
    save = args.save
    task_vectors = [
        ParamFolderTaskVector(checkpoint_dir=f"{save}/{t}") for t in eval_datasets
    ]
    merged_tv = combine_task_vectors(
        task_vectors,
        args.merge_func,
        merge_mode=args.merge_mode,
        ignore_keys=args.ignore_keys,
        **args.merge_kwargs,
    )

    pretrained_dir = Path(save) / "pretrained"
    pre_manifest = _load_manifest(pretrained_dir)
    merged_vector = merged_tv.vector
    del merged_tv

    freeze_keys = args.freeze_keys if args.freeze_keys is not None else DEFAULT_FREEZE_KEYS
    body_sd = {}
    frozen = 0
    for key in tqdm(list(merged_vector.keys()), desc="Applying deltas"):
        delta = merged_vector.pop(key)
        pre_t = _load_single_tensor(
            _build_param_file_path(pretrained_dir, pre_manifest, key)
        )
        if freeze_keys and any(s in key for s in freeze_keys):
            # Freeze: keep pretrained value, discard delta
            body_sd[key] = pre_t
            frozen += 1
        else:
            body_sd[key] = (pre_t.float() + delta).to(pre_t.dtype)
    if freeze_keys:
        print(f"Froze {frozen}/{len(body_sd)} keys at pretrained (substrings={freeze_keys})")
    return body_sd


def load_task_model(task_dir: Path, merged_body_sd: dict, device):
    """Instantiate a SeqCls model on the merged body + the task's classifier head."""
    num_labels = json.loads((task_dir / "num_labels.json").read_text())["num_labels"]
    config = AutoConfig.from_pretrained(str(task_dir), num_labels=num_labels)
    model = RobertaForSequenceClassification(config)

    missing, unexpected = model.load_state_dict(merged_body_sd, strict=False)
    # We expect only classifier.* keys to be missing — they're loaded next.
    leaked = [k for k in missing if not k.startswith("classifier.")]
    if leaked:
        print(f"  [warn] missing non-classifier keys in body load: {leaked[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys in body load: {unexpected[:5]}")

    classifier = torch.load(
        task_dir / "classifier_head.pt", map_location="cpu", weights_only=False
    )
    model.classifier = classifier
    return model.to(device).eval()


def eval_task(task: str, task_dir: Path, model, batch_size: int, device):
    tokenizer = AutoTokenizer.from_pretrained(str(task_dir))
    s1, s2 = GLUE_TEXT_FIELDS[task]

    split = "validation_matched" if task == "mnli" else "validation"
    ds = datasets.load_dataset("glue", task, split=split)

    def tok(ex):
        return tokenizer(
            ex[s1],
            ex[s2] if s2 else None,
            truncation=True,
            max_length=128,
        )

    ds = ds.map(tok, batched=True, remove_columns=[c for c in ds.column_names if c not in ("label",)])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    metric = evaluate.load("glue", task)
    is_regression = task == "stsb"

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{task} eval", leave=False):
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(**batch).logits.float().cpu()
            preds = logits.squeeze(-1) if is_regression else logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=labels)

    raw = metric.compute()
    if len(raw) > 1:
        raw["averaged_scores"] = float(np.mean(list(raw.values())))
    else:
        raw["averaged_scores"] = float(next(iter(raw.values())))
    return raw


def main():
    args = parse_arguments()
    args.save = resolve_run_dir(args)
    args.merge_mode = args.merge_mode or "d"

    merge_name = args.merge_func
    eval_datasets = args.eval_datasets or list(GLUE_TASKS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    merge_mode_str = f"-{args.merge_mode}" if args.merge_mode != "d" else ""
    results_file = Path(
        f"{args.results_dir}/{args.model}-{merge_name}{merge_mode_str}/metrics.json"
    )
    if results_file.exists() and not args.overwrite:
        print(f"Skipping: {results_file} already exists (use --overwrite to rerun)")
        return

    print("*" * 100)
    print(f"Evaluating GLUE merge ({merge_name}, mode={args.merge_mode}) on {eval_datasets}")
    print("*" * 100)

    # 1. Merge + apply to pretrained body
    merged_body_sd = build_merged_body(args, eval_datasets)

    # 2. Per-task: swap classifier head, run inference, compute metric
    tasks = []
    primary_scores = []
    for task in eval_datasets:
        task_dir = Path(args.save) / task
        print(f"\n>>> {task}")
        model = load_task_model(task_dir, merged_body_sd, device)
        raw = eval_task(task, task_dir, model, args.batch_size, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        primary_key = GLUE_PRIMARY_METRIC[task]
        primary = float(raw[primary_key])
        primary_scores.append(primary)
        tasks.append(
            {
                "alias": task,
                "metrics": {**{k: float(v) for k, v in raw.items()}, "primary_score": primary},
                "task_config": {"primary_metric": primary_key},
            }
        )
        print(f"    {task}: {primary_key}={primary:.4f} (raw={raw})")

    avg = float(np.mean(primary_scores))
    print("\n" + "=" * 80)
    print(f"Average primary score across {len(eval_datasets)} tasks: {avg:.4f}")

    metrics_json = {
        "all_primary_scores": [
            f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
        ],
        "average_primary_score": avg,
        "tasks": tasks,
        "model_config": {
            "model": args.model,
            "merge_func": merge_name,
            "merge_mode": args.merge_mode,
            "merge_kwargs": args.merge_kwargs,
            "eval_datasets": eval_datasets,
            "seed": args.seed,
        },
    }
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(metrics_json, indent=2))
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
