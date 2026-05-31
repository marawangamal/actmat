import json
from pathlib import Path

from src.args import parse_arguments
from src.utils import get_prefix, multitask_results_path, resolve_run_dir
from src.vision.eval import eval_single_dataset
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector


VISION_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

args = parse_arguments()
args.save = resolve_run_dir(args)

prefix = get_prefix(args.finetuning_mode)
checkpoint_dir = f"{args.save}/multitask"

print("*" * 100)
print(f"Evaluating MTL checkpoint at {checkpoint_dir}")

task_vector = (
    LinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
    if args.finetuning_mode == "linear"
    else NonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
)
image_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)

eval_datasets = args.eval_datasets or VISION_DATASETS
accuracies = {}

for dataset in eval_datasets:
    for split_name in [dataset, f"{dataset}Val"]:
        print("=" * 100)
        print(f"Evaluating on {split_name}")
        accuracies[split_name] = eval_single_dataset(
            image_encoder, split_name, args
        )["top1"]

val_scores = [
    v for k, v in accuracies.items()
    if k.endswith("Val") and isinstance(v, (int, float))
]
test_scores = [
    v for k, v in accuracies.items()
    if not k.endswith("Val") and isinstance(v, (int, float))
]
accuracies["avg_val"] = (sum(val_scores) / len(val_scores)) if val_scores else None
accuracies["avg_test"] = (sum(test_scores) / len(test_scores)) if test_scores else None

results_file = Path(multitask_results_path(args.results_dir, args.model, prefix))
results_file.parent.mkdir(parents=True, exist_ok=True)

tasks = [
    {
        "alias": k,
        "metrics": {"top1": v, "primary_score": v},
        "task_config": {"primary_metric": "top1"},
    }
    for k, v in accuracies.items()
    if isinstance(v, (int, float)) and k not in ("avg_val", "avg_test")
]
metrics_json = {
    "all_primary_scores": [
        f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
    ],
    "tasks": tasks,
    "model_config": {
        "model": args.model,
        "finetuning_mode": args.finetuning_mode,
        "seed": args.seed,
        "avg_val": accuracies.get("avg_val"),
        "avg_test": accuracies.get("avg_test"),
    },
}
results_file.write_text(json.dumps(metrics_json, indent=2))
print(f"Results saved to {results_file}")
