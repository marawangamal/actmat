import json
from pathlib import Path

from src.args import parse_arguments
from src.utils import (
    expert_dir,
    experts_results_path,
    get_prefix,
    pretrained_results_path,
    resolve_run_dir,
)
from src.vision.eval import eval_single_dataset
from src.vision.linearize import LinearizedImageEncoder
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
args.save = resolve_run_dir(args)

prefix = get_prefix(args.finetuning_mode)

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
elif args.finetuning_mode == "lora":
    print("Evaluating LoRA FT models.")

eval_datasets = args.eval_datasets or [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

# Trajectory analysis: optionally load `checkpoint_{step}.pt` instead of `finetuned.pt`.
ckpt_step = getattr(args, "checkpoint_step", None)
finetuned_filename = (
    f"checkpoint_{ckpt_step}.pt"
    if ckpt_step is not None and ckpt_step != "final"
    else None
)

for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    checkpoint_dir = expert_dir(args.save, dataset)

    try:
        task_vector = (
            LinearizedTaskVector(
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                finetuned_filename=finetuned_filename,
            )
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                finetuned_filename=finetuned_filename,
            )
        )
    except FileNotFoundError as e:
        print(f"{e}\n\nError: Could not find checkpoint in {checkpoint_dir}.")
        continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=0.0)
    elif args.finetuning_mode in ("standard", "linear", "lora"):
        image_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )
    else:
        raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]


# NOTE: Uncomment to evaluate zero-shot accuracy on ImageNet
# if args.finetuning_mode == "none":
#     # Evaluate zero-shot accuracy on ImageNet
#     for split in ["ImageNetVal", "ImageNet"]:
#         accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Add averages:
val_scores = [
    v
    for k, v in accuracies.items()
    if k.endswith("Val") and isinstance(v, (int, float))
]
test_scores = [
    v
    for k, v in accuracies.items()
    if (not k.endswith("Val")) and isinstance(v, (int, float))
]
accuracies["avg_val"] = (sum(val_scores) / len(val_scores)) if val_scores else None
accuracies["avg_test"] = (sum(test_scores) / len(test_scores)) if test_scores else None

# Save results (pretrained/zero-shot baseline parallel to eval_task_addition.py).
if args.finetuning_mode == "none":
    results_file = Path(pretrained_results_path(args.results_dir, args.model, prefix, group=args.group))
else:
    results_file = Path(experts_results_path(args.results_dir, args.model, prefix, group=args.group))
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
