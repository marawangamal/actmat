import argparse
import copy
import csv
import os
import os.path as osp
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from scripts.t5.common import T5_DATASETS, parse_csv  # noqa: E402
from src.language.datasets.batcher import Batcher  # noqa: E402
from src.language.datasets.dataset_readers import get_datasetReader  # noqa: E402
from src.language.datasets.pytorch_dataset import PytorchDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 1D T5 loss landscape between two checkpoints. "
            "The loss is cross-entropy over multiple-choice log-probability scores."
        )
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--datasets", type=parse_csv, default=",".join(T5_DATASETS))
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--alpha-start", type=float, default=-0.5)
    parser.add_argument("--alpha-end", type=float, default=1.5)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-val-samples", type=int, default=32)
    parser.add_argument("--data-location", default="data")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "huggingface"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/agents/loss-landscape",
    )
    parser.add_argument("--output-prefix", default="landscape")
    parser.add_argument("--no-plot", action="store_true", default=False)
    return parser.parse_args()


def build_eval_batches(tokenizer, dataset_name, args):
    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": args.num_val_samples,
        "max_datapoints_per_dataset_without_templates": None,
    }
    reader = get_datasetReader(dataset_name, dataset_kwargs)
    create_dataset = lambda dataset: PytorchDataset(dataset, tokenizer, args.device)
    batcher = Batcher(
        reader,
        create_dataset,
        train_batchSize=None,
        eval_batchSize=args.batch_size,
        world_size=1,
        device=0,
    )
    return list(batcher.get_evalBatches(args.eval_split, template_idx=0))


def interpolate_state_dict(model, state_a, state_b, alpha):
    current = model.state_dict()
    interpolated = {}
    for key, value_a in state_a.items():
        value_b = state_b.get(key)
        if (
            value_b is not None
            and torch.is_floating_point(value_a)
            and torch.is_floating_point(value_b)
            and value_a.shape == value_b.shape
        ):
            interpolated[key] = torch.lerp(value_a, value_b.to(value_a.dtype), alpha)
        else:
            interpolated[key] = value_a

    for key, value in current.items():
        if key not in interpolated:
            interpolated[key] = value

    model.load_state_dict(interpolated, strict=False)


def eval_choice_loss(model, batches, args):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, leave=False)):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            with torch.autocast(device_type=args.device_type, dtype=torch.bfloat16):
                scores, _, _ = model.compute_logProb_ofAllChoices(
                    batch["input_ids"],
                    batch["input_mask"],
                    batch["all_choices_ids"],
                    batch["all_choices_mask"],
                    length_normalization=False,
                )

            labels = batch["lbl"].to(scores.device).long()
            loss = F.cross_entropy(scores.float(), labels, reduction="sum")
            predictions = scores.argmax(dim=1)

            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.numel()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
    }


def plot_results(csv_path, png_path, label_a, label_b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    avg = df[df["dataset"] == "avg"]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(avg["alpha"], avg["loss"], marker="o", color="#1f77b4", label="loss")
    ax1.set_xlabel(f"alpha: {label_a} -> {label_b}")
    ax1.set_ylabel("choice CE loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax1.axvline(1.0, color="0.6", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    ax2.plot(
        avg["alpha"],
        avg["accuracy"],
        marker="s",
        color="#d62728",
        label="accuracy",
    )
    ax2.set_ylabel("accuracy", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.device_type = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device_type)

    model_a = torch.load(args.checkpoint_a, map_location="cpu", weights_only=False)
    model_b = torch.load(args.checkpoint_b, map_location="cpu", weights_only=False)
    state_a = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
    state_b = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}
    del model_b

    model = copy.deepcopy(model_a).to(args.device)
    model.eval()
    del model_a

    batches_by_dataset = {
        dataset: build_eval_batches(model.tokenizer, dataset, args)
        for dataset in args.datasets
    }

    alphas = torch.linspace(args.alpha_start, args.alpha_end, args.num_points).tolist()
    rows = []
    for alpha in alphas:
        print(f"Evaluating alpha={alpha:.4f}", flush=True)
        interpolate_state_dict(model, state_a, state_b, alpha)
        per_alpha = []
        for dataset, batches in batches_by_dataset.items():
            metrics = eval_choice_loss(model, batches, args)
            row = {
                "alpha": alpha,
                "dataset": dataset,
                **metrics,
            }
            rows.append(row)
            per_alpha.append(metrics)
            print(
                f"  {dataset}: loss={metrics['loss']:.4f}, "
                f"acc={metrics['accuracy']:.4f}, n={metrics['num_examples']}",
                flush=True,
            )

        total_n = sum(x["num_examples"] for x in per_alpha)
        rows.append(
            {
                "alpha": alpha,
                "dataset": "avg",
                "loss": sum(x["loss"] * x["num_examples"] for x in per_alpha)
                / total_n,
                "accuracy": sum(
                    x["accuracy"] * x["num_examples"] for x in per_alpha
                )
                / total_n,
                "num_examples": total_n,
            }
        )

    csv_path = osp.join(args.output_dir, f"{args.output_prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["alpha", "dataset", "loss", "accuracy", "num_examples"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {csv_path}")

    if not args.no_plot:
        png_path = osp.join(args.output_dir, f"{args.output_prefix}.png")
        plot_results(csv_path, png_path, args.label_a, args.label_b)
        print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
