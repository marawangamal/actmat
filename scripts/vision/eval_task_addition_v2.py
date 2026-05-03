"""
Evaluate and merge vision task vectors across multiple datasets.

Performs hyperparameter search, merges task vectors (e.g., sum, mean), evaluates
on validation and test splits, and saves accuracy metrics to a JSON file.
"""

import itertools
import json
import os
import os.path as osp

from src import mhap, mhas
from src.args import parse_arguments
from src.vision.eval import evaluate_v2
from src.merging import combine_task_vectors
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector


def main(args):

    # HACK: override args
    args.save = f"checkpoints/{args.model}" if args.save is None else args.save
    eval_datasets = args.eval_datasets or [
        "SUN397",
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SVHN",
    ]
    validation_datasets = [dataset + "Val" for dataset in eval_datasets]
    results_path = f"{args.results_dir}/{args.model}-{args.merge_func}-{args.merge_mode}/{args.finetuning_mode}-metrics.json"
    pretrained_dir = f"{args.save}/{eval_datasets[-1]}Val"

    if osp.exists(results_path) and not args.overwrite:
        print(f"Skipping: {results_path} already exists (use --overwrite to rerun)")
        exit(0)

    task_vectors = []
    for i, dataset in enumerate(eval_datasets):
        checkpoint_dir = f"{args.save}/{dataset}Val"
        if args.finetuning_mode == "linear":
            task_vectors.append(
                LinearizedTaskVector(
                    checkpoint_dir=checkpoint_dir,
                    prefix=args.prefix,
                    save_pt=args.merge_mode == "w" and i == 0,
                )
            )
        else:
            task_vectors.append(
                NonLinearTaskVector(
                    checkpoint_dir=checkpoint_dir,
                    prefix=args.prefix,
                    save_pt=args.merge_mode == "w" and i == 0,
                )
            )
        print(f"Task vector {dataset} loaded")

    # For use with RegMean and Projected RegMean.
    #   i)  for projected regmean, mhap will package together orthgonally invariant matrices.
    #   ii) for regmean, mhas will use linear modules for all attention operations to collect covariances.
    if args.mha is not None:
        print(f"Mapping task vectors dicts to {args.mha} MHA mode")
        copy_fn = {
            "packed": mhap.copy_from_pytorch_state_dict,
            "split": mhas.copy_from_pytorch_state_dict,
        }[args.mha]
        task_vectors = [t.map(copy_fn) for t in task_vectors]

    # Build HP grid
    hpo = args.hpo or {}
    hp_names = list(hpo.keys())
    hp_value_lists = list(hpo.values())
    hp_combos = (
        [dict(zip(hp_names, combo)) for combo in itertools.product(*hp_value_lists)]
        if hp_names
        else [{}]
    )

    # ========================================================================================
    # Phase 1: HP grid search on eval-val-split (optionally capped by eval-val-max-batches).
    # ========================================================================================
    print("=" * 100)
    print("PHASE 1: VAL SPLIT")
    print("=" * 100)
    best_val_score = -float("inf")
    best_merge_kwargs = {}
    # HACK: use subset of train split for HP search
    args.eval_split = "train"

    if len(hp_combos) <= 1:
        best_merge_kwargs = hp_combos[0] if hp_combos else {}
        print(f"SKIPPED (single HP combo: {best_merge_kwargs})")
    else:
        print(f"Num combos: {len(hp_combos)} | Max batches: {args.eval_max_batches}")

        for merge_kwargs in hp_combos:
            task_vector = combine_task_vectors(
                task_vectors, args.merge_func, **merge_kwargs
            )
            metrics = evaluate_v2(
                task_vector, pretrained_dir, validation_datasets, args
            )
            score = metrics["avg_top1"]
            print(f"  {merge_kwargs} -> avg_top1={score:.4f}")
            if score > best_val_score:
                best_val_score = score
                best_merge_kwargs = merge_kwargs

    print(f"Best merge HP (from phase 1): {best_merge_kwargs}")

    # ========================================================================================
    # Phase 2: evaluate at best HP combo on the test split (use all batches).
    # ========================================================================================
    print("=" * 100)
    print("PHASE 2: TEST SPLIT")
    print("=" * 100)
    # HACK: use subset of train split for HP search
    args.eval_split = "test"

    test_metrics = evaluate_v2(task_vector, pretrained_dir, eval_datasets, args)

    print("=" * 100)
    print(f"Test accuracy: {test_metrics['avg_top1']}")

    # Build olmes-style metrics.json
    tasks = []
    for dataset in eval_datasets:
        top1 = test_metrics[f"{dataset}:top1"]
        tasks.append(
            {
                "alias": dataset,
                "metrics": {"top1": top1, "primary_score": top1},
                "task_config": {"primary_metric": "top1"},
            }
        )

    metrics_json = {
        "all_primary_scores": [
            f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
        ],
        "tasks": tasks,
        "model_config": {
            "model": args.model,
            "merge_func": args.merge_func,
            "finetuning_mode": args.finetuning_mode,
            "seed": args.seed,
            "mha": args.mha,
            "optimal_merge_hp": best_merge_kwargs,
        },
    }

    # Save results
    os.makedirs(osp.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
