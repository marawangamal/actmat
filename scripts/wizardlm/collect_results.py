"""Collect olmes evaluation results for the WizardLM-13B merging benchmark.

Usage:
    python scripts/wizardlm/collect_results.py \\
        --dirs artifacts/results/Llama-2-13b-wizardlm-sum \\
               artifacts/results/Llama-2-13b-wizardlm-dare \\
               artifacts/results/Llama-2-13b-wizardlm-actmat
"""

import argparse
import json
from pathlib import Path

from src.results_db import append_result, args_to_dict, make_run_hash, record_exists


def load_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return {alias: {metric_name: value}} from metrics.json."""
    f = results_dir / "metrics.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    tasks = {}
    for task in data.get("tasks", []):
        alias = task["alias"]
        metrics = {
            k: v
            for k, v in task.get("metrics", {}).items()
            if isinstance(v, (int, float))
        }
        if metrics:
            tasks[alias] = metrics
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument(
        "--dry", action="store_true", help="Print records that would be logged."
    )
    parser.add_argument(
        "--results-db", default="results-tracked/results-wizardlm.jsonl"
    )
    args = parser.parse_args()

    all_results = {Path(d).name: load_results(Path(d)) for d in args.dirs}
    all_tasks = sorted(set().union(*(s.keys() for s in all_results.values())))
    methods = list(all_results.keys())

    task_metrics: dict[str, list[str]] = {}
    for task in all_tasks:
        keys = set()
        for m in methods:
            keys.update(all_results[m].get(task, {}).keys())
        task_metrics[task] = sorted(keys - {"primary_score", "extra_metrics"})

    row_labels = []
    for task in all_tasks:
        for metric in task_metrics[task]:
            row_labels.append((task, metric))

    col_w = max(15, *(len(m) + 2 for m in methods))
    label_w = max((len(f"{t}/{m}") for t, m in row_labels), default=10) + 2
    print(f"{'Task/Metric':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in methods))
    for task, metric in row_labels:
        label = f"{task}/{metric}"
        print(
            f"{label:<{label_w}}"
            + "".join(
                f"{all_results[m].get(task, {}).get(metric, float('nan')):{col_w}.3f}"
                for m in methods
            )
        )

    averages = {}
    for m in methods:
        vals = [
            all_results[m][t]["primary_score"]
            for t in all_tasks
            if "primary_score" in all_results[m].get(t, {})
        ]
        if vals:
            averages[m] = sum(vals) / len(vals)
    print(
        f"{'Average (primary)':<{label_w}}"
        + "".join(f"{averages.get(m, float('nan')):{col_w}.3f}" for m in methods)
    )

    if args.log or args.dry:
        for method in methods:
            tasks = all_results[method]
            args.merge_func = method.rsplit("-", 1)[-1]
            run_hash = make_run_hash(
                "collect_results_wizardlm",
                args,
                ignore={"log", "dry", "results_db"},
            )
            if not args.dry and record_exists(args.results_db, run_hash):
                print(f"Skipping {method}: already logged")
                continue
            record = {**args_to_dict(args), "script": "collect_results_wizardlm"}
            for task, metrics in tasks.items():
                for k, v in metrics.items():
                    if k not in ("primary_score", "extra_metrics"):
                        record[f"test_{task}/{k}"] = v
            if method in averages:
                record["test_avg"] = averages[method]
            if args.dry:
                print(f"\n[dry] {method}:")
                print(json.dumps(record, indent=2))
            else:
                append_result(args.results_db, record, run_hash)


if __name__ == "__main__":
    main()
