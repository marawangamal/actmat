"""Collect lm-evaluation-harness results for Twin-Merging benchmarks.

Usage:
    python scripts/twin/collect_results.py --dirs results/twin/eigcov results/twin/mean results/twin/tsv results/twin/isoc
"""

import argparse
import json
import sys
from pathlib import Path

# Benchmark display name -> (lm-eval task key, metric key)
# Adjust task/metric keys after verifying with: lm_eval --tasks list
BENCHMARKS = {
    "BBQ": ("bbq", "acc"),
    "CNN/DailyMail": ("cnn_dailymail", "rouge2"),
    "MMLU": ("mmlu", "acc"),
    "TruthfulQA": ("truthfulqa_mc2", "acc"),
}

DISPLAY_ORDER = ["BBQ", "CNN/DailyMail", "MMLU", "TruthfulQA"]


def load_results(results_dir: Path) -> dict[str, float]:
    """Load lm-eval-harness results from a directory.

    lm_eval writes results to <output_path>/<model_name>/results_*.json.
    The JSON has structure: {"results": {"task_name": {"metric,filter": value, ...}}}.
    """
    # Find the results JSON — lm_eval may nest under a model subdir.
    candidates = list(results_dir.rglob("results_*.json"))
    if not candidates:
        # Fallback: look for results.json
        candidates = list(results_dir.rglob("results.json"))
    if not candidates:
        print(f"Warning: no results JSON found in {results_dir}", file=sys.stderr)
        return {}

    # Use the most recent file if multiple exist.
    results_file = max(candidates, key=lambda p: p.stat().st_mtime)
    data = json.loads(results_file.read_text())
    task_results = data.get("results", {})

    scores: dict[str, float] = {}
    for display_name, (task_key, metric_key) in BENCHMARKS.items():
        # lm-eval results keys can be exact or prefixed; try exact first.
        task_data = task_results.get(task_key, {})
        if not task_data:
            # Try matching with prefix (e.g., "mmlu" matches group average).
            for k, v in task_results.items():
                if k.startswith(task_key) and isinstance(v, dict):
                    task_data = v
                    break
        if not task_data:
            continue

        # lm-eval stores metrics as "metric,filter_name" keys.
        value = None
        for k, v in task_data.items():
            if k.startswith(metric_key) and isinstance(v, (int, float)):
                value = v
                break
        if value is not None:
            scores[display_name] = value

    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect lm-eval-harness results for Twin-Merging benchmarks."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more lm-eval output directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_results: dict[str, dict[str, float]] = {}
    for d in args.dirs:
        p = Path(d)
        name = p.name
        all_results[name] = load_results(p)

    methods = list(all_results.keys())
    col_w = max(15, *(len(m) + 2 for m in methods))

    header = f"{'Benchmark':<15}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    for bench in DISPLAY_ORDER:
        row = f"{bench:<15}"
        for m in methods:
            val = all_results[m].get(bench)
            row += f"{val:{col_w}.4f}" if val is not None else f"{'--':>{col_w}}"
        print(row)

    print("-" * len(header))
    row = f"{'Average':<15}"
    for m in methods:
        vals = [all_results[m][b] for b in DISPLAY_ORDER if b in all_results[m]]
        row += f"{sum(vals)/len(vals):{col_w}.4f}" if vals else f"{'--':>{col_w}}"
    print(row)


if __name__ == "__main__":
    main()
