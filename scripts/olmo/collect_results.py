"""Collect olmes evaluation results.

Usage:
    python scripts/olmo/collect_results.py --dirs results-olmo-mean results-olmo-eigcov
"""

import argparse
import json
from pathlib import Path

from src.results_db import append_result, args_to_dict, make_run_hash, record_exists


def load_results(results_dir: Path) -> dict[str, float]:
    scores = {}
    for f in sorted(results_dir.glob("*-metrics.json")):
        data = json.loads(f.read_text())
        alias = data["task_config"]["metadata"]["alias"]
        score = data.get("metrics", {}).get("primary_score")
        if score is not None:
            scores[alias] = score
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument(
        "--dry", action="store_true", help="Print records that would be logged."
    )
    parser.add_argument("--results-db", default="results-tracked/results.jsonl")
    args = parser.parse_args()

    all_results = {Path(d).name: load_results(Path(d)) for d in args.dirs}
    all_tasks = sorted(set().union(*(s.keys() for s in all_results.values())))
    methods = list(all_results.keys())

    # Print table
    col_w = max(15, *(len(m) + 2 for m in methods))
    task_w = max((len(t) for t in all_tasks), default=10) + 2
    print(f"{'Task':<{task_w}}" + "".join(f"{m:>{col_w}}" for m in methods))
    for task in all_tasks:
        print(
            f"{task:<{task_w}}"
            + "".join(
                f"{all_results[m].get(task, float('nan')):{col_w}.3f}" for m in methods
            )
        )
    averages = {m: sum(s.values()) / len(s) for m, s in all_results.items() if s}
    print(
        f"{'Average':<{task_w}}"
        + "".join(f"{averages.get(m, float('nan')):{col_w}.3f}" for m in methods)
    )

    if args.log or args.dry:
        for method, scores in all_results.items():
            args.merge_func = method.rsplit("-", 1)[-1]
            run_hash = make_run_hash(
                "collect_results_olmo", args, ignore={"log", "dry", "results_db"}
            )
            if not args.dry and record_exists(args.results_db, run_hash):
                continue
            record = {**args_to_dict(args), "script": "collect_results_olmo"}
            record.update({f"test_{t}": v for t, v in scores.items()})
            if method in averages:
                record["test_avg"] = averages[method]
            if args.dry:
                print(f"\n[dry] {method}:")
                print(json.dumps(record, indent=2))
            else:
                append_result(args.results_db, record, run_hash)


if __name__ == "__main__":
    main()
