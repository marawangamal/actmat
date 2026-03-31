"""Build a combined experts metrics.json from per-domain expert results.

Looks for expert dirs in rootdir by suffix (Math, Code, IF) and picks
the matching tasks from each.

Usage:
    python scripts/olmo/collect_expert_results.py results-olmo
"""

import argparse
import json
from pathlib import Path

# suffix of expert dir name → task alias prefixes to take from it
EXPERT_TASKS = {
    "Math": ["aime"],
    "Code": ["codex_humaneval", "codex_humanevalplus"],
    "IF": ["ifeval"],
}


def load_tasks(results_dir: Path) -> list[dict]:
    f = results_dir / "metrics.json"
    if not f.exists():
        return []
    return json.loads(f.read_text()).get("tasks", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        help="Directory containing expert result dirs.",
        default="results-olmo",
    )
    parser.add_argument("--output", required=True, help="Output dir for combined metrics.json.")
    args = parser.parse_args()

    rootdir = Path(args.rootdir)

    # Find expert dirs by matching suffix
    combined_tasks = []
    for suffix, prefixes in EXPERT_TASKS.items():
        matches = [
            d for d in rootdir.iterdir() if d.is_dir() and d.name.endswith(f"-{suffix}")
        ]
        if not matches:
            print(f"Warning: no dir ending with '-{suffix}' in {rootdir}")
            continue
        if len(matches) > 1:
            print(
                f"Warning: multiple dirs for '{suffix}': {[m.name for m in matches]}, using {matches[0].name}"
            )
        for task in load_tasks(matches[0]):
            if any(task["alias"].startswith(p) for p in prefixes):
                combined_tasks.append(task)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"tasks": combined_tasks}, indent=2))
    print(f"Wrote {len(combined_tasks)} tasks to {out / 'metrics.json'}")
    for t in combined_tasks:
        print(f"  {t['alias']}")


if __name__ == "__main__":
    main()
