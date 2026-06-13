import itertools
import json
import os
import time

from src.args import parse_arguments
from src.language.task_vectors import (
    LanguageLinearizedTaskVector,
    LanguageNonLinearTaskVector,
)
from src.merging import combine_task_vectors
from src.utils import expert_dir, get_prefix, resolve_run_dir

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()
args.save = resolve_run_dir(args)
prefix = get_prefix(args.finetuning_mode)

print("*" * 100)
if args.finetuning_mode == "standard":
    print(f"Timing merge for non-linear FT models. ({args.merge_func})")
elif args.finetuning_mode == "linear":
    print(f"Timing merge for linear FT models. ({args.merge_func})")
else:
    print(f"Timing merge for {args.finetuning_mode} models. ({args.merge_func})")
print("*" * 100)

# Load task vectors
eval_datasets = list(T5_DATASETS)
task_vectors = []
merge_name = getattr(args, "merge_func", "sum")

for dataset in eval_datasets:
    checkpoint_dir = expert_dir(args.save, dataset, val_suffix=False)
    if args.finetuning_mode == "linear":
        task_vectors.append(
            LanguageLinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
        )
    else:
        task_vectors.append(
            LanguageNonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
        )
    print(f"Task vector {dataset} loaded")

# Build HP grid — use first combo only (no grid search)
hpo = args.hpo or {}
hp_names = list(hpo.keys())
hp_value_lists = list(hpo.values())
hp_combos = (
    [dict(zip(hp_names, combo)) for combo in itertools.product(*hp_value_lists)]
    if hp_names
    else [{}]
)
merge_kwargs = hp_combos[0] if hp_combos else {}

# Time the merge
print("=" * 100)
print(f"Merging with {merge_name}, kwargs={merge_kwargs}")
print("=" * 100)

start = time.time()
task_vector = combine_task_vectors(task_vectors, merge_name, **merge_kwargs)
merge_time = time.time() - start

print(f"Merge time: {merge_time:.4f} seconds")

# Save results
result = {
    "merge_func": merge_name,
    "merge_kwargs": merge_kwargs,
    "merge_time_seconds": merge_time,
}

save_file = f"{args.save}/merge_time_{merge_name}.json"
with open(save_file, "w") as f:
    json.dump(result, f, indent=4)
print(f"Results saved to {save_file}")
