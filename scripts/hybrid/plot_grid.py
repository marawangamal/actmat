"""Bar plot of the layer x method hybrid grid (docs/experiments.md).

Style: a basic loop builds the `rows` list, drop it into a DataFrame, then do the
small transforms on the frame; one facet per benchmark, hue = merge method, a
black line per facet for the all-mean floor.
"""

import os.path as osp
import glob
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))

task_to_metric = {
    "gsm8k::tulu": "exact_match",
    "minerva_math_500::tulu": "math_verify",   # not in metrics.json; mean over predictions
    "codex_humaneval::tulu": "pass_at_1",
    "codex_humanevalplus::tulu": "pass_at_1",
    "ifeval::tulu": "prompt_level_loose_acc",
}

task_to_task = {
    "gsm8k::tulu": "GSM8K",
    "minerva_math_500::tulu": "MATH",
    "codex_humaneval::tulu": "HE",
    "codex_humanevalplus::tulu": "HE+",
    "ifeval::tulu": "IF",
}

task_to_chat_templ = {
    "gsm8k::tulu": "math",
    "minerva_math_500::tulu": "math",
    "codex_humaneval::tulu": "code",
    "codex_humanevalplus::tulu": "code",
    "ifeval::tulu": "code",
}

layer_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

method_to_method = {
    "isoc": "Iso-C",
    "tsv": "TSV",
    "actmat_herm": "ACTMat",
}


def read_score(results_dir, alias, metric):
    # minerva math_verify lives per-instance in predictions, not in metrics.json
    if metric == "math_verify":
        fs = glob.glob(osp.join(results_dir, "*minerva*predictions.jsonl"))
        if not fs:
            return None
        vals = [json.loads(line)["metrics"].get("math_verify", 0) for line in open(fs[0])]
        return 100 * sum(vals) / len(vals) if vals else None
    for f in glob.glob(osp.join(results_dir, "task-*-metrics.json")):
        d = json.load(open(f))
        if d["task_config"]["metadata"]["alias"] == alias:
            return 100 * d["metrics"][metric]
    return None


rows = []
grid_dir = f"{ROOT}/artifacts/results-simpler-olmo-exps/Olmo-3-7b/group-hybrid/merged"
mean_dir = f"{ROOT}/artifacts/results-simpler-olmo-exps/Olmo-3-7b/merged/mean"
for layer in layer_types:
    for method in method_to_method:
        for alias, metric in task_to_metric.items():
            chat = task_to_chat_templ[alias]
            score = read_score(f"{grid_dir}/{layer}_{method}/ct-{chat}", alias, metric)
            floor = read_score(f"{mean_dir}/ct-{chat}", alias, metric)
            rows.append(
                {
                    "layer": layer,
                    "method": method_to_method[method],
                    "task": task_to_task[alias],
                    "score": score,
                    "floor": floor,
                }
            )
df = pd.DataFrame(rows)

# transforms on the frame
plot_tasks = ["MATH", "IF"]  # the two benchmarks with real per-layer signal
df = df[df["task"].isin(plot_tasks)]
df["layer"] = pd.Categorical(df["layer"], categories=layer_types, ordered=True)
task_order = plot_tasks
method_order = list(method_to_method.values())
floor = df.groupby("task")["floor"].first()

sns.set_theme(style="whitegrid")
g = sns.catplot(
    df, kind="bar", x="layer", y="score", hue="method",
    col="task", col_order=task_order, order=layer_types, hue_order=method_order,
    col_wrap=2, height=4, aspect=1.5, palette="Set2", sharey=False,
)
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("layer type", "performance")
for task, ax in g.axes_dict.items():
    if pd.notna(floor.get(task)):
        ax.axhline(floor[task], color="black", lw=1.5, label=f"all-mean ({floor[task]:.1f})")
        ax.legend(loc="lower right", fontsize=7)
    ax.set_title(task, fontweight="bold")
g.figure.suptitle("Layer × method hybrid grid (one layer merged by method, rest by mean)",
                  y=1.02, fontweight="bold")

out = f"{ROOT}/artifacts/results-simpler-olmo-exps/Olmo-3-7b/group-hybrid/grid.png"
g.savefig(out, dpi=150, bbox_inches="tight")
print(f">>> wrote {out}  ({len(df)} rows)")
