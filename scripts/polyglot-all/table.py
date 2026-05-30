import glob
import json
import pandas as pd

# result globs (run from repo root)
LIGHTEVAL = "artifacts/results-polyglot-all/{d}/**/results_*.json"           # MMLU + MRB
COTMGSM   = "artifacts/results-polyglot-all-mgsmcot/{d}/mgsm/**/results_*.json"  # CoT MGSM

# benchmark -> {lang: (json task key, metric)}
mmlu  = {l: (f"global_mmlu_lite:{l}|0", "acc")                   for l in ["ar", "de", "es"]}
mrb   = {l: (f"mrewardbench_mcf:{l}|0", "weighted_acc")          for l in ["ar", "cs", "de", "es"]}
cmgsm = {l: (f"mgsm_native_cot_{l}", "exact_match,strict-match") for l in ["de", "es"]}
benchmarks = [("MMLU", mmlu), ("MRB", mrb), ("CMGSM", cmgsm)]

# method -> (lighteval dir, cot-mgsm dir)
methods = {
    "base":      ("base-Olmo-3-1025-7B",            "base"),
    "mean":      ("Olmo-3-7b-polyglot-all-mean",      "merge-mean"),
    "tsv":       ("Olmo-3-7b-polyglot-all-tsv",       "merge-tsv"),
    "isoc":      ("Olmo-3-7b-polyglot-all-isoc",      "merge-isoc"),
    "wudi":      ("Olmo-3-7b-polyglot-all-wudi",      "merge-wudi"),
    "actmat":    ("Olmo-3-7b-polyglot-all-actmat",    "merge-actmat"),
    "actmat_gd": ("Olmo-3-7b-polyglot-all-actmat_gd", "merge-actmat_gd"),
    "ties":      ("Olmo-3-7b-polyglot-all-ties",      "merge-ties"),
    "dare_ties": ("Olmo-3-7b-polyglot-all-dare_ties", "merge-dare_ties"),
}
# experts collapse into one row: each language routed to its own expert
experts = {l: (f"expert-{l}", f"expert-{l}") for l in ["ar", "cs", "de", "es"]}

def load(pattern, d):
    files = sorted(glob.glob(pattern.format(d=d), recursive=True))
    return json.load(open(files[-1]))["results"] if files else {}

def scores(le_dir, cot_dir, langs=None):
    """{(bench, lang): score} for one model, optionally restricted to `langs`."""
    le, cot = load(LIGHTEVAL, le_dir), load(COTMGSM, cot_dir)
    out = {}
    for bench, spec in benchmarks:
        res = cot if bench == "CMGSM" else le
        for lang, (task, metric) in spec.items():
            if (langs is None or lang in langs) and task in res:
                out[(bench, lang)] = res[task][metric]
    return out

rows = []
for method, (le_dir, cot_dir) in methods.items():
    for (bench, lang), s in scores(le_dir, cot_dir).items():
        rows.append({"method": method, "bench": bench, "score": s})

# experts: per cell take the corresponding-language expert (max if a cell overlaps)
expert_cells = {}
for lang, (le_dir, cot_dir) in experts.items():
    for cell, s in scores(le_dir, cot_dir, langs={lang}).items():
        expert_cells[cell] = max(expert_cells.get(cell, s), s)
for (bench, lang), s in expert_cells.items():
    rows.append({"method": "experts", "bench": bench, "score": s})

df = pd.DataFrame(rows)

avg = df.groupby(["method", "bench"])["score"].mean().unstack()[["MMLU", "MRB", "CMGSM"]]
avg["AVG"] = avg.mean(axis=1)
avg = avg.reindex(["base", "experts"] + [m for m in methods if m != "base"])
print(avg.round(3))
