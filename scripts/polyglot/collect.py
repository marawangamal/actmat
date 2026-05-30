"""Collect Polyglot merge results into a table (Global-MMLU + MGSM)."""
import json
import glob
import os

ROOT = "artifacts/results-polyglot"


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def gmmlu(d):
    f = latest(f"{d}/global_mmlu/*/results_*.json")
    if not f:
        return {}
    r = json.load(open(f))["results"]
    return {
        l: r[f"global_mmlu_{l}"]["acc,none"]
        for l in ("ar", "de", "es")
        if r.get(f"global_mmlu_{l}")
    }


def mgsm(d):
    f = latest(f"{d}/mgsm/*/results_*.json")
    if not f:
        return {}
    r = json.load(open(f))["results"]
    return {
        l: r[f"mgsm_native_cot_{l}"].get("exact_match,flexible-extract")
        for l in ("de", "es")
        if r.get(f"mgsm_native_cot_{l}")
    }


def fm(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


ROWS = [
    ("merge-mean", "Olmo-3-7b-polyglot-mean"),
    ("merge-tsv", "Olmo-3-7b-polyglot-tsv"),
    ("merge-actmat", "Olmo-3-7b-polyglot-actmat"),
]

print(f"{'model':13s}| {'GM_ar':>6s} {'GM_de':>6s} {'GM_es':>6s} | {'MGSM_de':>7s} {'MGSM_es':>7s}")
print("-" * 52)
for label, d in ROWS:
    g = gmmlu(f"{ROOT}/{d}")
    m = mgsm(f"{ROOT}/{d}")
    print(
        f"{label:13s}| {fm(g.get('ar')):>6s} {fm(g.get('de')):>6s} {fm(g.get('es')):>6s} "
        f"| {fm(m.get('de')):>7s} {fm(m.get('es')):>7s}"
    )
