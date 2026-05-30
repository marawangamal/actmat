"""Collect Polyglot results into a table (Global-MMLU + MGSM).

Merged models live under  results-polyglot/Olmo-3-7b-polyglot-<method>/{global_mmlu,mgsm}/...
Reference runs (base + experts) come from the simple.sh single-task dirs
  results-polyglot/simple_global_mmlu_<lang>/<model>/...
  results-polyglot/simple_mgsm_native_cot_<lang>/<model>/...
Cells with no result are shown as '-'.
"""
import json
import glob
import os

ROOT = "artifacts/results-polyglot"
GM_LANGS = ("ar", "de", "es")
MGSM_LANGS = ("de", "es")


def latest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None


def _read(f, key, metric):
    if not f:
        return None
    r = json.load(open(f))["results"]
    v = r.get(key)
    return v.get(metric) if v else None


def merged_gmmlu(d, lang):
    return _read(latest(f"{ROOT}/{d}/global_mmlu/*/results_*.json"),
                 f"global_mmlu_{lang}", "acc,none")


def merged_mgsm(d, lang):
    return _read(latest(f"{ROOT}/{d}/mgsm/*/results_*.json"),
                 f"mgsm_native_cot_{lang}", "exact_match,flexible-extract")


def simple_gmmlu(model, lang):
    return _read(latest(f"{ROOT}/simple_global_mmlu_{lang}/{model}/results_*.json"),
                 f"global_mmlu_{lang}", "acc,none")


def simple_mgsm(model, lang):
    return _read(latest(f"{ROOT}/simple_mgsm_native_cot_{lang}/{model}/results_*.json"),
                 f"mgsm_native_cot_{lang}", "exact_match,flexible-extract")


def fm(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


# (label, gmmlu_fn(lang), mgsm_fn(lang))
ROWS = [
    ("base",         lambda l: simple_gmmlu("allenai__Olmo-3-1025-7B", l),
                     lambda l: simple_mgsm("allenai__Olmo-3-1025-7B", l)),
    ("expert-de",    lambda l: simple_gmmlu("ljvmiranda921__Polyglot-OLMo3-7B-SFT-de", l),
                     lambda l: simple_mgsm("ljvmiranda921__Polyglot-OLMo3-7B-SFT-de", l)),
    ("merge-mean",   lambda l: merged_gmmlu("Olmo-3-7b-polyglot-mean", l),
                     lambda l: merged_mgsm("Olmo-3-7b-polyglot-mean", l)),
    ("merge-tsv",    lambda l: merged_gmmlu("Olmo-3-7b-polyglot-tsv", l),
                     lambda l: merged_mgsm("Olmo-3-7b-polyglot-tsv", l)),
    ("merge-isoc",   lambda l: merged_gmmlu("Olmo-3-7b-polyglot-isoc", l),
                     lambda l: merged_mgsm("Olmo-3-7b-polyglot-isoc", l)),
    ("merge-wudi",   lambda l: merged_gmmlu("Olmo-3-7b-polyglot-wudi", l),
                     lambda l: merged_mgsm("Olmo-3-7b-polyglot-wudi", l)),
    ("merge-actmat", lambda l: merged_gmmlu("Olmo-3-7b-polyglot-actmat", l),
                     lambda l: merged_mgsm("Olmo-3-7b-polyglot-actmat", l)),
    ("merge-actmat_gd", lambda l: merged_gmmlu("Olmo-3-7b-polyglot-actmat_gd", l),
                        lambda l: merged_mgsm("Olmo-3-7b-polyglot-actmat_gd", l)),
]

def avg(d):
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


# ── Per-language table ────────────────────────────────────────────────────────
print(f"{'model':13s}| {'GM_ar':>6s} {'GM_de':>6s} {'GM_es':>6s} | {'MGSM_de':>7s} {'MGSM_es':>7s}")
print("-" * 52)
for label, gfn, mfn in ROWS:
    g = {l: gfn(l) for l in GM_LANGS}
    m = {l: mfn(l) for l in MGSM_LANGS}
    print(
        f"{label:13s}| {fm(g['ar']):>6s} {fm(g['de']):>6s} {fm(g['es']):>6s} "
        f"| {fm(m['de']):>7s} {fm(m['es']):>7s}"
    )

# ── Averaged view (CULTURE = mean Global-MMLU, MATH = mean MGSM) ───────────────
print()
print(f"{'model':13s}| {'CULTURE':>8s} {'MATH':>8s}   (avg over available langs)")
print("-" * 40)
for label, gfn, mfn in ROWS:
    g = {l: gfn(l) for l in GM_LANGS}
    m = {l: mfn(l) for l in MGSM_LANGS}
    print(f"{label:13s}| {fm(avg(g)):>8s} {fm(avg(m)):>8s}")
