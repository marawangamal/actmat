# polyglot-all experiments — file map

Merging the 4 Polyglot OLMo3-7B language experts (ar, cs, de, es) into one
multilingual model with several data-free methods, then evaluating merges,
experts, and the base model. MMLU + M-RewardBench run on the **lighteval fork**
(`./lighteval` submodule, 0.13.1.dev0); M-GSM runs on **lm-eval** native CoT.
See the repo [README](../../README.md#polyglot-experiments-olmo-3-7b-multilingual)
for the two-venv setup (incl. the `--no-deps` fork overlay).

Paper: *Polyglot Teachers* (arXiv 2604.11290). Eval settings = paper Table 10:
Global-MMLU Lite (MCF acc, 0-shot), M-RewardBench (MCF weighted_acc, 0-shot),
M-GSM (CoT, 5-shot). Metric keys: `acc`, `weighted_acc`, `exact_match,strict-match`.

> **M-GSM note:** the lighteval `mgsm_custom` (number-only) protocol floors scores
> (~0.1) and unfairly penalizes the chat-tuned experts, so final M-GSM comes from
> lm-eval `mgsm_native_cot` instead (see `eval_all-mgsm.sh`). The MMLU+MRB scripts
> no longer run `mgsm_custom`.

## Scripts (`scripts/polyglot-all/`)

| File | Env | What it does | Submit |
|---|---|---|---|
| `merge.sh` | merge: `.venv-pg-mgsm`<br>eval: `.venv-pg-mmlu-mrb` | array 0-7: merge 4 experts per method → eval merged on MMLU+MRB | `sbatch merge.sh` |
| `eval_experts-mmlu-mrb.sh` | `.venv-pg-mmlu-mrb` | array 0-3: eval each HF-hub expert on its OWN language's MMLU+MRB | `sbatch eval_experts-mmlu-mrb.sh` |
| `eval_base-mmlu-mrb.sh` | `.venv-pg-mmlu-mrb` | eval base `allenai/Olmo-3-1025-7B` on MMLU+MRB | `sbatch eval_base-mmlu-mrb.sh` |
| `eval_all-mgsm.sh` | `.venv-pg-mgsm` | array 0-10: M-GSM CoT for base + 8 merges + de/es experts | `sbatch eval_all-mgsm.sh` |
| `lighteval_tasks.py` | — | vendored custom-tasks file (MMLU + MRB + mgsm_custom defs) | (used via `--custom-tasks`) |
| `agg_table.py` | — | rebuild the base/experts/merges comparison table from results on disk | `python scripts/polyglot-all/agg_table.py` |
| `test.sh` | `.venv-pg-mmlu-mrb` | single-task lighteval sanity smoke | `bash test.sh` |

## Checkpoints (`artifacts/checkpoints/Olmo-3-7b-polyglot-all/`)

- `ar/ cs/ de/ es/` — symlinks to `../Olmo-3-7b-polyglot/<lang>` (each = `finetuned/` + `pretrained` symlink)
- `pretrained/` — symlink to the shared base
- `merged-<method>/` — merged model written by `merge.py` (real safetensors, self-contained, vLLM-loadable)

Experts are evaluated from the **HF hub** (`ljvmiranda921/Polyglot-OLMo3-7B-SFT-<lang>`),
not these local param-folder dirs (which vLLM can't load).

## Results

MMLU+MRB → `artifacts/results-polyglot-all/`; M-GSM → `artifacts/results-polyglot-all-mgsmcot/`.

| Row | Path | Notes |
|---|---|---|
| merges | `results-polyglot-all/Olmo-3-7b-polyglot-all-<method>/___network/results_*.json` | `___network` is a quirk: model_name was an abs path, so `{org}___{model}` parsed oddly. Method distinguished by the outer dir. |
| experts | `results-polyglot-all/expert-<lang>/ljvmiranda921___Polyglot-OLMo3-7B-SFT-<lang>/results_*.json` | per-language subset only |
| base | `results-polyglot-all/base-Olmo-3-1025-7B/allenai___Olmo-3-1025-7B/results_*.json` | MMLU+MRB |
| M-GSM (all) | `results-polyglot-all-mgsmcot/<name>/mgsm/**/results_*.json` | `<name>` ∈ base, merge-<method>, expert-{de,es} |

`--save-details` also writes per-sample parquet under each lighteval results dir's `details/`.

## Logs (`artifacts/logs/`)

- `polyglot_all_merge_eval_<arrayjob>_<idx>.{out,err}`
- `polyglot_all_expert_eval_<arrayjob>_<idx>.{out,err}`
- `polyglot_all_base_eval_<jobid>.{out,err}`

## Job IDs (2026-05-30)

- merges: `9701831_0` (mean), `9701822_1` (tsv), `9701822_2` (actmat), `9701862_3` (isoc), `9701862_4` (wudi), `9702222_5` (actmat_gd)
  (earlier `9701817` array failed: transformers import race + dead mgsm dataset — both fixed)
- experts: `9702101_[0-3]` (ar, cs, de, es)
- base: `9702217`

## Known fixes baked in

- `merge.sh`/`eval_*.sh` stagger array starts (`sleep $((idx*60))`) to avoid a
  transformers lazy-import TOCTOU race on the shared scratch venv.
- M-GSM uses lm-eval `mgsm_native_cot` (the lighteval number-only `mgsm_custom`
  floored scores ~0.1). The vendored `lighteval_tasks.py` still defines
  `mgsm_custom` (pointing at `juletxara/mgsm`, since `ljvmiranda921/mgsm` 404'd)
  but the MMLU+MRB scripts no longer run it.
