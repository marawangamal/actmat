# polyglot-all experiments — file map

Merging the 4 Polyglot OLMo3-7B language experts (ar, cs, de, es) into one
multilingual model with several data-free methods, then evaluating merges,
experts, and the base model with the paper's lighteval stack
([polyglot-teachers](../../polyglot-teachers), fork `lighteval` 0.13.1.dev0).

Paper: *Polyglot Teachers* (arXiv 2604.11290). Eval settings = paper Table 10:
Global-MMLU Lite (MCF acc, 0-shot), M-RewardBench (MCF weighted_acc, 0-shot),
M-GSM (generative extractive_match, 5-shot). Metric keys: `acc`, `weighted_acc`,
`extractive_match`.

## Scripts (`scripts/polyglot-all/`)

| File | What it does | Submit |
|---|---|---|
| `merge.sh` | array 0-4: merge 4 experts with {mean,tsv,actmat,isoc,wudi} → eval merged model on 9 tasks | `sbatch merge.sh` |
| `eval_experts.sh` | array 0-3: eval each HF-hub expert SFT model on its OWN language's tasks | `sbatch eval_experts.sh` |
| `eval_base.sh` | eval base `allenai/Olmo-3-1025-7B` (paper's S_phi) on all 9 tasks | `sbatch eval_base.sh` |
| `agg_table.py` | rebuild the base/experts/merges comparison table from results on disk | `python scripts/polyglot-all/agg_table.py` |

Merge runs in `.venv-pg` (CPU); eval runs in `polyglot-teachers/.venv` (GPU,
lighteval fork). Both activated inside the scripts.

## Checkpoints (`artifacts/checkpoints/Olmo-3-7b-polyglot-all/`)

- `ar/ cs/ de/ es/` — symlinks to `../Olmo-3-7b-polyglot/<lang>` (each = `finetuned/` + `pretrained` symlink)
- `pretrained/` — symlink to the shared base
- `merged-<method>/` — merged model written by `merge.py` (real safetensors, self-contained, vLLM-loadable)

Experts are evaluated from the **HF hub** (`ljvmiranda921/Polyglot-OLMo3-7B-SFT-<lang>`),
not these local param-folder dirs (which vLLM can't load).

## Results (`artifacts/results-polyglot-all/`)

| Row | Path | Notes |
|---|---|---|
| merges | `Olmo-3-7b-polyglot-all-<method>/___network/results_*.json` | `___network` is a quirk: model_name was an abs path, so `{org}___{model}` parsed oddly. Method distinguished by the outer dir. |
| experts | `expert-<lang>/ljvmiranda921___Polyglot-OLMo3-7B-SFT-<lang>/results_*.json` | per-language subset only |
| base | `base-Olmo-3-1025-7B/allenai___Olmo-3-1025-7B/results_*.json` | full 9 tasks (job 9702217) |
| base (older, partial) | `polyglot-teachers/lighteval-results/allenai___Olmo-3-1025-7B/results_*.json` | from tesh.sh runs (mmlu_de + mrb only) |

`--save-details` also writes per-sample parquet under each results dir's `details/`.

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
- mgsm dataset: `ljvmiranda921/mgsm` (404, removed) → `juletxara/mgsm` in
  `polyglot-teachers/scripts/lighteval_tasks.py` (parquet, same schema).
