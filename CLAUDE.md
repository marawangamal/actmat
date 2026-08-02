# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Source code for the paper **Model Merging via Data-Free Covariance Estimation** (ACTMat, COLM 2026). Implements and benchmarks task-vector merging methods (sum/TA, mean, TSV, IsoC, RegMean, Fisher, ACTMat) across three pipelines: vision (OpenCLIP ViTs), language (T5), and OLMo (RL-Zero 7B reasoning experts + a multilingual "polyglot" suite).

ACTMat is **data-free**: `merge_actmat` never touches the covariance/fisher sidecars. Only RegMean (covariance) and Fisher (fisher) consume collected statistics.

## Environments

Each experiment family uses its **own uv venv** because their dependency groups conflict (declared in `pyproject.toml`: `vision-language`, `olmo`, `gemma`, `polyglot-mgsm`, `polyglot-mmlu-mrb`). Create/activate them separately:

```sh
UV_PROJECT_ENVIRONMENT=.venv-vl   uv sync --group vision-language   # ViT + T5
UV_PROJECT_ENVIRONMENT=.venv-olmo uv sync --group olmo              # OLMo / HF merges
```

Every shell that runs a script needs (repo root is the src root):

```sh
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
source .venv-vl/bin/activate           # or .venv-olmo / .venv-pg-*
```

`olmes/` and `lighteval/` are git submodules — run `git submodule update --init --recursive` after a fresh clone. See the README for polyglot (M-GSM / MMLU+MRB) venv setup, which overlays the `./lighteval` fork with `--no-deps`.

## Common commands

Full reproduction recipes (finetune → covariance → eval experts → eval merged) are in `README.md`, driven by SLURM env vars `NUM_TASKS`, `FT_MODE={fft,lora}`, `MODEL`, and `METHODS`:

```sh
# Vision (NUM_TASKS=8|14|20 selects the suite)
METHODS="tsv isoc actmat regmean" NUM_TASKS=8 FT_MODE=fft MODEL=ViT-B-16 \
  sbatch --array=0-3 scripts/vit/eval_merged.sh
# Language
METHODS="tsv isoc actmat regmean" NUM_TASKS=7 FT_MODE=fft MODEL=t5-base \
  sbatch --array=0-3 scripts/t5/eval_merged.sh
# OLMo RL-Zero
METHODS="tsv isoc actmat" sbatch --array=0-2 scripts/olmo_rl_zero/eval_merged.sh
```

Run a single vision merge eval directly (skip SLURM):

```sh
python scripts/vit/eval_merged.py \
  --model ViT-B-32 \
  --experts-dir artifacts/checkpoints/ViT-B-32/group-fft-8/experts \
  --merge-method actmat \
  --output-dir /tmp/out \
  --expert-kwargs '{"mha": "packed"}'
```

Unit tests (pytest) live in `scripts/__tests__/`:

```sh
python -m pytest scripts/__tests__/test_merging.py -q      # tensor merge methods
python -m pytest scripts/__tests__/ -q                     # core/expert/merging suites
```

Note: `scripts/__tests__/test_{vision,language}_e2e.sh` still reference the pre-refactor `scripts/vision/` / `scripts/language/` paths (removed) and are **stale** — prefer the pytest files and the README recipes.

## Architecture

### Merge flow (the core)

`src/core/merge.py::merge_experts(base_expert, expert_experts, merged_expert, merge_method, ...)` is the single in-process merge entry point for **all** pipelines. It walks layers through the generic `Expert` interface, stacks the task deltas `d = stack(w_i - w0)`, and dispatches to `getattr(src.merging, "merge_" + merge_method)`, calling it as:

```python
merge_fn(d=d, w0=w0, stat_fetcher_maps=[...], **merge_kwargs)
```

- Non-2D params (and any layer matching `--ignore-mean`) fall back to a plain mean; layers matching `--ignore-keep-pt` keep the base weights.
- `stat_fetcher_maps[i]` lazily fetches expert `i`'s `{"covariance": ..., "fisher": ...}`; a method returns `d.mean(0)` if the stat it needs is missing.

### Tensor-level merge methods — `src/merging.py`

Each method is `merge_<name>(d, **kwargs)` and the `<name>` **is** the `--merge-method` CLI value (`sum`, `mean`, `tsv`, `isoc`, `regmean`, `regmean_interp`, `fisher`, `actmat`, `actmat_w`, `actmat_herm`, `actmat_gd`, ...). `d` has shape `(N_experts, Do, Di)`; the return is the merged delta `(Do, Di)`. To add a method, just define the function here. `w0` (base weight) and `stat_fetcher_maps` are always passed in; ignore them via `**kwargs` if unused. `actmat_w` is the weight-space variant that uses `w0`.

### Expert interface — `src/core/experts.py`

`Expert` is the minimal contract merge consumes: `get_layers`, `get_layer_params`, `save_layer_params`, `flush`, plus optional `get_layer_cov` / `get_layer_fish`. Pipeline wrappers implement it over their own weight sources:
- `src/vit/experts.py::ViTExpert`, `src/t5/experts.py::T5Expert` — load local `.pt` checkpoints + sidecars.
- `src/hf/experts.py::HFExpert` — pulls weights from the Hub / local HF dirs; stats keyed via `sanitize_hf_id`.

### Pipeline entry points

- **ViT / T5**: `scripts/{vit,t5}/{finetune,covariance,eval_experts,eval_merged,eval_single}.py`, each with a matching `.sh` SLURM driver. Eval scripts take explicit `--experts-dir`, `--output-dir`, `--merge-method`, plus JSON `--merge-kwargs` (forwarded to the merge fn, e.g. `'{"angular_distance": 0.3}'`) and `--expert-kwargs` (forwarded to the Expert, e.g. `'{"mha": "packed"}'`).
- **HF (OLMo RL-Zero, polyglot, and other HF-style models)**: shell drivers in `scripts/olmo_rl_zero/` and `scripts/olmo_polyglot/` call `src/hf/merge.py` directly (`--base-model-name-or-path`, `--expert-model-names-or-paths`, `--merge-method`, `--expert-stats-dir`, `--output-dir`).

### Statistics collection

`scripts/{vit,t5}/covariance.py` walks the model's linear/attention layers, accumulates (un)centered input second moments over a few batches, and writes `covariance.pt` next to the finetuned checkpoint (discovered later by the Expert wrappers). Knobs: `--cov-num-batches`, `--cov-batch-size`, `--cov-type {sm,cov}`, `--cov-estimator {sampled,full,avg}`.

### Vision MHA quirk

Vision covariance collection **replaces** `nn.MultiheadAttention` with a custom split-QKV module (`src/mhas.py::swap_mha` / `MultiHeadAttentionSplit`) so per-head Q/K/V covariances are collectible. A merge that consumes those stats must select the **same** MHA layout the covariance run used (via `--expert-kwargs '{"mha": "split"|"packed"}'`); mismatches cause key-set errors.

### Artifacts layout

Path builders in `src/utils.py` (`resolve_run_dir`, `expert_dir`, `head_path`, `group_dir`, `*_results_path`, `sanitize_hf_id`) define the on-disk convention. The **`group-<g>`** level sits between `{model}` and the `experts|multitask|merged|pretrained` subdirs. Current vision/language shell drivers build paths directly as `group-{ft_mode}-{num_tasks}` (e.g. `group-fft-8`, `group-lora-7`); OLMo uses `group-rl-zero` / `group-polyglot`. A parallel `group-legacy-*` holds an older checkpoint sweep, toggled by commenting the two `EXPERTS_DIR`/`OUT` lines in `eval_merged.sh`.

```
artifacts/checkpoints/{model}/group-{g}/experts/{dataset}[Val]/  pretrained.pt, [lora_]finetuned.pt, [lora_]covariance.pt[, head.pt]
artifacts/checkpoints/{model}/group-{g}/multitask/               MTL checkpoint
artifacts/checkpoints/{model}/pretrained.pt                      shared base (model-level, above the group)

artifacts/results/{model}/group-{g}/merged/{method}[-{mode}]/[lora_]metrics.json
artifacts/results/{model}/group-{g}/{experts,pretrained,multitask}/[lora_]metrics.json
```

`[lora_]` is the `get_prefix()` filename prefix. Vision-only quirks: dataset dirs carry a `Val` split suffix and each holds a co-located `head.pt`; language/OLMo use the bare dataset name and have neither. HF (OLMo) merges create **no** `experts/` dir — expert weights are referenced from `$HF_HOME`.

## Repository conventions

- `src/modeling.py` is a **compatibility shim** re-exporting `src/vision/modeling.py`; old pickled vision checkpoints reference the original module path. Don't remove it without re-pickling checkpoints.
- Reproduce paper plots via `scripts/analysis.ipynb` (reads from `artifacts/results/`). Vision analysis helpers live in `scripts/vit/analysis/`.
- The repo uses `uv` and Python 3.10–3.13 (`.python-version`). Source of truth is `pyproject.toml` + `uv.lock`; `requirements.txt` is a frozen export — don't edit it by hand.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) with **Capital-case** types and **no** parenthesized scopes (use `Fix: ...`, not `Fix(x): ...`):
`Feat` · `Fix` · `Refactor` · `Perf` · `Test` · `Docs` · `Chore` · `Revert`.
Subject ≤ 72 chars, imperative mood, lowercase after the colon. Add a body only when the *why* isn't obvious. No Claude co-author trailers.
