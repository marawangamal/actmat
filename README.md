# ACTMat

This is the source code to reproduce the experiments of the paper [Model Merging via Data-Free Covariance Estimation](https://arxiv.org/pdf/2604.01329).

<p align="center">
  <img src="docs/crown-jewel.png" alt="Overview" width="70%">
</p>


## Setup

> **Note:** Vision/language and OLMo environments conflict, so separate venvs are used.

```sh
# Clone the repository (with submodules)
git clone --recurse-submodules git@github.com:marawangamal/actmat.git
cd actmat

# If you already cloned without submodules, initialize them with:
#   git submodule update --init --recursive

# Vision & language experiments
UV_PROJECT_ENVIRONMENT=.venv-vl uv sync --group vision-language

# OLMo experiments
UV_PROJECT_ENVIRONMENT=.venv-olmo uv sync --group olmo

# Set env vars
export PYTHONPATH="$PYTHONPATH:$(pwd)" # Add src to python path
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data

# Download data & ckpts
# NOTE: you might need rclone to download this
gdown --folder https://drive.google.com/drive/u/4/folders/1Vc-cGalI9bE5M099x6t4XqGTN-YkQ0Lf -O ./downloads
# extract all to artifacts/
mkdir -p artifacts && for f in downloads/*.tar.gz downloads/*.tgz; do [ -e "$f" ] && tar -xzvf "$f" -C artifacts; done
```


## Artifacts layout

All pipelines share one nested convention — `{model}/group-{group}/experts/…` for
per-expert artifacts and `{model}/group-{group}/merged/{method}/…` for merges — whose
path builders are the single source of truth in `src/utils.py`. `group-{group}` is the
experiment-suite path level (`--group`, default `main`) that sits between `{model}` and
the `experts|multitask|merged|pretrained` subdirs: vision uses `group-{8,14,20}` for the
task-count suite, OLMo uses `group-{rl-zero,polyglot}`, everything else uses `group-main`.
The *contents* of `experts/` differ by pipeline (local weights vs. remote-on-the-Hub),
and only the vision pipeline carries `Val`/`head`/`lora` extras. `[lora_]` is the LoRA
filename prefix; `{mode}` is `-w` for weight-space merges (omitted for the default
difference merge).

### Vision (ViT) & language (T5)

```
artifacts/
├── checkpoints/{model}/
│   ├── group-{group}/
│   │   ├── experts/{dataset}[Val]/  pretrained.pt, [lora_]finetuned.pt,
│   │   │                            [lora_]covariance.pt, fisher.pt[, head.pt]
│   │   └── multitask/               MTL checkpoint
│   └── pretrained.pt                shared base, model-level (above the group)
└── results/{model}/group-{group}/
    ├── merged/{method}[-{mode}]/[lora_]metrics.json
    ├── experts/[lora_]metrics.json
    ├── pretrained/[lora_]metrics.json   # zero-shot baseline
    └── multitask/[lora_]metrics.json
```

Vision selects the 8 / 14 / 20-task suite via `NUM_TASKS` in the eval scripts, which
pass `--group=$NUM_TASKS` — so a single `eval_task_addition.sh` / `eval_experts.sh`
covers all three (`group-8` / `group-14` / `group-20`). Expert **checkpoints** are
shared across suites: they live physically in `group-20` (the superset), and
`group-8` / `group-14` hold per-dataset **symlinks** into it, so finetuning runs once.
Named buckets (`results-wang`, `results-sgd`, `results-mixed`) are exotic and still on
`group-main` by default. **Language (T5)** uses the same helpers with `val_suffix=False`
(bare dataset dirs, no `Val`, no `head.pt`) and a single fixed suite, so it uses
`group-main` throughout.

### HF-merge pathway (Qwen, WizardLM, OLMo)

`src/hf/merge.py` merges models **directly from the HuggingFace Hub** (`scripts/hf/eval_*.sh`).
Here `{model}` is the merge *family* (base model); the experts are remote HF repos, so we
never store their weights — `experts/{expert}/` holds **only** stats sidecars we compute
(absent entirely for the data-free methods sum / mean / tsv / actmat). `{expert}` is the
sanitized HF id (`Qwen/Qwen2.5-Math-1.5B` → `Qwen2.5-Math-1.5B`).

```
artifacts/
├── checkpoints/{family}/group-{group}/
│   ├── experts/{expert}/            covariance.pt, fisher.pt   # stats only; weights on the Hub
│   └── merged/{method}/             full self-contained HF model dir (safetensors + tokenizer)
└── results/{family}/group-{group}/
    └── merged/{method}/             lm-eval / olmes output
```

Qwen / WizardLM use `group-main`. Covariance methods (e.g. regmean) read sidecars from
`--stats-dir <dir>/<expert>/covariance.pt` (no default; the data-free methods never look).
**OLMo** adds per-task **chat-template views**: since one merged model can bake in only one
template, `scripts/hf/eval_olmo*.sh` materialize `merged/{method}-ct-code` and
`merged/{method}-ct-math` (weights symlinked, tokenizer/template swapped) and nest results
as `merged/{method}/{ct-code,ct-math}/`.

OLMo **checkpoints** unify the RL-Zero and polyglot experiments under one `Olmo-3-7b`
model dir, split by group (`group-rl-zero` / `group-polyglot`), with param-folder (not
`.pt`) experts and a preserved `legacy/` for superseded dirs:

```
artifacts/checkpoints/Olmo-3-7b/
├── pretrained/                      shared base (param folder), model-level (above groups)
├── group-rl-zero/
│   ├── experts/{Math,Code,IF}/      pretrained/ (→ ../../../pretrained), finetuned/, covariance.pt
│   ├── merged/{method}/             merged HF model
│   └── legacy/                      parked: old *-chat-* views and *-old snapshots (nothing deleted)
└── group-polyglot/
    └── merged/{method}/             merged HF model (4 multilingual SFT experts)
```

### Migration

Two passes, both dry-run by default with reversible, move-only undo scripts:
- `scripts/vision/migrate_artifacts.py` — old **flat → nested** layout
  (`--apply`, `--canonical`, `--pipeline {vision,language,olmo}`; undo `artifacts/migrate_undo.sh`).
- `scripts/migrate_to_groups.py` — **nested → grouped** (inserts `group-{group}`;
  `--apply`, `--pipeline {vision,language,olmo,all}`; undo `artifacts/migrate_groups_undo.sh`).
  Vision builds the `group-8` / `group-14` expert symlink farms into `group-20`; OLMo
  unifies `Olmo-3-7b` (rl-zero) + `Olmo-3-7b-polyglot-all` (polyglot) and repoints the
  merged-view symlinks. Exotic vision buckets (`results-{wang,sgd,mixed}`, the dashless
  `results14` / `results20`) and the `results-polyglot*` buckets remain deferred.

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

```sh
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
bash scripts/vision/finetune.sh   # if ckpts not downloaded
# 2. Evaluate experts        (NUM_TASKS=8|14|20 selects the suite)
bash scripts/vision/eval_experts.sh
# 3. Evaluate merged models  (NUM_TASKS=8|14|20 selects the suite)
bash scripts/vision/eval_task_addition.sh
```

Results land under `artifacts/results/{model}/group-{N}/merged/{method}/metrics.json`
(see [Artifacts layout](#artifacts-layout)).

## Language Experiments (T5-Base / T5-Large)

```sh
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
bash scripts/language/finetune.sh
# 2. Evaluate experts
bash scripts/language/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/language/eval_task_addition.sh
```

Results are saved to `artifacts/results/{model}-{method}/metrics.json`.

## OLMo Experiments (Olmo-3-7b)

```sh
# 1. Download checkpoints
bash scripts/olmo/download_models.sh
# 2. Evaluate experts
bash scripts/olmo/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/olmo/eval_task_addition.sh # (default gpus: 4)
```

### HF-merge eval (`scripts/hf/eval_olmo*.sh`)

The going-forward pathway merges the RL-Zero experts straight from the Hub via
`src/hf/merge.py` and evaluates with olmes (one array task per merge method):

```sh
sbatch --array=0-3 scripts/hf/eval_olmo.sh      # AIME (pass@32) + code/IF
sbatch --array=0-3 scripts/hf/eval_olmo_v2.sh   # faster: gsm8k instead of AIME, max_length 4096
```

**Eval runtime — single L40S GPU, `eval_olmo_v2.sh` (one method):** olmes is invoked
twice (once per chat-template view: `ct-code`, then `ct-math`), so the model is loaded
twice. Measured below for the `mean` merge (job 9711661); excludes the SLURM queue wait,
and the merge step was skipped (`merged/mean/` already present).

| Phase / task                  | View    | Generations            | Time (L40S) | Score (mean)      |
|-------------------------------|---------|------------------------|-------------|-------------------|
| model load + build both views | —       | merge-skip + 2× vllm   | ~1 + 2.4 + 2.7 min | —          |
| `codex_humaneval::tulu`       | ct-code | 164 × 20 (pass@10)     | 11.2 min    | pass@10 0.858     |
| `codex_humanevalplus::tulu`   | ct-code | 164 × 20 (pass@10)     | 11.5 min    | pass@10 0.819     |
| `ifeval::tulu`                | ct-code | 541 × 1 (greedy ≤2048) | 8.7 min     | loose 0.673       |
| `gsm8k::tulu`                 | ct-math | 1319 × 1 (CoT ≤512)    | 3.0 min     | exact_match 0.795 |

End-to-end **~41 min/method** (`00:40:52` here). The code tasks dominate (20 samples
each); `ifeval`'s greedy ≤2048-token generations also run long. `max_length=4096` caps
each generation at `4096 − prompt`. `eval_olmo.sh` is substantially slower — AIME uses
`pass@32` with effectively unbounded generation, so its `ct-math` task alone can exceed
this entire v2 run.


## Polyglot Experiments (Olmo-3-7b, multilingual)

Merging the 4 Polyglot OLMo3-7B language experts (ar, cs, de, es) and evaluating
on Global-MMLU-Lite, M-RewardBench, and M-GSM. These need **two** dedicated venvs
(their dependency stacks are mutually incompatible — the lighteval/vllm stack pins
`numpy<2`/`datasets<4`, while lm-eval needs `numpy>=2`/`datasets>=4`):

- `.venv-pg-mmlu-mrb` — MMLU + M-RewardBench, via the **lighteval fork** (submodule
  `./lighteval`, 0.13.1.dev0; required for the M-RewardBench metric).
- `.venv-pg-mgsm` — M-GSM, via lm-eval native CoT (`mgsm_native_cot`).

```sh
# MGSM env (lm-eval) — plain uv sync
UV_PROJECT_ENVIRONMENT=.venv-pg-mgsm uv sync --group polyglot-mgsm

# MMLU+MRB env (lighteval fork). The group installs the PyPI BASE (vllm, a
# throwaway PyPI lighteval, plus the fork's non-conflicting runtime deps:
# inspect-ai, more-itertools). The fork itself is then OVERLAID with --no-deps so
# it doesn't drag in numpy>=2 / datasets>=4 and break the vllm stack.
git submodule update --init lighteval
UV_PROJECT_ENVIRONMENT=.venv-pg-mmlu-mrb uv sync --group polyglot-mmlu-mrb
UV_PROJECT_ENVIRONMENT=.venv-pg-mmlu-mrb uv pip install --python .venv-pg-mmlu-mrb -e ./lighteval --no-deps

# Verify the fork + its M-RewardBench metric are importable:
.venv-pg-mmlu-mrb/bin/python -c "from lighteval.metrics.metrics_corpus import MRewardBenchWeightedAccuracy; print('ok')"
```

> **Caveat:** re-running `uv sync --group polyglot-mmlu-mrb` reinstalls the PyPI
> `lighteval` and clobbers the fork — re-run the `-e ./lighteval --no-deps` overlay
> afterwards. If a run dies with another `ModuleNotFoundError`, add that package to
> the `polyglot-mmlu-mrb` group (don't let it bump `numpy`/`datasets`/`vllm`).

```sh
# 1. Merge experts + eval merges on MMLU+MRB (array over methods)
sbatch scripts/polyglot-all/merge.sh
# 2. Eval base + experts on MMLU+MRB
sbatch scripts/polyglot-all/eval_base-mmlu-mrb.sh
sbatch scripts/polyglot-all/eval_experts-mmlu-mrb.sh
# 3. Eval base + experts + merges on M-GSM (one array job)
sbatch scripts/polyglot-all/eval_all-mgsm.sh
# 4. Aggregate the comparison table
python scripts/polyglot-all/agg_table.py
```

Results land in `artifacts/results-polyglot-all/` (MMLU+MRB) and
`artifacts/results-polyglot-all-mgsmcot/` (M-GSM). See
[scripts/polyglot-all/README.md](scripts/polyglot-all/README.md) for the file map.

## Reproducing Plots
See [analysis.ipynb](analysis.ipynb) notebook.

