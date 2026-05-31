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

Vision checkpoints and per-run metrics use a structured, nested convention (path
builders are the single source of truth in `src/utils.py`). `{suffix}` is the
experiment bucket; `[lora_]` is the LoRA filename prefix; `{mode}` is `-w` for
weight-space merges (omitted for the default difference merge).

```
artifacts/
├── checkpoints[-{suffix}]/{model}/
│   ├── experts/{dataset}Val/        pretrained.pt, [lora_]finetuned.pt,
│   │                                [lora_]covariance.pt, fisher.pt, head.pt
│   ├── multitask/                   MTL checkpoint
│   └── pretrained.pt
└── results-{suffix}/{model}/
    ├── merged/{method}[-{mode}]/[lora_]metrics.json
    ├── experts/[lora_]metrics.json
    ├── pretrained/[lora_]metrics.json   # zero-shot baseline
    └── multitask/[lora_]metrics.json
```

There is no task-count level in the path. The 8 / 14 / 20-task suites are selected
via `NUM_TASKS` in the eval scripts, which write to `results-8tasks` / `results-14tasks`
/ `results-20tasks` — so a single `eval_task_addition.sh` / `eval_experts.sh` covers
all three. Named experiment buckets (`results-wang`, `results-sgd`, `results-mixed`)
imply their own count; checkpoints are shared across counts. Migrate an old flat tree
with `scripts/vision/migrate_artifacts.py` (dry-run by default, `--apply`, `--canonical`,
`--pipeline {vision,language}`).

The **language** (T5) pipeline uses the same nested layout via the same helpers
(`val_suffix=False` — bare dataset dirs, no head files), in the single bare
`artifacts/results/{model}/…` tree (one fixed task suite, no count). The **OLMo/polyglot**
pipelines still use the legacy flat `artifacts/results/{model}-{method}/metrics.json` layout.

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

```sh
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
bash scripts/vision/finetune.sh   # if ckpts not downloaded
# 2. Evaluate experts        (NUM_TASKS=8|14|20 selects the suite)
bash scripts/vision/eval_experts.sh
# 3. Evaluate merged models  (NUM_TASKS=8|14|20 selects the suite)
bash scripts/vision/eval_task_addition.sh
```

Results land under `artifacts/results-{N}tasks/{model}/merged/{method}/metrics.json`
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

