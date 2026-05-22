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


## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

```sh
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
bash scripts/vision/finetune.sh if ckpts not downloaded.
# 2. Evaluate experts
bash scripts/vision/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/vision/eval_task_addition.sh
```

Results are saved to `artifacts/results/{model}-{method}/metrics.json`.

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


## Reproducing Plots
See [analysis.ipynb](analysis.ipynb) notebook.

