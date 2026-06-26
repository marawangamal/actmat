# ACTMat

This is the source code to reproduce the experiments of the paper [Model Merging via Data-Free Covariance Estimation](https://arxiv.org/pdf/2604.01329).

<p align="center">
  <img src="docs/crown-jewel.png" alt="Overview" width="70%">
</p>


## Setup

> **Note:** Each experiment uses it's own uv environment

```sh
# Clone the repository (with submodules)
git clone --recurse-submodules git@github.com:marawangamal/actmat.git
cd actmat

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
# 0. Setup env
uv sync --group vision-language
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
sbatch scripts/vision/finetune.sh   # if ckpts not downloaded
# 2. Evaluate experts        (NUM_TASKS=8|14|20 selects the suite)
sbatch scripts/vision/eval_experts.sh
# 2b. Evaluate pretrained ViTs with the explicit-path wrapper
SINGLE_DIR=pretrained NUM_TASKS=8 bash scripts/vit/eval_single.sh
# 2c. Evaluate multitask ViTs with the explicit-path wrapper
SINGLE_DIR=multitask NUM_TASKS=8 bash scripts/vit/eval_single.sh
# 3a. Evaluate merged models  (NUM_TASKS=8|14|20 selects the suite)
METHODS="tsv isoc actmat" NUM_TASKS=8 FT_MODE=fft MODEL=ViT-B-16 sbatch --array=0-2 scripts/vit/eval_merged.sh
# 3b. Evaluate merged models (packed qkv)
METHODS="tsv isoc actmat" NUM_TASKS=8 FT_MODE=fft MODEL=ViT-B-16 sbatch --array=0-2 scripts/vit/eval_merged_packed.sh
```

Results land under `artifacts/results/{model}/group-{N}/merged/{method}/metrics.json`
(see [Artifacts layout](#artifacts-layout)).

## Language Experiments (T5-Base / T5-Large)

```sh
# 0. Setup env
uv sync --group vision-language
# 1. (Optional) Finetune models (ckpts can be downloaded as described in setup)
bash scripts/language/finetune.sh
# 2. Evaluate experts
bash scripts/language/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/language/eval_task_addition.sh
```

Results are saved to `artifacts/results/{model}-{method}/metrics.json`.

## Reasoning experiments (Olmo-3-7b)

```sh
# 0. Setup env
uv sync --group olmo
# 2. Evaluate base model
bash scripts/olmo_rl_zero/eval_olmo_base.sh 
# 3. Evaluate expert models
bash scripts/olmo_rl_zero/eval_olmo_experts.sh
# 4. Evaluate merged models
bash scripts/olmo_rl_zero/eval_olmo_rl_zero.sh
```

## Clinical experiments (Phi-3.5 / MediPhi)

Merge the 5 [MediPhi](https://huggingface.co/microsoft/MediPhi) clinical experts onto
`Phi-3.5-mini-instruct` and evaluate on the [CLUE](https://github.com/TIO-IKIM/CLUE)
benchmark (the 2 openly-available tasks; the other 4 need PhysioNet/MIMIC credentials).
See [scripts/medphi/README.md](scripts/medphi/README.md) for details and the paper-reproduction numbers.

```sh
# 0. Setup env (clones + patches the CLUE harness, builds .venv-med, fetches data)
bash scripts/medphi/setup.sh
# 2. Evaluate base model (Phi-3.5-mini-instruct)
bash scripts/medphi/eval_medphi_base.sh
# 3. Evaluate expert models
bash scripts/medphi/eval_medphi_experts.sh
# 4. Evaluate merged models
bash scripts/medphi/eval_medphi.sh
```

## Reproducing Plots
See [analysis.ipynb](analysis.ipynb) notebook.



## Artifacts directory structure

The directory structure generally follows the following pattern:

```sh
artifacts/checkpoints/{model}/group-{group}/{experts|multitask|merged}/[{expert|method}]
artifacts/results/{model}/group-{group}/{experts|multitask|merged}/[{expert|method}]/*.json
```

> Note: for checkpoints made by HF scripts, no `experts` dir is created, as experts are directly referenced from $HF_HOME. 
