# EigenCov

Data-free model merging via eigenvalue covariance estimation of task vectors.

## Setup

```sh
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

```sh
# 1. Fine-tune on all datasets
bash scripts/vision_train.sh

# 2. Evaluate (single-task + merged models)
bash scripts/vision_eval.sh
```

Results are saved to `results/{model}-{method}/metrics.json`.

## OLMo Experiments (Olmo-3-7b)

```sh
# Merge, evaluate (via olmes), and collect results
# Edit the variables at the top of the script to change models, methods, or GPU count.
bash scripts/olmo/eval_task_addition.sh
```
