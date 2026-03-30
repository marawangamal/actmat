# NLG Experiments (Llama-3.1-8B)

Merge 5 capability-specialized LoRA fine-tunes and evaluate on BBH-CoT, HumanEval(+), DROP, GSM8K, IFEval, MATH, PopQA.

## Setup

```sh
# olmes (evaluation framework)
git clone https://github.com/allenai/olmes.git
cd olmes && uv sync && uv sync --group gpu
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
python -c "import nltk; nltk.download('punkt', download_dir='$SCRATCH/nltk_data')"
export NLTK_DATA=$SCRATCH/nltk_data
```

## Full pipeline

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-funcs "eigcov_gd" \
  --merge-kwargs '{"lr": 1e-5, "max_iters": 300, "alpha_weighted": true}' \
  --gpus 4
```


## Olmo 3

### Fine-tune on Tulu 3

```sh
# Single GPU (LoRA)
python scripts/nlg/finetune.py --model olmo --capability math --use-lora --hf-cache-dir $SCRATCH/huggingface

# Multi-GPU full fine-tune with FSDP
torchrun --nproc_per_node=4 scripts/nlg/finetune.py --save-strategy steps --save-steps 200 --model olmo --use-lora --capability all --fsdp --hf-cache-dir $SCRATCH/huggingface --batch-size 64 --grad-accum 1
```

### Merge and evaluate

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir allenai/Olmo-3-1025-7B \
  --finetuned-dirs \
    allenai/Olmo-3-7B-RL-Zero-Math \
    allenai/Olmo-3-7B-RL-Zero-Code \
    allenai/Olmo-3-7B-RL-Zero-IF \
  --merge-funcs "eigcov" \
  --gpus 4
```
