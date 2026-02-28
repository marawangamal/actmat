# Covariance Estimation using Task Matrices

## TODO
- [ ] Move src/args.py => src/vision/args.py

## Setup

```sh
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
cp vit_datasets_08.zip $SLURM_TMPDIR/
unzip -q $SLURM_TMPDIR/vit_datasets_08.zip -d $SLURM_TMPDIR/
```

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

### 1. Fine-tune

```sh
python scripts/vision/finetune.py \
  --finetuning-mode=standard \
  --model=ViT-B-16 \
  --world-size=1 \
  --num-workers=1 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets \
  --save=$SCRATCH/eigcov/checkpoints/vision 
```

Options for `--finetuning-mode`: `standard`, `lora`, `linear`, `posthoc`.

### 2. Evaluate single model (zeroshot / standard / linear / lora)

```sh
python scripts/vision/eval_single_task.py \
  --finetuning-mode=lora \
  --model=ViT-B-32 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

### 3. Evaluate merged models
**NOTE:** Must run single task script in (2) first.

```sh
# EigenCov (data-free)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=eigcov

# RegMean (must run covariance.py first, see below)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=lora --merge-func=regmean \
  --mha=split \
  --cov-dir=results/ViT-B-16/covariances_strain_n10_b32_tsm_attnsplit_efull_ftlora \
  --coeff-start=0 --n-eval-points=11
```

## Scripts 

### Generate covariance matrices (only required for RegMean)
```sh
python scripts/vision/covariance.py \
  --model=ViT-B-32 \
  --cov-split=train \
  --cov-num-batches=10 \
  --cov-batch-size=32 \
  --mha=split \
  --cov-type=sm \
  --cov-estimator=full
```
**NOTE:** `--mha`: `split` Splits q,k,v into separate linear modules so their activation covariances can be collected (otherwise will be ignored).


### Reproducing Vision Experiments
To reproduce vision experiments simply run the following commands. The results will be saved to `results/results.jsonl`
```sh
bash scripts/vision/train.sh
bash scripts/vision/eval.sh
```

### Reproducing gradient orientation experiments
```sh
python scripts/vision/finetune.py \
  --finetuning-mode=standard \
  --model=ViT-B-16 \
  --world-size=1 \
  --num-workers=1 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets \
  --save=$SCRATCH/eigcov/checkpoints/vision \
  --cosine-samples=128
```

## Language Experiments
### 1. Fine-tune

```sh
python scripts/language/finetune.py \
  --finetuning-mode=standard \
  --model=t5-base \
  --save=$SCRATCH/eigcov/checkpoints/language \
  --hf-cache-dir=$SCRATCH/hf_cache
```

### 2. Evaluate single model (zeroshot / standard / linear / lora)
```sh
python scripts/language/eval_single_task.py \
--finetuning-mode=standard  --save=$SCRATCH/eigcov/checkpoints/language
```

### 3. Evaluate merged models
**NOTE:** Must run single task script in (2) first.
```sh
python scripts/language/eval_task_addition.py \
  --model=t5-base --finetuning-mode=standard --merge-func=eigcov --save=$SCRATCH/eigcov/checkpoints/language
```

## Repository Structure

```
src/                          # Library code (importable modules)
├── task_vectors.py           # Task vector arithmetic (model-agnostic)
├── merging.py                # Merging strategies: EigenCov, RegMean, TSV, ...
├── covariance.py             # OnlineCovariance + register_hooks (model-agnostic)
├── distributed.py            # DDP utilities (model-agnostic)
├── mhap.py / mhas.py         # Custom multi-head attention variants: packed / split (model-agnostic)
├── args.py                   # Shared argument parser
├── utils.py                  # Shared utilities
│
├── vision/                   # Vision-specific library code (OpenCLIP / ViT)
│   ├── modeling.py           # ImageEncoder, ImageClassifier
│   ├── heads.py              # Zero-shot classification heads
│   ├── eval.py               # eval_single_dataset, evaluate_task_vector, ...
│   ├── linearize.py          # Taylor-linearized vision encoder
│   └── datasets/             # MNIST, Cars, DTD, EuroSAT, GTSRB, ...
│
└── language/                 # Language-specific code
    ├── modeling.py
    ├── linearize.py
    ├── args.py
    ├── eval/
    └── datasets/

scripts/                      # Entry points (run directly)
├── setup.sh
├── vision/
│   ├── finetune.py           # Fine-tune vision models
│   ├── eval_single_task.py   # Evaluate single fine-tuned model
│   ├── eval_task_addition.py # Evaluate merged model
│   ├── eval_task_negation.py
│   ├── covariance.py         # Collect per-layer covariance matrices
│   └── correlation.py        # Collect correlations of actiavations & gradients
└── language/
    ├── finetune.py           # Fine-tune T5 models
    ├── eval_single_task.py   # Evaluate single fine-tuned model
    ├── eval_task_addition.py
    └── eval_task_negation.py
```


## Overview of Language Experiments

**Model: T5**(encoder-decoder)                                                                                                                                                  

t5-base is a seq2seq (encoder-decoder) transformer. The encoder reads the input prompt, the decoder generates the output. It's NOT a causal/decoder-only LM like GPT — it was
  pretrained on span-corruption (masking spans and predicting them), but the lm-adapt variants were further trained on language modeling.                                     
                

**What's being tested**

All 7 datasets (qasc, wiki_qa, quartz, paws, story_cloze, winogrande, wsc) are multiple-choice classification tasks. During eval, the model doesn't generate text — instead
it scores each candidate answer by computing the log probability of the decoder producing that answer string given the encoded input:

score(choice_i) = sum of log P(token | context) over all tokens in choice_i

The predicted answer is argmax over all choices. This is called rank classification or closed-form eval — much faster and more reliable than generation.

During training, it's standard cross-entropy on the correct answer text (teacher forcing).


**Prompt templating: promptsource, not HF chat templates**

It's completely different from HF chat templates. promptsource (from the BigScience T0 paper) is a library of hand-written Jinja2 templates that convert structured dataset
fields into natural language. Each dataset has multiple templates written by crowd workers.

For example, for qasc a template might look like:

**Jinja2 template**

{{fact1}} {{fact2}} Based on these facts, {{question}}
- {{choices[0]}}
- {{choices[1]}}
...
Answer: ||| {{choices[answer_idx]}}

The ||| separator splits input from target. So for a concrete example:

Input:  "Magnets attract certain metals. Iron is a metal. Based on these facts,
          what do magnets attract? - Iron - Wood - Plastic"
Target: "Iron"

HF chat templates in contrast add structural tokens (<|system|>, <|user|>, <|assistant|>) and are designed for instruction-tuned conversational models. Promptsource
templates are purely natural language reformulations with no special tokens — the task is framed as a fill-in-the-blank text completion problem, which fits T5's seq2seq
pretraining naturally.