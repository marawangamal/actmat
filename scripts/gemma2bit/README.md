# Gemma-2-2B-IT — MergeBench pipeline

Merge and evaluate the [MergeBench](https://arxiv.org/abs/2505.10833) Gemma-2-2B-IT
expert family with this repo's task-vector merging methods, following
MergeBench's `scripts/evaluate.sh` exactly.

## Models

- Base: `google/gemma-2-2b-it`
- Experts (HF org `MergeBench`):
  - `MergeBench/gemma-2-2b-it_instruction`
  - `MergeBench/gemma-2-2b-it_math`
  - `MergeBench/gemma-2-2b-it_coding`
  - `MergeBench/gemma-2-2b-it_multilingual`

The safety expert is intentionally skipped — its evaluation requires the
`safety-eval` fork plus an OpenAI judge key, which is out of scope here.

## Evaluation matrix (matches MergeBench's `scripts/evaluate.sh`)

All evals use **lm-eval (HF backend)**. Code tasks use custom yamls that bake
in MergeBench's bigcode-eval generation hyperparameters.

| Capability | Task(s) | Notes |
|---|---|---|
| Math | `gsm8k_cot` | bs=16 |
| Multilingual | `m_mmlu_{fr,es,de,ru}, arc_{fr,es,de,ru}, hellaswag_{fr,es,de,ru}` | bs=8 |
| Instruction | `ifeval` | bs=8 |
| Coding | `humaneval_plus_mb, mbpp_plus_mb` | bs=10, MergeBench gen kwargs (see below) |

## Code-eval task overrides

MergeBench evaluates HumanEval+ and MBPP+ with `bigcode-evaluation-harness`:

```
--max_length_generation 512  --temperature 0.2  --n_samples 10  --batch_size 10
```

To match those numbers via lm-eval, this repo ships two custom task configs at
[`configs/lm_eval_tasks/`](../../configs/lm_eval_tasks/):

- `humaneval_plus_mb.yaml` — inherits `humaneval_plus`, overrides gen kwargs
  (`max_gen_toks=512, do_sample, temperature=0.2, top_p=0.95`) and `repeats=10`
- `mbpp_plus_mb.yaml` — same overrides on top of `mbpp_plus`

lm-eval discovers tasks by scanning its own `lm_eval/tasks/` directory tree, so
the configs must be reachable from there. We symlink them in:

```sh
ln -sf "$PWD/configs/lm_eval_tasks/humaneval_plus_mb.yaml" \
  .venv-gemma/lib/python3.11/site-packages/lm_eval/tasks/humaneval/humaneval_plus_mb.yaml
ln -sf "$PWD/configs/lm_eval_tasks/mbpp_plus_mb.yaml" \
  .venv-gemma/lib/python3.11/site-packages/lm_eval/tasks/mbpp/mbpp_plus_mb.yaml
```

Re-run those two `ln -sf` after every `uv sync` (the sync wipes the venv).

## Setup

```sh
UV_PROJECT_ENVIRONMENT=.venv-gemma uv sync --group gemma
source .venv-gemma/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export HF_ALLOW_CODE_EVAL=1   # required for humaneval_plus_mb / mbpp_plus_mb
# Re-link the custom task yamls into the venv (see "Code-eval task overrides")
ln -sf "$PWD/configs/lm_eval_tasks/humaneval_plus_mb.yaml" \
  .venv-gemma/lib/python3.11/site-packages/lm_eval/tasks/humaneval/humaneval_plus_mb.yaml
ln -sf "$PWD/configs/lm_eval_tasks/mbpp_plus_mb.yaml" \
  .venv-gemma/lib/python3.11/site-packages/lm_eval/tasks/mbpp/mbpp_plus_mb.yaml
```

## End-to-end

```sh
# 1. Download base + 4 experts into param-folder layout
bash scripts/gemma2bit/download_models.sh

# 2. Merge + eval (loops over METHODS in the script)
sbatch scripts/gemma2bit/eval_task_addition.sh

# Optional: per-expert baselines (used as "best single expert" reference)
sbatch scripts/gemma2bit/eval_single_task.sh
```

## Outputs

```
artifacts/checkpoints/gemma-2-2b-it/
├── pretrained/
├── {instruction,math,coding,multilingual}/{pretrained→sym, finetuned/}
└── {mean,tsv,isoc,actmat,wudi,ace,…}/    # merged HF checkpoints

artifacts/results/gemma-2-2b-it-{method}/
├── gsm8k_cot/
├── multilingual/
├── ifeval/
└── code/                                  # humaneval_plus_mb + mbpp_plus_mb
```

## Notes

- The driver requests `gpu:l40s:1` — sufficient now that code-task generation
  is capped at `max_gen_toks=512` (the 999999 default with olmes' `tulu` preset
  was overflowing the 256k-vocab logits buffer).
- `regmean` / `actmat` would normally need a per-capability `covariance.pt`, but
  the `merge_actmat` implementation here is data-free (uses `Δᵀ Δ`).
