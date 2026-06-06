# Llama-3.1-8B-Instruct — MergeBench pipeline

Merge and evaluate the [MergeBench](https://arxiv.org/abs/2505.10833)
Llama-3.1-8B-Instruct expert family with this repo's task-vector merging
methods, following MergeBench's `scripts/evaluate.sh` exactly.

## Models

- Base: `meta-llama/Llama-3.1-8B-Instruct` *(Meta-gated — run `huggingface-cli login` first)*
- Experts (HF org `MergeBench`):
  - `MergeBench/Llama-3.1-8B-Instruct_instruction`
  - `MergeBench/Llama-3.1-8B-Instruct_math`
  - `MergeBench/Llama-3.1-8B-Instruct_coding`
  - `MergeBench/Llama-3.1-8B-Instruct_multilingual`

The safety expert is intentionally skipped — its evaluation requires the
`safety-eval` fork plus an OpenAI judge key, which is out of scope here.

## Evaluation matrix (matches MergeBench's `scripts/evaluate.sh`)

All evals use **lm-eval (HF backend)**. Code tasks use the same custom yamls as
the gemma pipelines, which bake in MergeBench's bigcode-eval generation kwargs.

| Capability | Task(s) | bs |
|---|---|---|
| Math | `gsm8k_cot` | 64 |
| Multilingual | `m_mmlu_{fr,es,de,ru}, arc_{fr,es,de,ru}, hellaswag_{fr,es,de,ru}` | 16 |
| Instruction | `ifeval` | 64 |
| Coding | `humaneval_plus_mb`, `mbpp_plus_mb` | 64 |

## Shared setup

Reuses the **`.venv-gemma`** venv and the shared lm-eval task/patch symlinks
documented in [`scripts/gemma2bit/README.md`](../gemma2bit/README.md):

- `configs/lm_eval_tasks/{humaneval_plus_mb,mbpp_plus_mb}.yaml` symlinked into
  the venv `lm_eval/tasks/` tree
- `configs/lm_eval_patches/mbpp_utils.py` symlinked over
  `lm_eval/tasks/mbpp/utils.py` to fix the `extract_code_blocks` regex bug
  that drops `def` keywords from extracted code

If you haven't set those up yet, run the `ln -sf` commands from
`scripts/gemma2bit/README.md#setup` — they apply to any model evaluated with
the same lm-eval install.

```sh
source .venv-gemma/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export HF_ALLOW_CODE_EVAL=1
huggingface-cli login   # one-time, required for the Meta-gated pretrained
```

## End-to-end

```sh
# 1. Download base + 4 experts into param-folder layout
bash scripts/llama8b/download_models.sh

# 2. Merge + eval (array over METHODS in the script)
sbatch scripts/llama8b/eval_task_addition.sh

# Optional: per-expert baselines (used as "best single expert" reference)
sbatch scripts/llama8b/eval_single_task.sh
```

## Outputs

```
artifacts/checkpoints/Llama-3.1-8B-Instruct/
├── pretrained/
├── {instruction,math,coding,multilingual}/{pretrained→sym, finetuned/}
└── {mean,tsv,isoc,actmat,wudi,ace,…}/    # merged HF checkpoints

artifacts/results/Llama-3.1-8B-Instruct-{method}/
├── gsm8k_cot/
├── multilingual/
├── ifeval/
├── code/          # humaneval_plus_mb
└── mbpp_plus/     # mbpp_plus_mb (separate stage — patch-sensitive)
```

## Notes

- 8B + 128k vocab fits comfortably on a single L40S (44GB) for all stages,
  including the merge step (no SVD intermediates blow up like Gemma-2-9B's
  256k vocab × stacked-expert case).
- `regmean` / `actmat` here are data-free (use `Δᵀ Δ`) — no covariance step.
