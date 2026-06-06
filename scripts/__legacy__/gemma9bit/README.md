# Gemma-2-2B-IT — MergeBench pipeline

Merge and evaluate the [MergeBench](https://arxiv.org/abs/2505.10833) Gemma-2-2B-IT
expert family with this repo's task-vector merging methods.

## Models

- Base: `google/gemma-2-9b-it`
- Experts (HF org `MergeBench`):
  - `MergeBench/gemma-2-9b-it_instruction`
  - `MergeBench/gemma-2-9b-it_math`
  - `MergeBench/gemma-2-9b-it_coding`
  - `MergeBench/gemma-2-9b-it_multilingual`

The safety expert is intentionally skipped — its evaluation requires the
`safety-eval` fork plus an OpenAI judge key, which is out of scope here.

## Evaluation matrix

| Capability | Harness | Backend | Task(s) |
|---|---|---|---|
| Instruction | `olmes` | hf | `ifeval::tulu` |
| Math | `olmes` | hf | `gsm8k::tulu` |
| Coding | `olmes` | hf | `codex_humanevalplus::tulu`, `mbppplus:0-shot-chat` |
| Multilingual | `lm-eval` | hf | `m_mmlu_{fr,es,de,ru}`, `arc_{fr,es,de,ru}`, `hellaswag_{fr,es,de,ru}` |

`lm-eval` is used for the multilingual block because `olmes` does not ship
MMLU/ARC/HellaSwag in fr/es/de/ru; for the other three capabilities `olmes`
already matches MergeBench's reference benchmark.

**Backend note:** olmes runs on the HuggingFace backend (`--model-type hf`),
not vllm. vllm's streaming detokenizer stochastically leaves SentencePiece `▁`
(U+2581) markers in code indentation (e.g. `▁▁▁▁for` instead of `    for`),
which makes generated code fail to compile and zeroes out the coding scores.
The token IDs are correct — only vllm's incremental decode is buggy — so the
HF backend, which decodes correctly, is used instead (slower but correct).

## Setup

```sh
UV_PROJECT_ENVIRONMENT=.venv-gemma uv sync --group gemma
source .venv-gemma/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
```

## End-to-end

```sh
# 1. Download base + 4 experts into param-folder layout
bash scripts/gemma9bit/download_models.sh

# 2. Merge + eval (loops over METHODS in the script)
sbatch scripts/gemma9bit/eval_task_addition.sh

# Optional: per-expert baselines (used as "best single expert" reference)
sbatch scripts/gemma9bit/eval_single_task.sh
```

## Outputs

```
artifacts/checkpoints/gemma-2-9b-it/
├── pretrained/
├── {instruction,math,coding,multilingual}/{pretrained→sym, finetuned/}
└── {sum,mean,tsv,isoc,actmat,…}/      # merged HF checkpoints

artifacts/results/gemma-2-9b-it-{method}/
├── olmes/                              # ifeval, gsm8k, code
└── lm-eval/                            # multilingual
```

## Notes

- `regmean` / `actmat` need a per-capability `covariance.pt` and are not
  currently wired up here (no `covariance.py`); the default `METHODS` list
  stays on covariance-free merge functions.
- The driver requests `gpu:a100l:1` (A100L 80GB); Gemma-2-9B-IT in bf16 is
  ~18 GB and combined with the 256k-vocab logits buffer at `--batch-size 2`
  does not fit on an L40S 44 GB. Drop to `--batch-size 1` if OOM occurs.
