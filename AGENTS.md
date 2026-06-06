# Repository Guidelines

## Project Structure & Module Organization

ACTMat reproduces experiments for model merging via data-free covariance estimation. Core reusable code lives in `src/`: shared CLI parsing in `src/args.py`, task-vector logic in `src/task_vectors.py`, merge methods in `src/merging.py`, and artifact path helpers in `src/utils.py`. Pipeline-specific implementations are under `src/vision/`, `src/language/`, and `src/nlg/`.

Experiment drivers and one-off analysis scripts live in `scripts/`, grouped by pipeline (`scripts/vision/`, `scripts/language/`, `scripts/olmo_rl_zero/`, `scripts/medphi/`, etc.). Tests are in `scripts/__tests__/`. Configuration and task definitions are in `configs/`; documentation and figures are in `docs/`. Large generated outputs belong under `artifacts/`, `downloads/`, or external cache paths, not source directories.

## Build, Test, and Development Commands

Use `uv` and keep conflicting dependency groups in separate environments:

```sh
UV_PROJECT_ENVIRONMENT=.venv-vl uv sync --group vision-language
UV_PROJECT_ENVIRONMENT=.venv-olmo uv sync --group olmo
```

Every shell running project code should set:

```sh
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
```

Run focused tests with `python -m pytest scripts/__tests__/test_merging.py`. End-to-end smoke tests use SLURM: `sbatch scripts/__tests__/test_vision_e2e.sh` and `sbatch scripts/__tests__/test_language_e2e.sh`. Typical experiment entry points include `bash scripts/vision/eval_task_addition.sh` and `bash scripts/language/eval_task_addition.sh`.

## Coding Style & Naming Conventions

Write Python 3.10+ code with 4-space indentation and clear snake_case names. Keep shared flags in `src/args.py`; avoid creating separate parsers in pipeline scripts. Add merge methods as `merge_<name>` functions in `src/merging.py`, matching the `--merge-func` CLI value. Preserve existing artifact path conventions through `src/utils.py` instead of hard-coding paths.

## Testing Guidelines

Place unit tests in `scripts/__tests__/` using `test_*.py` names. Prefer small tests for shared math, path construction, and task-vector behavior before running expensive experiments. For pipeline changes, run the relevant pytest target and, when feasible, the matching SLURM smoke test. Document any skipped large-model validation in the PR.

## Commit & Pull Request Guidelines

Use the repository’s Conventional Commit style: `Feat: add qwen`, `Fix: handle missing covariance`, `Chore: update reproducibility scripts`. Keep subjects imperative, under 72 characters, and avoid parenthesized scopes.

PRs should include a concise description, affected pipeline(s), commands run, and paths to important outputs such as `artifacts/results/.../metrics.json`. Link issues when applicable and include screenshots only for plots or documentation changes.

## Security & Configuration Tips

Do not commit credentials, Hugging Face tokens, dataset dumps, model checkpoints, or generated `wandb/` runs. Keep submodules initialized with `git submodule update --init --recursive`; the `lighteval/` fork may need the overlay install documented in `README.md`.
