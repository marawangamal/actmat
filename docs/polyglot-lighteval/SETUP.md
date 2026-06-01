# Setup (eval environment)

Notes for getting the `lighteval`-based evals (e.g. `tesh.sh`) running in this repo.

## Why this is not just `uv sync`

`lighteval` is shipped here as a **git submodule** pointing at the author's fork
(`ljvmiranda921/lighteval`, version `0.13.1.dev0`). The custom tasks in
`scripts/lighteval_tasks.py` need symbols from that fork (e.g.
`LogLikelihoodAccMetric`, the MRewardBench metrics) that the PyPI release does
**not** have.

However, `pyproject.toml` declares `lighteval>=0.10.0` from **PyPI** (the `eval`
extra) and does *not* wire the submodule in via `[tool.uv.sources]`. A plain
`uv sync` therefore installs PyPI `lighteval==0.10.0`, and the custom tasks fail
with `ImportError: cannot import name 'LogLikelihoodAccMetric'`.

We can't make the fork a managed uv source either: the fork's declared deps
(`numpy>=2`, `datasets>=4`, newer `outlines`/`outlines-core`) are mutually
incompatible with the project's pinned `vllm` stack (`vllm==0.10.1.1` →
`outlines-core==0.2.10`, `numpy<2`, `datasets<4`). The resolver cannot satisfy
both at once.

So the working approach is: build a consistent env from PyPI, then **overlay
only the fork's `lighteval` package** with `--no-deps`.

## Steps

```sh
cd polyglot-teachers

# 0. ensure the lighteval fork submodule is checked out (needed for the overlay)
git submodule update --init lighteval

# 1. fresh venv (optional — skip the rm if you already have a good .venv)
rm -rf .venv

# 2. build a consistent env from PyPI.
#    The `eval` extra installs vllm (required at runtime) plus a throwaway
#    PyPI lighteval==0.10.0 that step 3 overwrites.
UV_PROJECT_ENVIRONMENT=.venv uv sync --extra eval

# 3. overlay ONLY the fork's lighteval package; --no-deps leaves the pinned
#    vllm / numpy<2 / datasets<4 stack untouched.
UV_PROJECT_ENVIRONMENT=.venv uv pip install --python .venv -e ./lighteval --no-deps

# 4. install fork-only deps that --no-deps skipped.
#    The fork added an inspect-ai backend; its CLI imports inspect_ai
#    unconditionally, so it must be present even for the vllm command.
UV_PROJECT_ENVIRONMENT=.venv uv pip install --python .venv "inspect-ai>=0.3.140"
```

### Verify

```sh
.venv/bin/python -c "import importlib.metadata as m; print('lighteval', m.version('lighteval'))"   # -> 0.13.1.dev0
.venv/bin/python -c "from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric; print('import OK')"
```

> First import is slow (tens of seconds): it pulls in torch / transformers /
> vllm and compiles bytecode. Subsequent imports are faster.

## Running an eval

```sh
source .venv/bin/activate
export HF_HOME=$SCRATCH/huggingface
bash tesh.sh
```

`tesh.sh` runs a single task (`global_mmlu_lite:de`) on `allenai/Olmo-3-1025-7B`
with the vllm backend, writing results to `lighteval-results/`. (Hub push is
disabled — no `--push-to-hub`.) Edit the `TASK` variable to run a different one;
the full task list is commented out at the top of the script.

## Caveats

- **A `uv sync` will clobber the fork.** It reinstalls PyPI `lighteval==0.10.0`.
  Re-run steps 3 and 4 after any sync.
- **`inspect-ai` may not be the only missing fork dep.** If a run dies with
  another `ModuleNotFoundError`, install that package the same way as step 4
  (the fork's full dep list is in `lighteval/pyproject.toml`). Avoid letting any
  install bump `numpy` to `>=2` or change `vllm` — that breaks the stack.
