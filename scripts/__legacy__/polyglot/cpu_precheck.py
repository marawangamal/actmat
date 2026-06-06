#!/usr/bin/env python3
"""CPU-only pre-check for the Polyglot eval stack — run before queueing a GPU job.

vLLM inference itself needs a GPU, but every dependency/API breakage we hit
while standing this up happened *before* GPU compute: the lighteval vllm-version
guard, vLLM's cached-tokenizer path (transformers API), the custom task configs,
and dataset availability. This validates all of that on CPU in seconds, so a
real GPU slot isn't wasted discovering a broken import or a missing dataset.

Usage:
    source .venv-polyglot/bin/activate
    python scripts/polyglot/cpu_precheck.py
"""

import importlib.util
import sys

MODEL = "ljvmiranda921/Polyglot-OLMo3-7B-SFT-ar"


def main() -> int:
    # 1. vLLM's cached-tokenizer path (broke under transformers 5.x).
    from vllm.transformers_utils.tokenizer import get_tokenizer

    tok = get_tokenizer(MODEL, tokenizer_mode="auto", trust_remote_code=False)
    _ = tok.all_special_tokens_extended
    print("OK  vllm get_tokenizer + all_special_tokens_extended")

    # 2. Custom task configs import + register.
    spec = importlib.util.spec_from_file_location(
        "lt", "scripts/polyglot/lighteval_tasks.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tasks = [t.name for t in m.TASKS_TABLE]
    print("OK  custom tasks:", tasks)

    # 3. Eval datasets resolve + a row matches the expected schema.
    from datasets import load_dataset

    gm = load_dataset(
        "CohereForAI/Global-MMLU-Lite", "ar", split="test", streaming=True
    )
    row = next(iter(gm))
    assert {"question", "option_a", "answer"} <= set(row), row.keys()
    print("OK  Global-MMLU-Lite ar schema")

    mg = load_dataset("juletxara/mgsm", "de", split="test", streaming=True)
    row = next(iter(mg))
    assert {"question", "answer_number"} <= set(row), row.keys()
    print("OK  MGSM de schema")

    print("CPU PRE-CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
