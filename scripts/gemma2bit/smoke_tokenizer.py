"""Probe whether the U+2581 leak is intrinsic to MergeBench experts or a merge artifact.

Runs the same prompt with sampling settings matching olmes' codex_humanevalplus::tulu
(temperature=0.8, sample 4 generations) on:
    - The raw, unmerged MergeBench coding expert
    - The base google/gemma-2-2b-it model
    - Our merged `mean` checkpoint
If the leak appears on the raw coding expert too, it's not a merge bug.
"""

import gc

import torch
from vllm import LLM, SamplingParams

PROMPT = (
    "<bos><start_of_turn>user\n"
    "Complete the following function:\n"
    "from typing import List\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    \"\"\"Check if any two numbers in the list are closer than threshold.\"\"\"\n"
    "Provide CONCISE reasoning then finish with:\n\n"
    "Here is the completed function:\n\n```python\n(CODE)\n```\n"
    "<end_of_turn>\n<start_of_turn>model\n"
)

CHECKPOINTS = [
    ("RAW_CODING_EXPERT", "MergeBench/gemma-2-2b-it_coding"),
    ("RAW_BASE_MODEL", "google/gemma-2-2b-it"),
    ("MERGED_MEAN", "artifacts/checkpoints/gemma-2-2b-it/mean"),
]


def run(label: str, ckpt: str) -> None:
    print(f"\n{'=' * 70}\n{label}: {ckpt}\n{'=' * 70}", flush=True)
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400, n=4, seed=0)
    llm = LLM(
        model=ckpt,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
    )
    out = llm.generate([PROMPT], sp)[0]
    for i, o in enumerate(out.outputs):
        txt = o.text
        print(f"-- sample {i}: U+2581={txt.count(chr(0x2581))} chars={len(txt)}", flush=True)
        # Show a small window around the first U+2581 if any
        idx = txt.find(chr(0x2581))
        if idx >= 0:
            print(f"     ...{txt[max(0, idx - 30):idx + 30]!r}...", flush=True)
        else:
            print(f"     head={txt[:120]!r}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for label, ckpt in CHECKPOINTS:
        run(label, ckpt)
