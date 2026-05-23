"""Probe whether `tokenizer_mode="slow"` in vllm fixes the U+2581 leak on Gemma."""

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

CKPT = "artifacts/checkpoints/gemma-2-2b-it/mean"


def run(mode: str) -> None:
    print(f"\n{'=' * 70}\nTOKENIZER_MODE = {mode}\n{'=' * 70}", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=400)
    llm = LLM(
        model=CKPT,
        tokenizer_mode=mode,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
    )
    out = llm.generate([PROMPT], sp)[0]
    txt = out.outputs[0].text
    print(f"U+2581 occurrences: {txt.count(chr(0x2581))}", flush=True)
    print("--- OUTPUT ---", flush=True)
    print(repr(txt[:600]), flush=True)
    print("---", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in ("auto", "slow"):
        run(m)
