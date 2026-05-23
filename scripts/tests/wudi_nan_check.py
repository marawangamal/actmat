"""Per-layer NaN audit of WUDI merge on T5 task vectors.

Runs `combine_task_vectors(..., 'wudi')` over the 7 T5 fine-tuned task
vectors for one model size, then for every key in the merged result prints
shape, per-task ‖τ_i‖_F^2, and whether the merged tensor contains NaN.

Output is appended to artifacts/tmp/t5-{base,large}.txt.

`wudi_iters` is forced to 5 to keep this fast; one NaN at iter 5 implies
NaN at iter 300.
"""

import io, sys, contextlib
from pathlib import Path

import torch

sys.path.insert(0, ".")

from src.language.task_vectors import LanguageNonLinearTaskVector
from src import merging

_orig = merging.merge_wudi
def _quick(d, **kw):
    kw["wudi_iters"] = 5
    return _orig(d, **kw)
merging.merge_wudi = _quick

from src.merging import combine_task_vectors

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]


def audit(model: str, out_path: Path):
    vecs = [
        LanguageNonLinearTaskVector(checkpoint_dir=f"artifacts/checkpoints/{model}/{d}", prefix="")
        for d in T5_DATASETS
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        merged = combine_task_vectors(vecs, "wudi", merge_mode="d")

    lines = []
    n_nan = 0
    for key, v in merged.vector.items():
        has_nan = torch.isnan(v).any().item()
        if has_nan:
            n_nan += 1
        per_task = []
        for tv in vecs:
            if key in tv.lazy_keys():
                t = tv.get_vector_element(key)
                per_task.append(t.pow(2).sum().item())
            else:
                per_task.append(float("nan"))
        per_task_str = ", ".join(f"{x:.3e}" for x in per_task)
        shape_str = str(tuple(v.shape))
        lines.append(
            f"NaN={'YES' if has_nan else 'no '}  shape={shape_str:<14}  "
            f"per_task_L2^2=[{per_task_str}]  {key}"
        )

    header = (
        f"# WUDI NaN audit on {model} (wudi_iters=5)\n"
        f"# total keys: {len(merged.vector)}, NaN keys: {n_nan}\n"
        f"# per-task order: {T5_DATASETS}\n\n"
    )
    out_path.write_text(header + "\n".join(lines) + "\n")
    print(f"wrote {out_path}  ({n_nan}/{len(merged.vector)} NaN keys)")


if __name__ == "__main__":
    out_dir = Path("artifacts/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in ["t5-base", "t5-large"]:
        audit(m, out_dir / f"{m}.txt")
