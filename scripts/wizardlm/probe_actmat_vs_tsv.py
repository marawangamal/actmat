"""Diagnose why actmat trails tsv on the WizardLM merge.

For each 2D linear-layer parameter:
  • per-task delta norms ‖d_t‖_F
  • cosine(merged_delta, d_t) for the actmat and tsv merges, where
    merged_delta = (merged_weight - pretrained_weight)

If vanilla actmat collapses toward the highest-norm expert, we should see
  • a heavy tail in per-task ‖d_t‖
  • actmat cosines concentrated on the dominant task per layer
  • tsv cosines more evenly distributed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm

from src.nlg.task_vectors import (
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)

WIZARDLM_TASKS = ["LM", "Math", "Code"]


def load_safetensors_index(model_dir: Path) -> dict[str, Path]:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        return {k: model_dir / v for k, v in idx["weight_map"].items()}
    # Single-shard fallback
    sf = list(model_dir.glob("model*.safetensors"))
    if len(sf) != 1:
        raise FileNotFoundError(f"No index and not a single safetensors at {model_dir}")
    return {k: sf[0] for k in load_safetensors(str(sf[0])).keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/checkpoints/wizardlm")
    ap.add_argument("--merges", nargs="+", default=["actmat", "tsv"])
    ap.add_argument("--limit", type=int, default=None,
                    help="optional limit on number of 2D layers to probe")
    ap.add_argument("--out", default="artifacts/results-analysis/wizardlm_actmat_vs_tsv.json")
    args = ap.parse_args()
    root = Path(args.root)

    # Param-folder manifests for pretrained + 3 task finetunes
    task_dirs = {t: root / t for t in WIZARDLM_TASKS}
    pre_dir = task_dirs[WIZARDLM_TASKS[0]] / "pretrained"
    ft_dirs = {t: task_dirs[t] / "finetuned" for t in WIZARDLM_TASKS}
    pre_man = _load_manifest(pre_dir)
    ft_mans = {t: _load_manifest(d) for t, d in ft_dirs.items()}

    # Safetensors indices for each merge
    merge_indices: dict[str, dict[str, Path]] = {}
    merge_cache: dict[str, dict[Path, dict[str, torch.Tensor]]] = {}
    for m in args.merges:
        merge_indices[m] = load_safetensors_index(root / m)
        merge_cache[m] = {}

    # Helper: read a merged tensor by name (sharded safetensors)
    def get_merged(merge_name: str, key: str) -> torch.Tensor:
        idx = merge_indices[merge_name]
        if key not in idx:
            return None
        shard = idx[key]
        if shard not in merge_cache[merge_name]:
            # Evict prior cached shard to keep memory bounded
            merge_cache[merge_name].clear()
            merge_cache[merge_name][shard] = load_safetensors(str(shard))
        return merge_cache[merge_name][shard].get(key)

    rows = []
    keys = [
        k for k, v in pre_man["params"].items()
        if v.get("shape") and len(v["shape"]) == 2 and "int" not in v.get("dtype", "")
    ]
    keys = [k for k in keys if all(k in ft_mans[t]["params"] for t in WIZARDLM_TASKS)]
    if args.limit:
        keys = keys[: args.limit]
    print(f"Probing {len(keys)} 2D layers")

    for key in tqdm(keys):
        pre_t = _load_single_tensor(_build_param_file_path(pre_dir, pre_man, key)).float()
        deltas = {}
        for t in WIZARDLM_TASKS:
            ft = _load_single_tensor(_build_param_file_path(ft_dirs[t], ft_mans[t], key)).float()
            if ft.shape != pre_t.shape:
                deltas = None
                break
            deltas[t] = ft - pre_t
        if deltas is None:
            continue
        norms = {t: deltas[t].norm().item() for t in WIZARDLM_TASKS}
        row = {"key": key, "shape": list(pre_t.shape), "norm": norms}
        for m in args.merges:
            mw = get_merged(m, key)
            if mw is None:
                continue
            md = mw.float() - pre_t
            row[f"norm_merged_{m}"] = md.norm().item()
            # cosine with each task delta
            cos = {}
            mflat = md.flatten()
            mn = mflat.norm().clamp_min(1e-30)
            for t in WIZARDLM_TASKS:
                df = deltas[t].flatten()
                dn = df.norm().clamp_min(1e-30)
                cos[t] = (mflat @ df / (mn * dn)).item()
            row[f"cos_{m}"] = cos
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    import statistics
    def avg(xs): return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== Per-task delta norms (averaged over layers) ===")
    for t in WIZARDLM_TASKS:
        ns = [r["norm"][t] for r in rows]
        print(f"  {t:6s}  mean={avg(ns):.4f}  median={statistics.median(ns):.4f}  max={max(ns):.4f}")

    print("\n=== Dominant task per layer (largest ‖d_t‖) ===")
    from collections import Counter
    dom = Counter()
    for r in rows:
        dom[max(r["norm"], key=r["norm"].get)] += 1
    for t, n in dom.most_common():
        print(f"  {t:6s}  {n}/{len(rows)} layers ({n/len(rows)*100:.1f}%)")

    print("\n=== Mean cosine(merged_delta, task_delta) per merge ===")
    for m in args.merges:
        print(f"  -- {m} --")
        for t in WIZARDLM_TASKS:
            cs = [r[f"cos_{m}"][t] for r in rows if f"cos_{m}" in r]
            print(f"    cos vs {t:6s}: mean={avg(cs):+.3f}  median={statistics.median(cs):+.3f}")

    print("\n=== Argmax-cosine task counts per merge (which expert does the merge most align with per layer?) ===")
    for m in args.merges:
        c = Counter()
        for r in rows:
            if f"cos_{m}" not in r: continue
            c[max(r[f"cos_{m}"], key=r[f"cos_{m}"].get)] += 1
        total = sum(c.values())
        print(f"  -- {m} --")
        for t, n in c.most_common():
            print(f"    {t:6s}  {n}/{total} ({n/total*100:.1f}%)")


if __name__ == "__main__":
    main()
