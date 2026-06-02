"""Split a full merge checkpoint into one safetensors file per weight key.

Reads <merge-dir>/model.safetensors.index.json and writes, for every key,
<out-dir>/<key>.safetensors holding that single tensor (under its real key).
These per-layer files are the building blocks for the layer x method hybrid grid
(docs/experiments.md): a hybrid cell is just a directory of symlinks into these,
picking each key from one method's split. One file = one layer => no two
symlinked files ever share a key, so vLLM/HF load them as a clean union.

Usage:
  python scripts/analysis/split_expert.py \
    --merge-dir artifacts/checkpoints/Olmo-3-7b/group-rl-zero/merged/tsv \
    --out-dir   artifacts/checkpoints/Olmo-3-7b/group-hybrid/experts/tsv
"""

import argparse
import json
import os
import os.path as osp

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-dir", required=True, help="source full merge dir")
    ap.add_argument("--out-dir", required=True, help="experts/<method> output dir")
    args = ap.parse_args()

    index_path = osp.join(args.merge_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]

    # group keys by their shard so each big shard is opened/mmapped only once
    by_shard = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    os.makedirs(args.out_dir, exist_ok=True)
    written = skipped = 0
    for shard, keys in by_shard.items():
        with safe_open(osp.join(args.merge_dir, shard), framework="pt", device="cpu") as f:
            for key in keys:
                out = osp.join(args.out_dir, key + ".safetensors")
                if osp.exists(out):
                    skipped += 1
                    continue
                save_file({key: f.get_tensor(key).contiguous()}, out,
                          metadata={"format": "pt"})
                written += 1
    print(f">>> {args.out_dir}: wrote {written}, skipped {skipped} "
          f"(total {written + skipped} per-layer files)")


if __name__ == "__main__":
    main()
