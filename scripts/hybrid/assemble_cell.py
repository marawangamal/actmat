"""Assemble one hybrid grid cell (layer_type x method) as a symlink-only dir.

A cell merges layer type `--layer-type` with `--method` and EVERYTHING else with
`--background` (default: mean). Because the per-layer expert files (from
split_expert.py) each hold exactly one key, the cell is just one symlink per key
pointing at the chosen method's split + a generated index.json + the tokenizer/
config copied from a reference merge. No weights are copied; storage per cell is
symlinks + a small index. See docs/experiments.md.

Layer type of a key = its second-to-last dotted component, e.g.
  model.layers.5.self_attn.q_proj.weight -> q_proj
  model.norm.weight                      -> norm
  lm_head.weight                         -> lm_head

Usage:
  python scripts/hybrid/assemble_cell.py \
    --experts-root artifacts/checkpoints/Olmo-3-7b/group-rl-zero-hybrid/experts \
    --layer-type q_proj --method tsv --background mean \
    --ref-merge artifacts/checkpoints/Olmo-3-7b/group-rl-zero/merged/mean \
    --out-dir artifacts/checkpoints/Olmo-3-7b/group-rl-zero-hybrid/merged/q_proj_tsv-ct-math
"""

import argparse
import json
import os
import os.path as osp
import shutil

# tokenizer/config files copied verbatim from the reference merge (the merges
# were built with the Math chat template, which is what minerva/ct-math needs).
META_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
)


def layer_type_of(key):
    parts = key.split(".")
    return parts[-2] if len(parts) >= 2 else key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-root", required=True, help="group-rl-zero-hybrid/experts")
    ap.add_argument("--layer-type", required=True, help="e.g. q_proj")
    ap.add_argument("--method", required=True, help="method for the chosen layer type")
    ap.add_argument("--background", default="mean", help="method for all other layers")
    ap.add_argument("--ref-merge", required=True,
                    help="a full merge dir: source of the key list + tokenizer/config")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ref_index = json.load(open(osp.join(args.ref_merge, "model.safetensors.index.json")))
    keys = list(ref_index["weight_map"].keys())

    os.makedirs(args.out_dir, exist_ok=True)
    weight_map = {}
    n_method = 0
    for key in keys:
        src_method = args.method if layer_type_of(key) == args.layer_type else args.background
        if src_method == args.method:
            n_method += 1
        src = osp.abspath(osp.join(args.experts_root, src_method, key + ".safetensors"))
        if not osp.exists(src):
            raise FileNotFoundError(f"missing split: {src}")
        fname = key + ".safetensors"
        link = osp.join(args.out_dir, fname)
        if osp.islink(link) or osp.exists(link):
            os.remove(link)
        os.symlink(src, link)
        weight_map[key] = fname

    with open(osp.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": ref_index.get("metadata", {}), "weight_map": weight_map},
            f, indent=2,
        )

    for name in META_FILES:
        src = osp.join(args.ref_merge, name)
        if osp.isfile(src):
            shutil.copy2(src, osp.join(args.out_dir, name))

    if n_method == 0:
        raise SystemExit(
            f"ERROR: no key matched layer-type '{args.layer_type}' "
            f"(nothing taken from {args.method}); check the layer type name")
    print(f">>> {args.out_dir}: {len(weight_map)} layers, "
          f"{n_method} from {args.method}, rest from {args.background}")


if __name__ == "__main__":
    main()
