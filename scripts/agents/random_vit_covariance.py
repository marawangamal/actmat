"""Collect input covariances for a RANDOMLY-INITIALIZED ViT-L-14.

Apples-to-apples control for the layer-type/depth covariance-similarity study:
same architecture, same data (SVHN train), same split-MHA hooks and same
cov knobs (sm, full, 10x32=320 samples) as the trained expert — but the encoder
weights are fresh random init (open_clip pretrained=None). This isolates whether
the depth/residual-stream structure in the covariance heatmaps comes from the
architecture+data or from learned features.

Forward-only: input second moments are independent of any classification head,
so we hook the bare encoder and skip head construction entirely.

Run (needs a GPU; see scripts/agents/random_vit_covariance.sh):
    export PYTHONPATH="$PYTHONPATH:$PWD"
    python scripts/agents/random_vit_covariance.py \
        --model=ViT-L-14 --data-location=artifacts/data/vision \
        --cache-dir=$SCRATCH/openclip --mha=split --seed 0 \
        --out=artifacts/agents/ViT-L-14-random/SVHNVal/covariance.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import mhas
from src.args import parse_arguments
from src.covariance import OnlineCovariance, register_hooks
from src.vision.datasets.registry import get_dataset
from src.vision.modeling import ImageEncoder


def main():
    # Pull our custom flags out of argv first; the shared parse_arguments()
    # does a strict parse_args() and would reject anything it doesn't define.
    extra = argparse.ArgumentParser()
    extra.add_argument("--seed", type=int, default=0)
    extra.add_argument("--out", required=True)
    known, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_arguments()

    torch.manual_seed(known.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov_device = torch.device("cpu")

    # Random-init encoder: the ImageEncoder __init__ path treats "<name>__init__"
    # as open_clip pretrained=None (random weights). Force it regardless of --model.
    base = args.model.split("__")[0]
    args.model = f"{base}__init__"
    print(f"Building RANDOM {base} (seed={known.seed})")
    encoder = ImageEncoder(args, keep_lang=False)
    encoder.eval().to(device)

    # split-MHA so q/k/v/o covariances are collectible per projection
    encoder = mhas.swap_mha(encoder).to(device)

    cobjs, handles = register_hooks(
        encoder,
        cov_device=cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        extra_module_types=(mhas.MultiHeadAttentionSplit,),
    )

    dataset = get_dataset(
        "SVHNVal",
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.cov_batch_size,
        num_workers=args.num_workers,
    )
    loader = dataset.train_loader if args.cov_split == "train" else dataset.test_loader
    max_batches = max(args.cov_num_batches)

    n_batches = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="random-ViT covariance", total=max_batches):
            if n_batches >= max_batches:
                break
            encoder(images.to(device))
            n_batches += 1
    print(f"Used {n_batches} batches (bs={args.cov_batch_size})")

    for h in handles:
        h.remove()

    saveable = {}
    for lname, obj in cobjs.items():
        if isinstance(obj, OnlineCovariance):
            saveable[lname] = obj.cov.cpu()
            saveable[f"{lname}_n"] = obj.n
        else:
            saveable[lname] = obj
    os.makedirs(os.path.dirname(known.out), exist_ok=True)
    torch.save(saveable, known.out)
    print(f"Saved {len([k for k in saveable if not k.endswith('_n')])} cov matrices to {known.out}")


if __name__ == "__main__":
    main()
