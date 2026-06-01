"""Print parameter counts for each vision model (the image encoder used for merging).

Loads each OpenCLIP ViT exactly as scripts/vision/finetune.py does (via ImageEncoder),
with the language transformer dropped (keep_lang=False) since only the image tower is merged.

Usage:
    python scripts/vision/count_params.py
    python scripts/vision/count_params.py --models ViT-B-32 ViT-L-14
"""

import argparse
import os
from types import SimpleNamespace

from src.vision.modeling import ImageEncoder

DEFAULT_MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]


def count(model_name, cache_dir):
    args = SimpleNamespace(model=model_name, cache_dir=cache_dir, feature_cache_dir=None)
    encoder = ImageEncoder(args, keep_lang=False)
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("SCRATCH", os.path.expanduser("~/.cache")), "models"
        ),
    )
    cli = parser.parse_args()

    print(f"{'Model':<12}{'Total params':>18}{'Trainable':>18}")
    print("-" * 48)
    for model_name in cli.models:
        total, trainable = count(model_name, cli.cache_dir)
        print(f"{model_name:<12}{total:>18,}{trainable:>18,}")


if __name__ == "__main__":
    main()
