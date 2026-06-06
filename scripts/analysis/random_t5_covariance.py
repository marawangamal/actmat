"""Collect input covariances for a RANDOMLY-INITIALIZED t5-large (paws data).

Language counterpart of random_vit_covariance.py. Architecture + tokenizer are
taken from the trained pretrained.pt (so the config is byte-identical), but the
transformer weights are freshly random (T5ForConditionalGeneration(config)).
Same data (paws train), same hooks (batch_first) and cov knobs (sm, full,
10x32=320 samples) as the trained expert -> isolates architecture+data vs
learned features in the covariance-similarity structure.

Run (needs GPU; see scripts/analysis/random_t5_covariance.sh):
    export PYTHONPATH="$PYTHONPATH:$PWD"
    python scripts/analysis/random_t5_covariance.py \
        --model=t5-large --seed 0 \
        --out=artifacts/checkpoints-analysis/t5-large-random/paws/covariance.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.args import parse_arguments
from src.covariance import OnlineCovariance
from src.language.modeling import T5Wrapper

# import compute_covs from the language covariance script without shadowing src.covariance
_spec = importlib.util.spec_from_file_location(
    "_lang_cov", project_root / "scripts" / "language" / "covariance.py"
)
_lang_cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lang_cov)
compute_covs = _lang_cov.compute_covs


def main():
    # Pull our custom flags out of argv first; the shared parse_arguments()
    # does a strict parse_args() and would reject anything it doesn't define.
    extra = argparse.ArgumentParser()
    extra.add_argument("--seed", type=int, default=0)
    extra.add_argument("--out", required=True)
    extra.add_argument("--pretrained", default="artifacts/checkpoints/t5-large/pretrained.pt")
    extra.add_argument("--dataset", default="paws")
    known, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_arguments()

    torch.manual_seed(known.seed)
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")
    args.max_seq_len = 128
    args.cov_num_batches = max(args.cov_num_batches)

    # architecture + tokenizer from the trained checkpoint; random weights
    print(f"Loading config/tokenizer from {known.pretrained}")
    trained = torch.load(known.pretrained, map_location="cpu", weights_only=False)
    config = trained.transformer.config
    tokenizer = trained.tokenizer
    del trained

    print(f"Building RANDOM t5-large from config (seed={known.seed})")
    rand_transformer = T5ForConditionalGeneration(config)
    model = T5Wrapper(rand_transformer, tokenizer)

    def on_end(cobjs, _out=known.out):
        saveable = {}
        for lname, obj in cobjs.items():
            if isinstance(obj, OnlineCovariance):
                saveable[lname] = obj.cov.cpu()
                saveable[f"{lname}_n"] = obj.n
            else:
                saveable[lname] = obj
        os.makedirs(os.path.dirname(_out), exist_ok=True)
        torch.save(saveable, _out)
        print(f"Saved {len([k for k in saveable if not k.endswith('_n')])} cov matrices to {_out}")

    compute_covs(model, known.dataset, args, on_end=on_end)


if __name__ == "__main__":
    main()
