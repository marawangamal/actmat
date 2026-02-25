"""Collects per-layer ||dL/dy||² and ||y||² per sample for offline independence analysis.

Saves two (N,) arrays per layer: g_sq/<layer> and a_sq/<layer>.
Follows gmag.py conventions (batch_size=1, sum over all dims).

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/decorrelation_offline.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
"""

import os
import torch
import numpy as np

from src.vision.task_vectors import NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset


def register_hooks(model):
    activations = {}
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    out.retain_grad()
                    activations[n] = out

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def collect_norms(encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.train()
    model.cuda()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    print(f"    {len(dataset.test_loader.dataset)} samples")

    activations, handles = register_hooks(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    g_sq, a_sq = {}, {}

    for images, labels in dataset.test_loader:
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()

        for name, act in activations.items():
            if act.grad is not None:
                g_sq.setdefault(name, []).append(act.grad.pow(2).sum().item())
                a_sq.setdefault(name, []).append(act.detach().pow(2).sum().item())

    for h in handles:
        h.remove()
    model.cpu()
    return {n: np.array(v) for n, v in g_sq.items()}, {
        n: np.array(v) for n, v in a_sq.items()
    }


if __name__ == "__main__":
    args = parse_arguments()
    args.batch_size = 1
    args.save = f"checkpoints/{args.model}"

    results_dir = f"results/{args.model}"
    os.makedirs(results_dir, exist_ok=True)

    tasks = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    pretrained_ckpt = f"checkpoints/{args.model}/{tasks[0]}Val/zeroshot.pt"

    for task in tasks:
        cache_path = f"{results_dir}/decorrelation_norm_{task}.npz"
        if os.path.exists(cache_path) and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        print(f"\nCollecting norms for {task}")
        tv = NonLinearTaskVector(
            pretrained_ckpt, f"checkpoints/{args.model}/{task}Val/finetuned.pt"
        )
        encoder = tv.apply_to(pretrained_ckpt, scaling_coef=1.0)
        del tv

        g_sq, a_sq = collect_norms(encoder, f"{task}Val", args)
        del encoder

        save_dict = {}
        for layer in g_sq:
            save_dict[f"g_sq/{layer}"] = g_sq[layer]
            save_dict[f"a_sq/{layer}"] = a_sq[layer]
        np.savez(cache_path, **save_dict)
        print(f"  Cached to {cache_path}")

        # # Quick summary
        # NOTE: proper correlation should compate a[l-] with g[l]
        # for layer in sorted(g_sq.keys()):
        #     rho = (
        #         np.corrcoef(g_sq[layer], a_sq[layer])[0, 1]
        #         if g_sq[layer].std() > 0 and a_sq[layer].std() > 0
        #         else 0.0
        #     )
        #     print(f"  {layer:<60} ρ={rho:>7.4f}")
