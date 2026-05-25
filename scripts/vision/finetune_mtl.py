"""Multi-task fine-tuning of an OpenCLIP ViT on the 8-task Ilharco suite.

Mirrors `scripts/language/finetune_mtl.py` but for the vision pipeline:

* tasks: Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
  (the standard task-arithmetic suite from Ilharco et al.)
* model: shared `ImageEncoder` + per-task frozen `MultiHeadImageClassifier`
  heads (built once from the zero-shot CLIP text classifier).
* batching: each optimizer step draws one sub-batch from every task in
  round-robin, sums the per-task losses, then takes a single step (so the
  effective batch is `batch_size * num_tasks`).
* optimizer: AdamW, lr=1e-5 (matches single-task vision FT), wd=0.0,
  cosine schedule with warmup.
* eval: at `--checkpoint-every` steps we run `eval_single_dataset` on each
  task's validation split and early-stop on mean accuracy with `--patience`.

Saves `pretrained.pt` and `{prefix}finetuned.pt` (encoder only, identical
layout to single-task FT) into `{args.save}/multitask/`, so the resulting
checkpoints can be consumed by `eval_task_addition.py` as a multitask
"upper bound" reference run.
"""

import os
import time

import numpy as np
import torch
import trackio

from src.args import parse_arguments
from src.utils import cosine_lr, get_prefix, resolve_run_dir
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.eval import eval_single_dataset
from src.vision.heads import get_classification_head
from src.vision.modeling import (
    ImageClassifier,
    ImageEncoder,
    MultiHeadImageClassifier,
    apply_lora,
    merge_lora,
)


# 8-task task-arithmetic benchmark (Ilharco et al., 2022).
VISION_MIXTURE = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]


def _format_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _evaluate_mixture(image_encoder, args):
    """Evaluate the shared encoder on each task's validation split."""
    per_task = {}
    for name in VISION_MIXTURE:
        per_task[name] = eval_single_dataset(image_encoder, name + "Val", args)["top1"]
    mean = float(np.mean(list(per_task.values())))
    return per_task, mean


def finetune_mtl(args):
    ckpdir = os.path.join(args.save, "multitask")

    assert args.finetuning_mode in [
        "standard",
        "lora",
    ], "Vision MTL supports only 'standard' and 'lora' fine-tuning modes."
    lora = args.finetuning_mode == "lora"

    prefix = get_prefix(args.finetuning_mode)
    ft_path = os.path.join(ckpdir, f"{prefix}finetuned.pt")
    zs_path = os.path.join(ckpdir, "pretrained.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path) and not args.overwrite:
        print(f"Skipping fine-tuning because {ft_path} already exists.")
        return zs_path, ft_path

    print(f"Building image encoder ({args.model}).")
    image_encoder = ImageEncoder(args)

    os.makedirs(ckpdir, exist_ok=True)
    if not os.path.exists(zs_path):
        image_encoder.save(zs_path)

    if lora:
        image_encoder = apply_lora(
            image_encoder,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            target_modules="all-linear",
        )

    classification_heads = [
        get_classification_head(args, name + "Val") for name in VISION_MIXTURE
    ]
    model = MultiHeadImageClassifier(image_encoder, classification_heads)
    model.freeze_head()
    model = model.cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Build one infinite iterator per task, all sharing the encoder's train
    # preprocess. `is_train=True` uses the "Val" variant's train split.
    preprocess_fn = model.image_encoder.train_preprocess
    train_iters = []
    for name in VISION_MIXTURE:
        ds = get_dataset(
            name + "Val",
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loader = get_dataloader(ds, is_train=True, args=args, image_encoder=None)
        train_iters.append(_infinite(loader))

    num_batches = args.num_batches

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, num_batches)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(
        f"\n=== Vision MTL config ===\n"
        f"  mixture:       vision_mixture ({len(VISION_MIXTURE)} tasks)\n"
        f"  tasks:         {VISION_MIXTURE}\n"
        f"  optimizer:     AdamW lr={args.lr} wd={args.wd}\n"
        f"  batch:         {args.batch_size} per task x {len(VISION_MIXTURE)} tasks "
        f"(eff {args.batch_size * len(VISION_MIXTURE)})\n"
        f"  num_batches:   {num_batches} optimizer steps\n"
        f"  finetune mode: {args.finetuning_mode}\n"
    )

    trackio.init(
        project="actmat-mtl-vision",
        name=f"{args.model}-{args.finetuning_mode}-mtl",
        config={
            "model": args.model,
            "finetuning_mode": args.finetuning_mode,
            "mixture": "vision_mixture",
            "tasks": VISION_MIXTURE,
            "lr": args.lr,
            "wd": args.wd,
            "batch_size": args.batch_size,
            "effective_batch": args.batch_size * len(VISION_MIXTURE),
            "num_batches": num_batches,
            "warmup_length": args.warmup_length,
            "checkpoint_every": args.checkpoint_every,
            "patience": args.patience,
        },
    )

    best_mean_acc = -1.0
    bad_ckpts = 0
    saved_best = False

    print("Starting MTL training...", flush=True)
    model.train()
    train_start = time.time()
    for step in range(1, num_batches + 1):
        t0 = time.time()
        optimizer.zero_grad()

        total_loss = 0.0
        data_t = 0.0
        for head_idx in range(len(VISION_MIXTURE)):
            t_data = time.time()
            batch = next(train_iters[head_idx])
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            data_t += time.time() - t_data

            logits = model(inputs, head_idx)
            loss = loss_fn(logits, labels) / len(VISION_MIXTURE)
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scheduler(step - 1)
        optimizer.step()

        if step % args.print_every == 0:
            pct = 100 * step / num_batches
            elapsed = time.time() - train_start
            print(
                f"Step {step}/{num_batches} [{pct:.0f}%]\t"
                f"Loss {total_loss:.6f}\t"
                f"Data {data_t:.3f}\tBatch {time.time() - t0:.3f}\t"
                f"Best mean acc: {100 * best_mean_acc:.2f}%\t"
                f"Elapsed {_format_duration(elapsed)}",
                flush=True,
            )
            trackio.log(
                {"train/loss": total_loss, "train/elapsed_sec": elapsed},
                step=step,
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            # PEFT-wrapped encoders forward correctly in eval without merging;
            # `merge_lora` would destructively fold the adapter into the base
            # and prevent further training.
            model.eval()
            per_task, mean_acc = _evaluate_mixture(model.image_encoder, args)
            print(
                f"\n[Eval @ step {step}] mean={100 * mean_acc:.2f}%  "
                + "  ".join(f"{k}={100 * v:.1f}" for k, v in per_task.items()),
                flush=True,
            )
            trackio.log(
                {
                    "val/mean_acc": mean_acc,
                    **{f"val/{k}_acc": v for k, v in per_task.items()},
                },
                step=step,
            )
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                bad_ckpts = 0
                if not lora:
                    model.image_encoder.save(ft_path)
                saved_best = True
                print(
                    f"New best mean acc: {100 * best_mean_acc:.2f}%"
                    + ("" if lora else " — checkpoint saved."),
                    flush=True,
                )
            else:
                bad_ckpts += 1
                print(
                    f"No improvement ({bad_ckpts}/{args.patience}). "
                    f"Best mean acc: {100 * best_mean_acc:.2f}%",
                    flush=True,
                )
            if bad_ckpts >= args.patience:
                print(f"Early stopping at step {step}.", flush=True)
                break
            model.train()

    # Final save: LoRA always merges + saves; standard saves if no periodic eval ran.
    if lora:
        merged = merge_lora(model.image_encoder)
        merged.save(ft_path)
    elif not saved_best:
        per_task, mean_acc = _evaluate_mixture(model.image_encoder, args)
        print(
            f"\n[Final eval] mean={100 * mean_acc:.2f}%  "
            + "  ".join(f"{k}={100 * v:.1f}" for k, v in per_task.items()),
            flush=True,
        )
        trackio.log(
            {
                "val/mean_acc": mean_acc,
                **{f"val/{k}_acc": v for k, v in per_task.items()},
            },
        )
        model.image_encoder.save(ft_path)

    trackio.finish()
    return zs_path, ft_path


if __name__ == "__main__":
    args = parse_arguments()

    # Vision MTL defaults: lr matches single-task FT; effective batch is
    # batch_size * num_tasks (8); cosine schedule over `num_batches`.
    args.lr = 1e-5
    args.wd = 0.0
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 1
    if args.num_batches is None or args.num_batches <= 0 or args.num_batches == 75000:
        args.num_batches = 4000
    args.warmup_length = min(args.warmup_length, args.num_batches // 10)
    args.print_every = 10
    if args.checkpoint_every is None or args.checkpoint_every < 0:
        args.checkpoint_every = 200

    args.save = resolve_run_dir(args)

    print("=" * 100)
    print(f"MTL fine-tuning {args.model} on vision_mixture ({args.finetuning_mode})")
    print("=" * 100)
    finetune_mtl(args)
