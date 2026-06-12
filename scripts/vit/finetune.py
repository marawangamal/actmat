import argparse
import copy
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch

from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.grad_cross import GradCrossTermTracker
from src.mhas import swap_mha, unswap_mha
from src.utils import LabelSmoothing, cosine_lr
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.heads import build_classification_head
from src.vision.modeling import ImageClassifier, ImageEncoder, apply_lora, merge_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--finetuning-mode", required=True, choices=["fft", "lora"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-location", default="data/vision")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "models"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--ls", type=float, default=0.0)
    parser.add_argument("--warmup-length", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=-1)
    parser.add_argument(
        "--checkpoint-first",
        action="store_true",
        default=False,
        help="Also save a checkpoint right after the first iteration.",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--grad-cross-matrix",
        action="store_true",
        default=False,
        help="Collect per-layer grad-cross-moment sidecars during fine-tuning.",
    )
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _format_duration(seconds):
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def seed_everything(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_args(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)


def init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project="actmat",
        name=f"vit-{args.model}-{args.finetuning_mode}-{args.train_dataset}",
        group=f"vit-{args.model}-{args.finetuning_mode}",
        config=vars(args),
        dir=args.output_dir,
    )


def save_head(args, dataset_name):
    head_path = osp.join(args.output_dir, "head.pt")
    if osp.exists(head_path) and not args.overwrite:
        return
    head_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = build_classification_head(
        head_encoder.model,
        dataset_name,
        None,
        args.data_location,
        args.device,
    )
    classification_head.save(head_path)
    del head_encoder


def _save_step_checkpoint(args, ddp_model, step, wandb_run, grad_cross_tracker=None):
    model_path = osp.join(args.output_dir, f"checkpoint_{step}.pt")
    enc = ddp_model.module.image_encoder
    if args.grad_cross_matrix:
        enc = copy.deepcopy(enc).cpu()
        for module in enc.modules():
            module._forward_hooks.clear()
            module._backward_hooks.clear()
        unswap_mha(enc)
    enc.save(model_path)
    if grad_cross_tracker is not None:
        grad_cross_tracker.save(args.output_dir, step=step)
    _prune_checkpoints(args.output_dir, args.keep_checkpoints)
    print(f"Saved checkpoint to {model_path}", flush=True)
    if wandb_run is not None:
        wandb_run.log({"checkpoint/saved": 1}, step=step)


def _prune_checkpoints(output_dir, keep):
    if keep < 0:
        return
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint_") and name.endswith(".pt"):
            step = int(name[len("checkpoint_") : -len(".pt")])
            checkpoints.append((step, name))
    checkpoints.sort()
    for _, name in checkpoints[:-keep]:
        os.remove(osp.join(output_dir, name))


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)
    seed_everything(args.seed, rank)

    assert not (
        args.grad_cross_matrix and args.num_grad_accumulation > 1
    ), "--grad-cross-matrix is incompatible with gradient accumulation > 1"

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process():
        save_args(args)

    train_dataset = args.train_dataset
    train_dataset_name = (
        train_dataset if train_dataset.endswith("Val") else f"{train_dataset}Val"
    )
    zs_path = osp.join(args.output_dir, "pretrained.pt")
    ft_path = osp.join(args.output_dir, "finetuned.pt")
    if osp.exists(zs_path) and osp.exists(ft_path) and not args.overwrite:
        print(f"Skipping fine-tuning because {ft_path} already exists.")
        cleanup_ddp()
        return

    lora_finetuning = args.finetuning_mode == "lora"
    print("Building image encoder.")
    image_encoder = ImageEncoder(args)

    if is_main_process() and not osp.exists(zs_path):
        image_encoder.save(zs_path)
    if is_main_process():
        save_head(args, train_dataset_name)
    torch.distributed.barrier()

    if lora_finetuning:
        image_encoder = apply_lora(
            image_encoder,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            target_modules="all-linear",
        )
    if args.grad_cross_matrix:
        swap_mha(image_encoder)

    classification_head = torch.load(
        osp.join(args.output_dir, "head.pt"), map_location="cpu", weights_only=False
    )
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    dataset = get_dataset(
        train_dataset_name,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.max_samples is not None and args.max_samples < len(dataset.train_dataset):
        g = torch.Generator().manual_seed(args.seed if args.seed is not None else 0)
        idx = torch.randperm(len(dataset.train_dataset), generator=g)[
            : args.max_samples
        ].tolist()
        dataset.train_dataset = torch.utils.data.Subset(dataset.train_dataset, idx)
        dataset.train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)
    total_steps = args.epochs * num_batches // args.num_grad_accumulation

    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)

    if is_main_process():
        print(f"Total steps: {total_steps}")

    wandb_run = init_wandb(args) if is_main_process() else None

    grad_cross_tracker = None
    if args.grad_cross_matrix and is_main_process():
        grad_cross_tracker = GradCrossTermTracker(ddp_model.module.image_encoder)

    run_start_time = time.perf_counter()
    step = 0
    first_ckpt_saved = False
    for epoch in range(args.epochs):
        ddp_model.train()
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()
            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time
            logits = ddp_model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if grad_cross_tracker is not None:
                    grad_cross_tracker.step()
                optimizer.step()
                optimizer.zero_grad()

            checkpoint_now = (
                args.checkpoint_every > 0
                and step > 0
                and step % args.checkpoint_every == 0
            )
            if args.checkpoint_first and not first_ckpt_saved:
                checkpoint_now = True
                first_ckpt_saved = True
            if checkpoint_now and is_main_process():
                _save_step_checkpoint(
                    args, ddp_model, step, wandb_run, grad_cross_tracker
                )

            if step % 100 == 0 and is_main_process():
                percent_complete = 100 * i / len(ddp_loader)
                elapsed = time.perf_counter() - run_start_time
                batch_time = time.time() - start_time
                print(
                    f"Train Epoch: {epoch}/{args.epochs} "
                    f"[{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\t"
                    f"Data (t) {data_time:.3f}\t"
                    f"Batch (t) {batch_time:.3f}\t"
                    f"Elapsed {_format_duration(elapsed)}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                            "train/data_time": data_time,
                            "train/batch_time": batch_time,
                            "train/elapsed_seconds": elapsed,
                            "train/progress": step / total_steps,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )

            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    if grad_cross_tracker is not None:
        grad_cross_tracker.save(args.output_dir)
        grad_cross_tracker.remove_hooks()

    if is_main_process():
        image_encoder = ddp_model.module.image_encoder
        if lora_finetuning:
            image_encoder = merge_lora(image_encoder)
        enc_to_save = copy.deepcopy(image_encoder).cpu()
        if args.grad_cross_matrix:
            for module in enc_to_save.modules():
                module._forward_hooks.clear()
                module._backward_hooks.clear()
            unswap_mha(enc_to_save)
        enc_to_save.save(ft_path)

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.feature_cache_dir = None
    if args.epochs is None:
        epochs = {
            "Cars": 35,
            "DTD": 76,
            "EuroSAT": 12,
            "GTSRB": 11,
            "MNIST": 5,
            "RESISC45": 15,
            "SUN397": 14,
            "SVHN": 4,
            "CIFAR10": 6,
            "CIFAR100": 6,
            "STL10": 60,
            "Food101": 4,
            "Flowers102": 147,
            "FER2013": 10,
            "PCAM": 1,
            "OxfordIIITPet": 82,
            "RenderedSST2": 39,
            "EMNIST": 2,
            "FashionMNIST": 5,
            "KMNIST": 5,
        }
        dataset_key = args.train_dataset.removesuffix("Val")
        args.epochs = epochs.get(dataset_key, 1)

    torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
