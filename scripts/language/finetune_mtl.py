"""Multi-task fine-tuning of T5 on the t5_mixture from ties-merging.

Follows the recipe in https://github.com/prateeky2806/ties-merging
(``configs/t5_base.json`` / ``configs/t5_large.json``):

* mixture: ``t5_mixture`` (paws, qasc, quartz, story_cloze, wiki_qa,
  winogrande, wsc) concatenated and shuffled once into a single stream
* loss: standard seq2seq LM loss on the mixture (single shared head)
* optimizer: AdamW, lr=1e-4, wd=0, no scheduler
* effective batch ~1024 via gradient accumulation
* 75000 optimizer steps, max_seq_len=128, bf16

Saves ``pretrained.pt`` and ``{prefix}finetuned.pt`` into
``{args.save}/multitask/``.
"""

import os
import time

import torch
import trackio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.args import parse_arguments
from src.utils import get_prefix, resolve_run_dir
from src.language.modeling import T5Wrapper
from src.language.linearize import LinearizedT5Wrapper
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_mixture import (
    T5_MIXTURE,
    get_datasetMixtureReader,
)
from src.language.eval import eval_single_dataset


def _format_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"


def _evaluate_mixture(model, tokenizer, args):
    """Evaluate on each task in ``T5_MIXTURE`` and return (per_task, mean)."""
    per_task = {}
    for name in T5_MIXTURE:
        per_task[name] = eval_single_dataset(
            "validation", model, tokenizer, name, args
        )["top1"]
    mean = sum(per_task.values()) / len(per_task)
    return per_task, mean


def finetune_mtl(args):
    ckpdir = os.path.join(args.save, "multitask")

    assert args.finetuning_mode in [
        "linear",
        "standard",
        "lora",
    ], "Only 'linear', 'standard', and 'lora' fine-tuning modes are supported."
    linearized = args.finetuning_mode == "linear"
    lora = args.finetuning_mode == "lora"

    prefix = get_prefix(args.finetuning_mode)
    ft_path = os.path.join(ckpdir, f"{prefix}finetuned.pt")
    zs_path = os.path.join(ckpdir, "pretrained.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} already exists.")
        return zs_path, ft_path

    print(f"Building model and tokenizer ({args.model}).")
    if linearized:
        model = LinearizedT5Wrapper(args)
        tokenizer = model.tokenizer
    else:
        transformer = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, model_max_length=args.max_seq_len
        )
        model = T5Wrapper(transformer, tokenizer)

    os.makedirs(ckpdir, exist_ok=True)
    model.save(zs_path)

    if lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
        )
        model.transformer = get_peft_model(model.transformer, lora_config)
        model.transformer.print_trainable_parameters()

    model = model.cuda()

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }
    mixture_reader, _ = get_datasetMixtureReader(
        T5_MIXTURE, args.max_datapoints_per_dataset, dataset_kwargs
    )
    createPytorchDataset_fn = lambda d: PytorchDataset(d, tokenizer, "cuda")
    batcher = Batcher(
        mixture_reader,
        createPytorchDataset_fn,
        train_batchSize=args.batch_size,
        eval_batchSize=args.batch_size * 2,
        world_size=None,
        device=None,
    )
    train_iterator = batcher.get_trainBatches("train", template_idx=0)

    num_batches = args.num_batches
    num_grad_accum = getattr(args, "num_grad_accumulation", 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")

    print(
        f"\n=== MTL config ===\n"
        f"  mixture:       t5_mixture ({len(T5_MIXTURE)} tasks)\n"
        f"  tasks:         {T5_MIXTURE}\n"
        f"  optimizer:     AdamW lr={args.lr} wd={args.wd}\n"
        f"  batch:         {args.batch_size} x grad_accum {num_grad_accum} "
        f"(eff {args.batch_size * num_grad_accum})\n"
        f"  num_batches:   {num_batches} optimizer steps\n"
        f"  max_seq_len:   {args.max_seq_len}\n"
        f"  max_pts/ds:    {args.max_datapoints_per_dataset}\n"
        f"  finetune mode: {args.finetuning_mode}\n"
    )

    trackio.init(
        project="actmat-mtl-language",
        name=f"{args.model}-{args.finetuning_mode}-mtl",
        config={
            "model": args.model,
            "finetuning_mode": args.finetuning_mode,
            "mixture": "t5_mixture",
            "tasks": T5_MIXTURE,
            "lr": args.lr,
            "wd": args.wd,
            "batch_size": args.batch_size,
            "num_grad_accumulation": num_grad_accum,
            "effective_batch": args.batch_size * num_grad_accum,
            "num_batches": num_batches,
            "max_seq_len": args.max_seq_len,
            "max_datapoints_per_dataset": args.max_datapoints_per_dataset,
            "checkpoint_every": args.checkpoint_every,
            "patience": getattr(args, "patience", 5),
        },
    )

    patience = getattr(args, "patience", 5)
    best_mean_acc = -1.0
    bad_ckpts = 0
    saved_best = False

    print("Loading and processing dataset (first batch may take ~1 min)...", flush=True)
    model.train()
    train_start = time.time()
    for i in range(num_batches * num_grad_accum):
        t0 = time.time()
        batch = next(train_iterator)
        data_t = time.time() - t0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
            loss = loss / num_grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % num_grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        step = (i + 1) // num_grad_accum
        if (i + 1) % (args.print_every * num_grad_accum) == 0:
            pct = 100 * step / num_batches
            elapsed = time.time() - train_start
            print(
                f"Step {step}/{num_batches} [{pct:.0f}%]\t"
                f"Loss {loss.item():.6f}\t"
                f"Data {data_t:.3f}\tBatch {time.time() - t0:.3f}\t"
                f"Best mean acc: {100 * best_mean_acc:.2f}%\t"
                f"Elapsed {_format_duration(elapsed)}",
                flush=True,
            )
            trackio.log(
                {
                    "train/loss": loss.item() * num_grad_accum,
                    "train/elapsed_sec": elapsed,
                },
                step=step,
            )

        if (
            args.checkpoint_every > 0
            and step > 0
            and (i + 1) % (args.checkpoint_every * num_grad_accum) == 0
        ):
            per_task, mean_acc = _evaluate_mixture(model, tokenizer, args)
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
                    model.save(ft_path)
                saved_best = True
                print(
                    f"New best mean acc: {100 * best_mean_acc:.2f}%"
                    + ("" if lora else " — checkpoint saved."),
                    flush=True,
                )
            else:
                bad_ckpts += 1
                print(
                    f"No improvement ({bad_ckpts}/{patience}). "
                    f"Best mean acc: {100 * best_mean_acc:.2f}%",
                    flush=True,
                )
            if bad_ckpts >= patience:
                print(f"Early stopping at step {step}.", flush=True)
                break
            model.train()

    # Final save (merge LoRA; otherwise save if periodic eval never fired).
    if lora:
        model.transformer = model.transformer.merge_and_unload()
        model.save(ft_path)
    elif not saved_best:
        per_task, mean_acc = _evaluate_mixture(model, tokenizer, args)
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
        model.save(ft_path)

    trackio.finish()
    return zs_path, ft_path


if __name__ == "__main__":
    args = parse_arguments()

    # ties-merging recipe defaults (configs/t5_{base,large}.json):
    args.lr = 1e-4
    args.wd = 0.0
    args.max_seq_len = 128
    args.max_datapoints_per_dataset = 500000
    # Effective batch ~1024.
    _bs_defaults = {"t5-base": (256, 4), "t5-large": (64, 16)}
    _bs, _ga = _bs_defaults.get(args.model, (64, 16))
    args.batch_size = _bs
    args.num_grad_accumulation = _ga
    args.num_batches = 75000
    args.print_every = 10
    # Validation cadence: default off (ties trains for the full 75k). Set
    # --checkpoint-every >0 on the CLI to enable periodic eval + early stop.
    if args.checkpoint_every is None or args.checkpoint_every < 0:
        args.checkpoint_every = -1

    args.save = resolve_run_dir(args)

    print("=" * 100)
    print(f"MTL fine-tuning {args.model} on t5_mixture ({args.finetuning_mode})")
    print("=" * 100)
    finetune_mtl(args)
