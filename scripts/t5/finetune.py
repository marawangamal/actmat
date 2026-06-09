import argparse
import copy
import gc
import os
import os.path as osp
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from src.grad_cross import GradCrossTermTracker  # noqa: E402
from src.language.datasets.batcher import Batcher  # noqa: E402
from src.language.datasets.dataset_readers import get_datasetReader  # noqa: E402
from src.language.datasets.pytorch_dataset import PytorchDataset  # noqa: E402
from src.language.eval import eval_single_dataset  # noqa: E402
from src.language.modeling import T5Wrapper  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--finetuning-mode", required=True, choices=["fft", "lora"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-location", default="data")
    parser.add_argument(
        "--cache-dir",
        default=osp.join(
            os.environ.get("SCRATCH", osp.expanduser("~/.cache")), "huggingface"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-grad-accumulation", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
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


def _batch_defaults(model_name):
    return {
        "t5-base": (256, 4),
        "t5-large": (64, 16),
    }.get(model_name, (64, 16))


def save_args(args):
    os.makedirs(args.output_dir, exist_ok=True)
    import json

    with open(osp.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)


def init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project="actmat",
        name=f"t5-{args.model}-{args.finetuning_mode}-{args.train_dataset}",
        group=f"t5-{args.model}-{args.finetuning_mode}",
        config=vars(args),
        dir=args.output_dir,
    )


def save_model_without_hooks(model, path):
    model_to_save = copy.deepcopy(model).cpu()
    for module in model_to_save.modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
    model_to_save.save(path)
    del model_to_save


def build_model(args):
    transformer = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=128)
    return T5Wrapper(transformer, tokenizer)


def build_train_iterator(model, args):
    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": args.max_samples,
    }
    dataset_reader = get_datasetReader(args.train_dataset, dataset_kwargs)
    create_dataset = lambda dataset: PytorchDataset(dataset, model.tokenizer, "cuda")
    batcher = Batcher(
        dataset_reader,
        create_dataset,
        train_batchSize=args.batch_size,
        eval_batchSize=args.batch_size * 2,
        world_size=None,
        device=None,
    )
    return batcher.get_trainBatches("train", template_idx=0)


def finetune(args):
    assert not (
        args.grad_cross_matrix and args.num_grad_accumulation > 1
    ), "--grad-cross-matrix is incompatible with gradient accumulation > 1"

    os.makedirs(args.output_dir, exist_ok=True)
    save_args(args)

    pretrained_path = osp.join(args.output_dir, "pretrained.pt")
    finetuned_path = osp.join(args.output_dir, "finetuned.pt")
    if (
        osp.exists(pretrained_path)
        and osp.exists(finetuned_path)
        and not args.overwrite
    ):
        print(f"Skipping fine-tuning because {finetuned_path} already exists.")
        return
    wandb_run = init_wandb(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Building model and tokenizer.")
    model = build_model(args)
    model.save(pretrained_path)

    lora = args.finetuning_mode == "lora"
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
    train_iterator = build_train_iterator(model, args)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    grad_cross_tracker = (
        GradCrossTermTracker(model) if args.grad_cross_matrix else None
    )

    best_val_acc = -1.0
    bad_checkpoints = 0
    saved_best = False
    start_time = time.time()
    model.train()
    for i in range(args.num_batches * args.num_grad_accumulation):
        iter_start = time.time()
        batch = next(train_iterator)
        data_time = time.time() - iter_start

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
            loss = loss / args.num_grad_accumulation

        loss.backward()
        if (i + 1) % args.num_grad_accumulation == 0:
            if grad_cross_tracker is not None:
                grad_cross_tracker.step()
            optimizer.step()
            optimizer.zero_grad()

        step = (i + 1) // args.num_grad_accumulation
        if (i + 1) % (10 * args.num_grad_accumulation) == 0:
            elapsed = time.time() - start_time
            batch_time = time.time() - iter_start
            print(
                f"Train Iteration: {step} [{100 * step / args.num_batches:.0f}% "
                f"{step}/{args.num_batches}]\tLoss: {loss.item():.6f}\t"
                f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                f"Best val acc: {100 * best_val_acc:.2f}%\t"
                f"Elapsed {_format_duration(elapsed)}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/data_time": data_time,
                        "train/batch_time": batch_time,
                        "train/best_val_acc": best_val_acc,
                        "train/elapsed_seconds": elapsed,
                        "train/progress": step / args.num_batches,
                    },
                    step=step,
                )

        if (
            args.checkpoint_every > 0
            and step > 0
            and (i + 1) % (args.checkpoint_every * args.num_grad_accumulation) == 0
        ):
            checkpoint_path = osp.join(args.output_dir, f"checkpoint_{step}.pt")
            if not lora:
                if grad_cross_tracker is None:
                    model.save(checkpoint_path)
                else:
                    save_model_without_hooks(model, checkpoint_path)
                _prune_checkpoints(args.output_dir, args.keep_checkpoints)

            val_acc = eval_single_dataset(
                "validation", model, model.tokenizer, args.train_dataset, args
            )["top1"]
            improved = val_acc > best_val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                bad_checkpoints = 0
                if not lora:
                    if grad_cross_tracker is None:
                        model.save(finetuned_path)
                    else:
                        save_model_without_hooks(model, finetuned_path)
                saved_best = True
            else:
                bad_checkpoints += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/top1": val_acc,
                        "val/best_top1": best_val_acc,
                        "val/improved": int(improved),
                        "val/bad_checkpoints": bad_checkpoints,
                    },
                    step=step,
                )
            if args.early_stop and bad_checkpoints >= args.patience:
                print(f"Early stopping at step {step}.", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/early_stop": 1}, step=step)
                break
            model.train()

        if args.max_steps is not None and step >= args.max_steps:
            break

    if lora:
        model.transformer = model.transformer.merge_and_unload()
        if grad_cross_tracker is None:
            model.save(finetuned_path)
        else:
            save_model_without_hooks(model, finetuned_path)
    elif not saved_best:
        if grad_cross_tracker is None:
            model.save(finetuned_path)
        else:
            save_model_without_hooks(model, finetuned_path)

    if grad_cross_tracker is not None:
        grad_cross_tracker.save(args.output_dir)
        grad_cross_tracker.remove_hooks()

    model.cpu()
    del model
    gc.collect()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.cache_dir)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.grad_cross_matrix and args.num_grad_accumulation is None:
        args.num_grad_accumulation = 1
    if args.batch_size is None or args.num_grad_accumulation is None:
        batch_size, grad_accum = _batch_defaults(args.model)
        args.batch_size = args.batch_size or batch_size
        args.num_grad_accumulation = args.num_grad_accumulation or grad_accum
    args.num_batches = args.max_steps or 75000
    finetune(args)
