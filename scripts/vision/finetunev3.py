import copy
import os
import time
from typing import Dict, Optional

import torch

# --finetuning-mode=standard   --model=ViT-B-16   --world-size=1   --num-workers=1   --openclip-cachedir=$SCRATCH/openclip   --data-location=data/vision   --save=$SCRATCH/actmat/checkpoints/vision
from src.args import parse_arguments
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.vision.eval import eval_single_dataset
from src.vision.heads import get_classification_head
from src.vision.linearize import LinearizedImageEncoder
from src.vision.modeling import ImageClassifier, ImageEncoder, apply_lora, merge_lora
from src.mhas import swap_mha, unswap_mha
from src.utils import LabelSmoothing, cosine_lr, get_prefix, resolve_run_dir


class GradCrossTermTracker:
    def __init__(self, model, *args, **kwargs):
        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }

        self.gbar = dict()
        self.sbar = dict()
        self.stilde = dict()

        for name, mod in self.layers.items():
            Do, Di = mod.weight.shape
            self.gbar[name] = torch.zeros(Do, Di)
            self.sbar[name] = torch.zeros(Di, Di)
            self.stilde[name] = torch.zeros(Di, Di)

        # Hook storage
        self._activations = {}
        self._output_grads = {}
        self._hooks = []

        for name, module in self.layers.items():
            self._hooks.append(module.register_forward_hook(self._make_fwd_hook(name)))
            self._hooks.append(
                module.register_full_backward_hook(self._make_bwd_hook(name))
            )

        print(f"\n=== Tracking {len(self.layers)} layers for grad cross-terms ===")
        for name, module in self.layers.items():
            print(f"  {name}: weight {tuple(module.weight.shape)}")
        print()

    def _make_fwd_hook(self, name):
        def hook(module, input, output):
            self._activations[name] = input[0].detach()

        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            self._output_grads[name] = grad_output[0].detach()

        return hook

    def step(self):
        """Compute per-sample gradients for one batch and accumulate statistics.

        (Optional): could also compute gw like this
            gw = torch.einsum("bto,bti->btio", gy, z)
        """

        for name, module in self.layers.items():
            # (B, T, Di)
            z = self._activations[name].float()  # (B, T, Di)
            gy = self._output_grads[name].float()  # (B, T, Do)
            gw_bar = module.weight.grad.detach().float()
            gynorm2 = gy.pow(2).sum(-1)  # (B,T)

            B, T, Di = z.shape  # NOTE: verify this is the actual shape
            z_flat = z.reshape(-1, Di)  # (B*T, Di)
            gnorm_flat = gynorm2.reshape(-1)  # (B*T,)
            self.gbar[name] += (gw_bar / (B * T)).cpu()
            self.sbar[name] += (
                (z_flat * gnorm_flat.unsqueeze(-1)).T @ z_flat / (B * T)
            ).cpu()
            self.stilde[name] += (
                (z_flat.T @ z_flat) / (B * T) * gnorm_flat.mean()
            ).cpu()

    def save(self, ckpdir):
        """Print summary and save per-layer results to disk."""
        os.makedirs(ckpdir, exist_ok=True)
        for attr in ["gbar", "sbar", "stilde"]:
            torch.save(getattr(self, attr), os.path.join(ckpdir, f"{attr}.pt"))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "standard",
        "lora",
    ], "Only linear, standard, and lora fine-tuning are supported."

    assert not (
        args.grad_cross_matrix and args.num_grad_accumulation > 1
    ), "--grad-cross-matrix is incompatible with gradient accumulation > 1"

    linearized_finetuning = args.finetuning_mode == "linear"
    lora_finetuning = args.finetuning_mode == "lora"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")
    if lora_finetuning:
        print("Using LoRA fine-tuning.")

    # Check if checkpoints already exist
    prefix = get_prefix(args.finetuning_mode)
    ft_path = os.path.join(ckpdir, f"{prefix}finetuned.pt")
    zs_path = os.path.join(ckpdir, "pretrained.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path) and not args.overwrite:
        print(f"Skipping fine-tuning because {ft_path} already exists.")
        cleanup_ddp()
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        if linearized_finetuning:
            image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        else:
            image_encoder = ImageEncoder(args)

    # Save the pretrained encoder before applying LoRA (reuses standard pretrained.pt)
    if lora_finetuning and args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        zs_path = os.path.join(ckpdir, "pretrained.pt")
        if not os.path.exists(zs_path):
            image_encoder.save(zs_path)

    if lora_finetuning:
        image_encoder = apply_lora(
            image_encoder,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            target_modules="all-linear",
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    # Save zeroshot before MHA swap so checkpoint is in standard format.
    # (LoRA zeroshot is already saved above, before LoRA is applied.)
    if args.save is not None and is_main_process() and not lora_finetuning:
        os.makedirs(ckpdir, exist_ok=True)
        model.image_encoder.save(os.path.join(ckpdir, "pretrained.pt"))

    # Swap nn.MultiheadAttention -> MultiHeadAttentionSplit so forward hooks
    # fire on per-projection Linear layers (needed for grad cross-term tracking).
    if args.grad_cross_matrix:
        swap_mha(model.image_encoder)

    model = model.cuda()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(
            f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    if lora_finetuning and is_main_process():
        print("\n=== LoRA Config ===")
        print(f"  rank: {args.lora_rank}")
        print(f"  alpha: {args.lora_alpha}")
        print(f"  dropout: {args.lora_dropout}")
        print(f"  target_modules: all-linear")
        print("=== LoRA Model Architecture ===")
        print(image_encoder)
        print("===============================\n")

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
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

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,  # Set False when all params are used (faster, avoids DDP warning)
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    if is_main_process():
        print(f"Total steps: {args.epochs * num_batches // args.num_grad_accumulation}")

    grad_cross_tracker = None
    if args.grad_cross_matrix and is_main_process():
        grad_cross_tracker = GradCrossTermTracker(ddp_model.module.image_encoder)

    run_start_time = time.perf_counter()
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
                if grad_cross_tracker is not None:
                    grad_cross_tracker.step()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if args.max_steps is not None and step >= args.max_steps:
                break

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                model_path = os.path.join(ckpdir, f"{prefix}checkpoint_{step}.pt")
                enc = ddp_model.module.image_encoder
                if args.grad_cross_matrix:
                    enc = copy.deepcopy(enc).cpu()
                    for m in enc.modules():
                        m._forward_hooks.clear()
                        m._backward_hooks.clear()
                    unswap_mha(enc)
                enc.save(model_path)
                print(f"Saved checkpoint to {model_path}", flush=True)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                run_elapsed = time.perf_counter() - run_start_time
                print(
                    f"Train Epoch: {epoch}/{args.epochs} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"  # noqa: E501
                    f"Elapsed {_format_duration(run_elapsed)}",  # noqa: E501
                    flush=True,
                )

        if args.max_steps is not None and step >= args.max_steps:
            break

    if grad_cross_tracker is not None:
        grad_cross_tracker.save(ckpdir)

    # FIXME: Make this work with DDP.
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder

        # Merge LoRA weights back into the base model before eval/save
        if lora_finetuning:
            image_encoder = merge_lora(image_encoder)

        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        zs_path = os.path.join(ckpdir, "pretrained.pt")
        ft_path = os.path.join(ckpdir, f"{prefix}finetuned.pt")
        enc_to_save = image_encoder
        if args.grad_cross_matrix:
            enc_to_save = copy.deepcopy(image_encoder).cpu()
            for m in enc_to_save.modules():
                m._forward_hooks.clear()
                m._backward_hooks.clear()
            unswap_mha(enc_to_save)
        enc_to_save.save(ft_path)
        cleanup_ddp()
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    # 20-dataset suite from Wang et al. (nik-dim/tall_masks). The first 8 match
    # the standard task-arithmetic benchmark (Ilharco et al.).
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "Food101",
        "Flowers102",
        "FER2013",
        "PCAM",
        "OxfordIIITPet",
        "RenderedSST2",
        "EMNIST",
        "FashionMNIST",
        "KMNIST",
    ]
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

    args = parse_arguments()

    args.save = resolve_run_dir(args)

    if args.train_dataset is not None:
        train_datasets = [ds.strip() for ds in args.train_dataset]

    for dataset in train_datasets:
        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        if not args.grad_cross_matrix and args.max_samples is None:
            args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        if args.model == "ViT-L-14":
            # tracker is also incompatible with grad accumulation (see assertion in finetune()),
            if args.grad_cross_matrix:
                args.batch_size = 16
                args.num_grad_accumulation = 1
            else:
                args.batch_size = 64
                args.num_grad_accumulation = 2
        else:
            args.batch_size = 128
            args.num_grad_accumulation = 1

        print("=" * 100)
        print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
