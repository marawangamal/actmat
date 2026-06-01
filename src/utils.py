import os
import pickle

import numpy as np
import torch


def get_prefix(finetuning_mode):
    return {"linear": "linear_", "lora": "lora_"}.get(finetuning_mode, "")


def group_dir(group):
    """The group path level: 'group-<group>' (default 'group-main').

    `group` is the experiment-suite axis carried as a real path level between
    <model> and the {experts|multitask|merged} subdirs (vision: 8/14/20; OLMo:
    rl-zero/polyglot; everything else: main). Single source of truth so writers
    and readers agree.
    """
    return f"group-{group or 'main'}"


def resolve_run_dir(args):
    """Resolve the per-run checkpoint directory:
    <save>/<model>/group-<group>[/seed_X][/max_steps_Y].

    Defaults args.save to 'artifacts/checkpoints' when unset. Used by both
    writers (finetune scripts that create the dir) and readers
    (eval/cov/fisher scripts that load from it) so the layout stays in
    one place.
    """
    base = args.save if args.save is not None else "artifacts/checkpoints"
    run = os.path.join(base, args.model, group_dir(getattr(args, "group", "main")))
    if getattr(args, "seed", None) is not None:
        run = os.path.join(run, f"seed_{args.seed}")
    if getattr(args, "max_steps", None) is not None:
        run = os.path.join(run, f"max_steps_{args.max_steps}")
    if getattr(args, "max_samples", None) is not None:
        run = os.path.join(run, f"max_samples_{args.max_samples}")
    return run


# --- Structured artifacts layout -------------------------------------------
# Single source of truth for the on-disk convention. `save` here is the
# per-model run dir (i.e. the output of resolve_run_dir, <base>/<model>/group-<g>);
# `results_dir` is the bucket base, artifacts/results.
#
#   checkpoints:  <save>/experts/<dataset>Val/       per-expert checkpoints
#                 <save>/multitask/                  MTL checkpoint
#   results:      <results_dir>/<model>/group-<g>/merged/<method>[-<mode>]/[lora_]metrics.json
#                 <results_dir>/<model>/group-<g>/experts/[lora_]metrics.json
#                 <results_dir>/<model>/group-<g>/pretrained/[lora_]metrics.json
#                 <results_dir>/<model>/group-<g>/multitask/[lora_]metrics.json
# The per-suite axis (vision 8/14/20-task; OLMo rl-zero/polyglot) is the
# `group-<g>` path level — uniform across the vision/language/OLMo pipelines.
# resolve_run_dir injects it for checkpoints; the results builders below take
# it as the `group` argument.


def expert_dir(save, dataset, val_suffix=True):
    """Per-expert checkpoint directory: <save>/experts/<dataset>[Val].

    Vision appends the "Val" split suffix to the dataset name (its checkpoints are
    keyed by the val variant); language/OLMo pass val_suffix=False to use the bare
    dataset name.
    """
    leaf = f"{dataset}Val" if val_suffix and not dataset.endswith("Val") else dataset
    return os.path.join(save, "experts", leaf)


def head_path(save, dataset):
    """Classification head, co-located with its expert: <save>/experts/<dataset>Val/head.pt.

    The head is a per-dataset (FT-mode-agnostic) artifact, so it lives in the same
    dir as that dataset's expert checkpoints rather than a separate top-level dir.
    """
    return os.path.join(expert_dir(save, dataset), "head.pt")


def sanitize_hf_id(name_or_path):
    """HF id / local path -> safe directory leaf (the repo/model basename).

    e.g. "Qwen/Qwen2.5-Math-1.5B" -> "Qwen2.5-Math-1.5B"; a local path keeps its
    final component. Used to key per-expert stats sidecars in the HF pathway,
    whose weights live on the Hub rather than in our tree.
    """
    return os.path.basename(str(name_or_path).rstrip("/"))


def _merge_mode_str(merge_mode):
    return f"-{merge_mode}" if merge_mode and merge_mode != "d" else ""


def merged_results_path(results_dir, model, method, merge_mode, prefix="", group="main"):
    """<results_dir>/<model>/group-<group>/merged/<method>[-<mode>]/[prefix]metrics.json."""
    return os.path.join(
        results_dir,
        model,
        group_dir(group),
        "merged",
        f"{method}{_merge_mode_str(merge_mode)}",
        f"{prefix}metrics.json",
    )


def experts_results_path(results_dir, model, prefix="", group="main"):
    """<results_dir>/<model>/group-<group>/experts/[prefix]metrics.json."""
    return os.path.join(results_dir, model, group_dir(group), "experts", f"{prefix}metrics.json")


def pretrained_results_path(results_dir, model, prefix="", group="main"):
    """<results_dir>/<model>/group-<group>/pretrained/[prefix]metrics.json (zero-shot baseline)."""
    return os.path.join(results_dir, model, group_dir(group), "pretrained", f"{prefix}metrics.json")


def multitask_results_path(results_dir, model, prefix="", group="main"):
    """<results_dir>/<model>/group-<group>/multitask/[prefix]metrics.json."""
    return os.path.join(results_dir, model, group_dir(group), "multitask", f"{prefix}metrics.json")


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu", weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)
