import re

import torch

from src import merging
from src.core.experts import Expert


def merge_experts(
    base_expert: Expert,
    expert_experts,
    merged_expert: Expert,
    merge_method,
    ignore_keep_pt=None,
    ignore_mean=None,
    device=None,
    merge_kwargs=None,
):
    """Merge experts layer-by-layer through the generic Expert interface."""
    experts = list(expert_experts)
    if len(experts) == 0:
        raise ValueError("merge_experts requires at least one expert")

    merge_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    merge_fn = getattr(merging, "merge_" + merge_method)

    with torch.no_grad():
        for layer_name in base_expert.get_layers():
            metadata = base_expert.get_layer_metadata(layer_name)
            w_0 = base_expert.get_layer_params(layer_name)

            if ignore_keep_pt and re.search(ignore_keep_pt, layer_name):
                w_merged = w_0
            else:
                w_list = []
                stat_fetcher_maps = []
                for expert in experts:
                    w_list.append(expert.get_layer_params(layer_name))
                    stat_fetcher_maps.append(expert.get_stat_fetcher_map(layer_name))

                if w_0.ndim != 2 or (
                    ignore_mean and re.search(ignore_mean, layer_name)
                ):
                    print(
                        f"[IGNORE-MEAN] forcing mean merge for layer: {layer_name}",
                        flush=True,
                    )
                    w_merged = torch.stack(w_list).mean(0)
                else:
                    w0 = w_0.to(merge_device).float()
                    d = torch.stack([w.to(merge_device).float() - w0 for w in w_list])
                    merged_delta = merge_fn(
                        d=d,
                        stat_fetcher_maps=stat_fetcher_maps,
                        **(merge_kwargs or {}),
                    )
                    w_merged = (w0 + merged_delta).to(w_0.dtype).cpu()

            merged_expert.save_layer_params(w_merged, layer_name, metadata=metadata)

    merged_expert.flush()
    return merged_expert
