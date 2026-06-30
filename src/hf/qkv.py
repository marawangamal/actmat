Q_SUFFIX = ".self_attn.q_proj.weight"
K_SUFFIX = ".self_attn.k_proj.weight"
V_SUFFIX = ".self_attn.v_proj.weight"
QKV_SUFFIX = ".self_attn.qkv_proj.weight"


def copy_to_packed_safetensor_index(weight_map, tensor_shape_fn, verbose=True):
    qkv_prefixes = {
        prefix
        for layer_name in weight_map
        if (prefix := _qkv_prefix(layer_name)) is not None
    }
    complete_prefixes = {
        prefix
        for prefix in qkv_prefixes
        if prefix + Q_SUFFIX in weight_map
        and prefix + K_SUFFIX in weight_map
        and prefix + V_SUFFIX in weight_map
    }

    packed_weight_map = {}
    emitted_prefixes = set()
    for layer_name, shard_name in weight_map.items():
        prefix = _qkv_prefix(layer_name)
        if prefix in complete_prefixes:
            if prefix not in emitted_prefixes:
                native_names = (
                    prefix + Q_SUFFIX,
                    prefix + K_SUFFIX,
                    prefix + V_SUFFIX,
                )
                shard_filenames = {weight_map[name] for name in native_names}
                if len(shard_filenames) != 1:
                    raise ValueError(
                        f"Cannot pack QKV layer {prefix}: tensors span multiple "
                        f"safetensor shards: {sorted(shard_filenames)}"
                    )

                row_counts = {tensor_shape_fn(name)[0] for name in native_names}
                if len(row_counts) != 1:
                    raise ValueError(
                        f"Cannot pack QKV layer {prefix}: Q/K/V row counts differ: "
                        f"{sorted(row_counts)}"
                    )

                packed_weight_map[prefix + QKV_SUFFIX] = [
                    {"tensor_name": name, "shard_filename": weight_map[name]}
                    for name in native_names
                ]
                emitted_prefixes.add(prefix)
            continue

        packed_weight_map[layer_name] = shard_name

    if verbose:
        incomplete_prefixes = qkv_prefixes - complete_prefixes
        print(
            "Packed QKV safetensor index: "
            f"{len(weight_map)} native tensors -> {len(packed_weight_map)} tensors; "
            f"packed {len(complete_prefixes)} QKV groups; "
            f"left {len(incomplete_prefixes)} incomplete QKV groups native."
        )

    return packed_weight_map


def _qkv_prefix(layer_name):
    for suffix in (Q_SUFFIX, K_SUFFIX, V_SUFFIX):
        if layer_name.endswith(suffix):
            return layer_name[: -len(suffix)]
    return None
