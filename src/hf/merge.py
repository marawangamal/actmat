# Get layer names:
def load_layer_params(model_dir, layer_name)
    # loads layer from hf dir with pottentially many shards
    pass

def save_layer_params(w_merged, shard_name, layer_name)
    # saves into hf compatible safetensors layer_file at key layer_name
    pass

layer_to_file = json.loads(osp.join(os.environ("HF_HOME"), args.base_model))
merged_model_path = osp.join(args.merge_dir, args.merge_method)
for layer_name in json.loads(model_index):
    w_list = []
    w_0 = load_layer_params(args.base_model, layer_name)

    if layer_name matches regex in args.ignore-keep-pt:
        save_layer_params(w_0, shard_name, layer_name)
        continue

    for model_name_or_path in args.expert_models:
        w_t = load_layer_params(model_name_or_path, layer_name)
        w_list.append(wt)

    if layer_name matches regex in args.ignore-mean:
        w_merged = apply_merge(w_list, w_0, args)
        save_layer_params(w_merged, shard_name, layer_name)
        continue

    w_merged = apply_merge(w_list, w_0, args)

    # shard_name = ...
    save_layer_params(w_merged, shard_name, layer_name)
