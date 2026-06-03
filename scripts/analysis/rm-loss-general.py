import torch

layers = []  # get layer keys
expert_paths = []  # get expert paths


class Expert:
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def get_layers(self):
        pass

    def get_layer_params(self):
        pass

    def get_layer_cov(self):
        pass


class HFExpert(Expert):
    pass


def compute_rm_loss(w_test, w_t, c_t):
    """Compute the RegMean loss for linear layer at a given w

    Args:
        w: Shape: (Do, Di)
        w_t: Shape: (T, Do, Di)
        c: Shape: (T, Di, Di)
    """
    w_test = w_test.unsqueeze(0)
    loss_1 = tr_abt(w_test @ c_t, w_test)
    loss_2 = tr_abt(w_t @ c_t, w_t)
    loss_3 = tr_abt(w_t @ c_t, w_test)
    return loss_1 + loss_2 - 2 * loss_3


def merge_actmat(w0: torch.Tensor, d: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_regmean(w0: torch.Tensor, d: torch.Tensor, c: torch.Tensor, **kwargs):
    return w0 + (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_mean(w0: torch.Tensor, d: torch.Tensor, **kwargs):
    return w0 + d.mean(0)


rows = []
configs = [
    {
        "model": "t5-base",
        "experts-path": "artifacts/checkpoints/t5-base/group-main/experts",
        "base-path": "artifacts/checkpoints/t5-base/group-main/experts",
        "type": "basic",
    },
    {
        "model": "Olmo-3-7b",
        "experts-path": "artifacts/checkpoints/Olmo-3-7b/group-main/experts",
        "base-path": "artifacts/checkpoints/Olmo-3-7b/group-main/experts",
        "type": "hf",
    },
]
for cfg in configs:
    layers = base.get_layers()
    for l in layers:
        w_0 = base.get_layer_params(l)  # (Do, Di)
        if not w_.ndim == 2:
            pass
        w_list = []
        for expert in experts:
            w_t = expert.get_layer_params(l)
            c_t = expert.get_layer_cov(l)
            w_list.append(w_t)
        for merge_method, merge_func in merge_configs:
            w_star = merge_func(w_0, torch.stack(w_list))
            loss = compute_rm_loss(w_star, w, c)
            rows.append({"model": model, "layer": l, "loss": loss})
