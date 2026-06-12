import os

import torch


class GradCrossTermTracker:
    def __init__(self, model):
        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }

        self.gbar = {}
        self.sbar = {}
        self.stilde = {}
        for name, module in self.layers.items():
            do, di = module.weight.shape
            self.gbar[name] = torch.zeros(do, di)
            self.sbar[name] = torch.zeros(di, di)
            self.stilde[name] = torch.zeros(di, di)

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
        def hook(module, inputs, output):
            if not inputs:
                return
            self._activations[name] = inputs[0].detach()

        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not grad_output or grad_output[0] is None:
                return
            self._output_grads[name] = grad_output[0].detach()

        return hook

    def step(self):
        for name, module in self.layers.items():
            if (
                name not in self._activations
                or name not in self._output_grads
                or module.weight.grad is None
            ):
                continue

            z = self._activations[name].float()
            gy = self._output_grads[name].float()
            gw_bar = module.weight.grad.detach().float()

            di = z.shape[-1]
            do = gy.shape[-1]
            z_flat = z.reshape(-1, di)
            gy_flat = gy.reshape(-1, do)
            if z_flat.shape[0] != gy_flat.shape[0]:
                continue

            n = z_flat.shape[0]
            gynorm2 = gy_flat.pow(2).sum(-1)
            self.gbar[name] += (gw_bar / n).cpu()
            self.sbar[name] += ((z_flat * gynorm2.unsqueeze(-1)).T @ z_flat / n).cpu()
            self.stilde[name] += ((z_flat.T @ z_flat) / n * gynorm2.mean()).cpu()

        self._activations.clear()
        self._output_grads.clear()

    def save(self, output_dir, step=None):
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{step}" if step is not None else ""
        for attr in ("gbar", "sbar", "stilde"):
            torch.save(
                getattr(self, attr), os.path.join(output_dir, f"{attr}{suffix}.pt")
            )

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
