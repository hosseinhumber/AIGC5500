"""LoRA for Module 06."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear and add the low-rank update (alpha / r) * B A.

    A is (r, in_features), random. B is (out_features, r), zero. So the layer
    starts exactly equal to the wrapped layer.
    """

    def __init__(self, base, r, alpha=None, zero_B=True):
        super().__init__()
        self.base, self.r = base, r
        self.alpha = alpha if alpha is not None else r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r) if zero_B
                              else torch.randn(base.out_features, r) * 0.01)
        self.scale = self.alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merged(self):
        """A single nn.Linear holding W + (alpha / r) B A, with no extra cost at inference."""
        out = nn.Linear(self.base.in_features, self.base.out_features)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        out.bias.copy_(self.base.bias)
        return out


def add_lora(mlp, r, alpha=None):
    """Replace the hidden Linear layers of a Sequential MLP with LoRA layers.

    The last Linear (the head) stays frozen and unwrapped. Returns a new Sequential;
    the original is not modified.
    """
    import copy
    net = copy.deepcopy(mlp)
    linears = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
    for i in linears[:-1]:
        net[i] = LoRALinear(net[i], r, alpha)
    for p in net[linears[-1]].parameters():
        p.requires_grad_(False)
    return net


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
