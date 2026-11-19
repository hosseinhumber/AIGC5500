"""Uniform quantization from scratch: post-training quantization, simulated integer inference, and quantization-aware training."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress import mlp, train, accuracy, count_params


def qparams(x, bits=8, per_channel=False, symmetric=True, clip=None):
    """Scale (and zero point) that map the range of x onto signed integers of `bits` bits.
    per_channel=True gives one scale per output row (dim 0). `clip` limits the range to a percentile of |x|."""
    qmax = 2 ** (bits - 1) - 1
    a = x.abs()
    if per_channel:
        m = a.flatten(1).max(1).values if clip is None else torch.quantile(a.flatten(1), clip, dim=1)
        scale = (m / qmax).clamp_min(1e-12).view(-1, *([1] * (x.dim() - 1)))
    else:
        m = a.max() if clip is None else torch.quantile(a.flatten(), clip)
        scale = (m / qmax).clamp_min(1e-12)
    return scale, qmax


def quantize(x, scale, qmax):
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax)


def fake_quant(x, bits=8, per_channel=False, clip=None):
    scale, qmax = qparams(x.detach(), bits, per_channel, clip=clip)
    return quantize(x, scale, qmax) * scale


class FakeQuantSTE(torch.autograd.Function):
    """Round in the forward pass, pass the gradient straight through in the backward pass."""
    @staticmethod
    def forward(ctx, x, scale, qmax):
        return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def ste_quant(x, bits, per_channel=False):
    scale, qmax = qparams(x.detach(), bits, per_channel)
    return FakeQuantSTE.apply(x, scale, qmax)


def quantize_weights(model, bits=8, per_channel=True):
    """Return a copy of a Sequential MLP whose Linear weights are rounded to `bits` bits (weight-only, simulated)."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for l in m:
            if isinstance(l, nn.Linear):
                l.weight.copy_(fake_quant(l.weight, bits, per_channel))
    return m


def weight_error(w, bits, per_channel):
    q = fake_quant(w, bits, per_channel)
    return ((q - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()


def int8_inference(model, X, calib, per_channel=True, act_clip=None):
    """Genuine integer arithmetic: int8 weights and int8 activations, int32 accumulation, float rescale between layers.
    Activation scales come from the calibration set. Returns the float logits."""
    lin = [l for l in model if isinstance(l, nn.Linear)]
    with torch.no_grad():
        # calibrate the input to every Linear layer
        acts, h = [], calib
        for i, l in enumerate(lin):
            acts.append(h); h = l(h)
            if i < len(lin) - 1: h = torch.relu(h)
        h = X
        for i, l in enumerate(lin):
            sw, qm = qparams(l.weight, 8, per_channel)
            wq = quantize(l.weight, sw, qm).to(torch.int8).numpy()
            sa, _ = qparams(acts[i], 8, False, clip=act_clip)
            aq = quantize(h, sa, 127).to(torch.int8).numpy()
            acc = aq.astype(np.int32) @ wq.astype(np.int32).T                        # integer matmul, int32 accumulator
            sw_row = sw.view(1, -1).numpy() if per_channel else float(sw)
            h = torch.from_numpy(acc.astype(np.float32) * (float(sa) * sw_row)) + l.bias
            if i < len(lin) - 1: h = torch.relu(h)
    return h


def model_bytes(model, bits=32, scales=False):
    """Storage for the weights at `bits` bits per weight (biases stay 32-bit), plus 4 bytes per output row if per-channel scales."""
    b = 0
    for l in model:
        if isinstance(l, nn.Linear):
            b += l.weight.numel() * bits / 8 + l.bias.numel() * 4 + (l.out_features * 4 if scales else 0)
    return int(b)


class QATLinear(nn.Module):
    def __init__(self, lin, bits):
        super().__init__(); self.lin, self.bits = lin, bits

    def forward(self, x):
        return F.linear(x, ste_quant(self.lin.weight, self.bits, True), self.lin.bias)


def qat_model(model, bits):
    layers = []
    for l in model:
        layers.append(QATLinear(copy.deepcopy(l), bits) if isinstance(l, nn.Linear) else copy.deepcopy(l))
    return nn.Sequential(*layers)


def freeze_qat(qm, bits):
    """Collapse a QAT model back into an ordinary MLP holding the rounded weights."""
    layers = []
    for l in qm:
        if isinstance(l, QATLinear):
            n = nn.Linear(l.lin.in_features, l.lin.out_features)
            with torch.no_grad():
                n.weight.copy_(fake_quant(l.lin.weight, bits, True)); n.bias.copy_(l.lin.bias)
            layers.append(n)
        else: layers.append(l)
    return nn.Sequential(*layers)


def train_qat(model, X, y, bits, epochs=20, lr=1e-3, seed=0):
    qm = qat_model(model, bits)
    train(qm, X, y, epochs=epochs, lr=lr, seed=seed)
    return freeze_qat(qm, bits)


def add_outlier_channel(model, layer=0, channel=3, factor=30.0):
    """A function-preserving change: scale one hidden neuron's incoming weights and bias by `factor` and its outgoing
    weights by 1/factor. The float network is unchanged, but its weight range is now dominated by one row."""
    m = copy.deepcopy(model)
    lin = [l for l in m if isinstance(l, nn.Linear)]
    with torch.no_grad():
        lin[layer].weight[channel] *= factor; lin[layer].bias[channel] *= factor
        lin[layer + 1].weight[:, channel] /= factor
    return m


# ---------------------------------------------------------------- experiments
def exp_bits(X, y, Xte, yte, bits_list=(8, 6, 4, 3, 2), seeds=range(3), qat_epochs=20):
    out = {b: {"ptq_tensor": [], "ptq_channel": [], "qat": []} for b in bits_list}
    out["float"] = []
    for seed in seeds:
        torch.manual_seed(seed)
        base = train(mlp([64, 512, 512, 10]), X, y, epochs=60, seed=seed)
        out["float"].append(accuracy(base, Xte, yte))
        for b in bits_list:
            out[b]["ptq_tensor"].append(accuracy(quantize_weights(base, b, False), Xte, yte))
            out[b]["ptq_channel"].append(accuracy(quantize_weights(base, b, True), Xte, yte))
            out[b]["qat"].append(accuracy(train_qat(base, X, y, b, epochs=qat_epochs, seed=seed), Xte, yte))
    return out


def exp_outlier(X, y, Xte, yte, seeds=range(3), factor=30.0):
    out = {"float": [], "tensor8": [], "channel8": [], "tensor6": [], "channel6": [], "err_tensor": [], "err_channel": []}
    for seed in seeds:
        torch.manual_seed(seed)
        base = train(mlp([64, 512, 512, 10]), X, y, epochs=60, seed=seed)
        m = add_outlier_channel(base, 0, 3, factor)
        out["float"].append(accuracy(m, Xte, yte))
        for b in (8, 6):
            out[f"tensor{b}"].append(accuracy(quantize_weights(m, b, False), Xte, yte))
            out[f"channel{b}"].append(accuracy(quantize_weights(m, b, True), Xte, yte))
        w = [l for l in m if isinstance(l, nn.Linear)][0].weight.detach()
        out["err_tensor"].append(weight_error(w, 6, False)); out["err_channel"].append(weight_error(w, 6, True))
    return out
