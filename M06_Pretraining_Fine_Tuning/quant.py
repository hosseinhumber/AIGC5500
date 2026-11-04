"""Weight quantization for Module 06: absmax (whole tensor or blocks) and a simplified normal-float code."""
import torch


def absmax_quantize(w, bits, block=None):
    """Quantize to signed integers with one scale per tensor, or per block of `block` weights.

    Returns (integer codes, scales, dequantized weights). s = max|w| / (2^(bits-1) - 1).
    """
    levels = 2 ** (bits - 1) - 1
    flat = w.reshape(-1)
    groups = flat.reshape(1, -1) if block is None else flat.reshape(-1, block)
    scales = groups.abs().amax(dim=1, keepdim=True) / levels
    scales = scales.clamp_min(1e-12)
    codes = torch.round(groups / scales).clamp(-levels, levels)
    deq = (codes * scales).reshape(w.shape)
    return codes.to(torch.int8).reshape(w.shape if block is None else (-1, block)), scales, deq


def normal_float_levels(bits=4):
    """2^bits levels at the quantiles of a standard normal, scaled so the largest is 1."""
    n = 2 ** bits
    probs = (torch.arange(n) + 0.5) / n
    levels = torch.distributions.Normal(0.0, 1.0).icdf(probs)
    return levels / levels.abs().max()


def normal_float_quantize(w, block=64, bits=4):
    """Simplified normal-float: one absmax scale per block, then the nearest of 16 normal-quantile levels."""
    levels = normal_float_levels(bits)
    groups = w.reshape(-1).reshape(-1, block)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = (groups / scales).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
    deq = (levels[idx] * scales).reshape(w.shape)
    return idx.to(torch.uint8), scales, deq


def relative_error(w, w_hat):
    return ((w - w_hat).norm() / w.norm()).item()
