"""Compute and memory arithmetic for Module 06. Memory figures exclude activations."""

GB = 1e9


def compute_flops(n_params, n_tokens):
    """About 6 floating-point operations per parameter per token (2 forward, 4 backward)."""
    return 6.0 * n_params * n_tokens


def training_time_seconds(n_params, n_tokens, flops_per_second=100e12):
    return compute_flops(n_params, n_tokens) / flops_per_second


def full_finetune_bytes(n_params):
    """Adam in mixed precision: 2 (weights) + 2 (grads) + 4 (fp32 master) + 4 + 4 (Adam moments)."""
    return 16 * n_params


def lora_params(n_layers, width, rank, matrices_per_layer=4):
    """Trainable parameters when every adapted matrix is width x width."""
    return n_layers * matrices_per_layer * rank * (width + width)


def lora_bytes(n_params, n_trainable, weight_bytes=2.0, block=None, scale_bytes=4):
    """Frozen base stored at weight_bytes per weight, plus 16 bytes per trainable parameter.

    If block is given, add one scale of scale_bytes for every block of weights.
    """
    base = weight_bytes * n_params
    if block:
        base += scale_bytes * n_params / block
    return base + 16 * n_trainable
