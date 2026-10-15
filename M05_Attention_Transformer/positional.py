"""Positional encodings for Module 05: sinusoidal (fixed) and learned."""
import math

import torch
import torch.nn as nn


def sinusoidal_encoding(max_len, d_model):
    """Fixed encoding. Pair (2i, 2i+1) of position p holds sin and cos of p / 10000^(2i/d_model).
    Returns a tensor of shape (max_len, d_model). Needs no training and is defined for any length."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe


class LearnedPositions(nn.Module):
    """One trainable vector per position, up to max_len. Positions beyond max_len do not exist."""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)

    def forward(self, length):
        return self.table(torch.arange(length, device=self.table.weight.device))
