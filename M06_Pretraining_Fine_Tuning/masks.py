"""Attention masks and a small attention function for Module 06.

True means "this query position may attend to this key position".
"""
import math

import torch


def full_mask(n_queries, n_keys=None):
    """Encoder-style mask: every query sees every key."""
    return torch.ones(n_queries, n_keys or n_queries, dtype=torch.bool)


def causal_mask(length):
    """Decoder-style mask: query i sees keys 0..i, never the future."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def cross_mask(n_decoder, n_encoder):
    """Cross-attention: every decoder query sees every encoder key."""
    return torch.ones(n_decoder, n_encoder, dtype=torch.bool)


def attention(q, k, v, mask=None):
    """softmax(q k^T / sqrt(d)) v with an optional boolean mask. Returns (output, weights)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class SelfAttentionLayer(torch.nn.Module):
    """One single-head self-attention layer with a residual connection."""

    def __init__(self, d_model):
        super().__init__()
        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)
        self.o = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        out, _ = attention(self.q(x), self.k(x), self.v(x), mask)
        return x + self.o(out)
