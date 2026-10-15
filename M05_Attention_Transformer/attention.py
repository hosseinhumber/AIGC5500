"""Attention building blocks for Module 05.

Everything here is written from scratch with plain tensor operations, so each
line can be read against the equations in the learning content.
"""
import math

import torch
import torch.nn as nn


def causal_mask(length):
    """Boolean mask of shape (length, length). True means 'this position may be attended to'.
    Row i (the query) may look at columns 0..i (the keys), never at the future."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def scaled_dot_product_attention(q, k, v, mask=None, scale=True):
    """softmax(q k^T / sqrt(d_k)) v.

    q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
    mask: boolean, broadcastable to (..., T_q, T_k); False entries are blocked.
    Returns (output, weights). Set scale=False to see why the scaling matters.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Several attention heads that run side by side on lower-dimensional projections.

    d_model is split into n_heads slices of size d_model // n_heads. Each head has
    its own query, key and value projection (stored together in one weight matrix
    each), attends on its own, and the head outputs are concatenated and mixed by
    a final linear layer.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)  # (batch, heads, T, d_head)

    def forward(self, x, context=None, mask=None, head_mask=None, return_weights=False):
        """x supplies the queries. context supplies keys and values (cross-attention);
        by default it is x itself (self-attention). head_mask is an optional tensor of
        shape (n_heads,) of 0/1 values used to switch individual heads off."""
        context = x if context is None else context
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(context))
        v = self.split_heads(self.w_v(context))
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        if head_mask is not None:
            out = out * head_mask.view(1, -1, 1, 1)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, self.d_model)  # concatenate the heads
        out = self.w_o(out)
        return (out, weights) if return_weights else out
