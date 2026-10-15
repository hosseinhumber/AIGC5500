"""Transformer blocks for Module 05: encoder block, decoder block, and a small classifier."""
import torch
import torch.nn as nn

from attention import MultiHeadAttention, causal_mask
from positional import LearnedPositions, sinusoidal_encoding


class FeedForward(nn.Module):
    """The position-wise network: two linear layers with a nonlinearity between them,
    applied to every position independently."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One encoder block: self-attention, then a feed-forward network, each wrapped in a residual path.

    norm='pre'  : x = x + sublayer(LayerNorm(x))   (normalize on the branch, residual path stays clean)
    norm='post' : x = LayerNorm(x + sublayer(x))   (normalize after the addition, as in the original design)
    causal=True blocks attention to future positions, which turns the block into a decoder-style block.
    """

    def __init__(self, d_model, n_heads, d_ff, norm="pre", causal=False):
        super().__init__()
        assert norm in ("pre", "post")
        self.norm_type, self.causal = norm, causal
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, head_mask=None, return_weights=False):
        mask = causal_mask(x.shape[1]).to(x.device) if self.causal else None
        weights = None
        if self.norm_type == "pre":
            a = self.attn(self.ln1(x), mask=mask, head_mask=head_mask, return_weights=return_weights)
            a, weights = a if return_weights else (a, None)
            x = x + a
            x = x + self.ff(self.ln2(x))
        else:
            a = self.attn(x, mask=mask, head_mask=head_mask, return_weights=return_weights)
            a, weights = a if return_weights else (a, None)
            x = self.ln1(x + a)
            x = self.ln2(x + self.ff(x))
        return (x, weights) if return_weights else x


class DecoderBlock(nn.Module):
    """A decoder block: masked self-attention, cross-attention to the encoder output, then feed-forward.
    Pre-norm layout. The cross-attention takes its queries from the decoder and its keys and values from the encoder."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, memory):
        x = x + self.self_attn(self.ln1(x), mask=causal_mask(x.shape[1]).to(x.device))
        x = x + self.cross_attn(self.ln2(x), context=memory)
        return x + self.ff(self.ln3(x))


class SequenceClassifier(nn.Module):
    """Token embedding, optional positional encoding, a stack of encoder blocks, mean pooling, a linear head.

    pos: 'none', 'learned' or 'sinusoidal'. For 'learned', max_len fixes how many positions exist.
    inputs: integer token ids of shape (batch, T).
    """

    def __init__(self, vocab, n_classes, d_model=32, n_heads=4, d_ff=64, n_layers=2,
                 pos="sinusoidal", max_len=64, norm="pre"):
        super().__init__()
        self.pos_kind, self.d_model = pos, d_model
        self.embed = nn.Embedding(vocab, d_model)
        if pos == "learned":
            self.pos = LearnedPositions(max_len, d_model)
        elif pos == "sinusoidal":
            self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, norm=norm) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model) if norm == "pre" else nn.Identity()
        self.head = nn.Linear(d_model, n_classes)

    def encode(self, tokens, head_mask=None):
        x = self.embed(tokens)
        length = tokens.shape[1]
        if self.pos_kind == "learned":
            x = x + self.pos(length)
        elif self.pos_kind == "sinusoidal":
            x = x + self.pe[:length]
        for block in self.blocks:
            x = block(x, head_mask=head_mask)
        return self.final_norm(x)

    def forward(self, tokens, head_mask=None):
        return self.head(self.encode(tokens, head_mask).mean(dim=1))
