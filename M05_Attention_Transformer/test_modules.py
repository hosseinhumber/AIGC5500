"""Checks that the Module 05 modules behave as the learning content says.
Run from this folder:  python test_modules.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from positional import sinusoidal_encoding
from transformer import DecoderBlock, SequenceClassifier, TransformerBlock
from vit import PatchEmbedding, TinyViT, patchify

torch.manual_seed(0)

# 1. Scaled dot-product attention equals PyTorch's, with and without the causal mask.
q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 8)
assert torch.allclose(scaled_dot_product_attention(q, k, v)[0], F.scaled_dot_product_attention(q, k, v), atol=1e-5)
s = torch.randn(2, 4, 5, 8)
assert torch.allclose(scaled_dot_product_attention(s, s, s, mask=causal_mask(5))[0],
                      F.scaled_dot_product_attention(s, s, s, is_causal=True), atol=1e-5)

# 2. Multi-head attention equals nn.MultiheadAttention once the stacked weights are copied.
d = 32
mha, ref = MultiHeadAttention(d, 4), nn.MultiheadAttention(d, 4, batch_first=True)
with torch.no_grad():
    W, b = ref.in_proj_weight, ref.in_proj_bias
    for lin, i in ((mha.w_q, 0), (mha.w_k, 1), (mha.w_v, 2)):
        lin.weight.copy_(W[i * d:(i + 1) * d]); lin.bias.copy_(b[i * d:(i + 1) * d])
    mha.w_o.weight.copy_(ref.out_proj.weight); mha.w_o.bias.copy_(ref.out_proj.bias)
x = torch.randn(3, 6, d)
assert torch.allclose(mha(x), ref(x, x, x)[0], atol=1e-5)

# 3. Attention is permutation-equivariant.
perm = torch.randperm(6)
assert torch.allclose(mha(x)[:, perm], mha(x[:, perm]), atol=1e-5)

# 4. The causal block ignores the future.
block = TransformerBlock(d, 4, 64, causal=True)
x2 = x.clone(); x2[:, 4:] = torch.randn(3, 2, d)
assert torch.allclose(block(x)[:, :4], block(x2)[:, :4], atol=1e-6)

# 5. A fixed matrix maps sinusoidal PE(p) to PE(p + k) for every p.
pe = sinusoidal_encoding(200, 32).double()
M = torch.linalg.lstsq(pe[:150], pe[5:155]).solution
assert (pe[:150] @ M - pe[5:155]).abs().max() < 1e-4

# 6. Patch embedding equals a convolution with kernel size equal to stride.
imgs = torch.randn(2, 3, 8, 8)
emb, conv = PatchEmbedding(4, 3, 16), nn.Conv2d(3, 16, 4, 4)
with torch.no_grad():
    conv.weight.copy_(emb.proj.weight.view(16, 3, 4, 4)); conv.bias.copy_(emb.proj.bias)
assert torch.allclose(emb(imgs), conv(imgs).flatten(2).transpose(1, 2), atol=1e-5)

# 7. Shapes of the larger pieces.
assert DecoderBlock(d, 4, 64)(torch.randn(2, 5, d), torch.randn(2, 9, d)).shape == (2, 5, d)
assert SequenceClassifier(10, 2)(torch.randint(0, 10, (3, 16))).shape == (3, 2)
assert TinyViT(8, 2)(torch.randn(2, 1, 8, 8)).shape == (2, 10)
assert patchify(torch.arange(16.0).view(1, 1, 4, 4), 2).shape == (1, 4, 4)

print("All Module 05 module checks passed.")
