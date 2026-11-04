"""Checks that the Module 06 modules behave as the learning content says.
Run from this folder:  python test_modules.py"""
import torch
import torch.nn as nn

from budget import GB, compute_flops, full_finetune_bytes, lora_bytes, lora_params, training_time_seconds
from lora import LoRALinear, add_lora, count_trainable
from masks import SelfAttentionLayer, attention, causal_mask, cross_mask, full_mask
from quant import absmax_quantize, normal_float_levels, normal_float_quantize, relative_error
from tasks import SHIFTS, make_mlp
from tokenizer import encode_word, train_bpe

torch.manual_seed(0)

# 1. Byte-pair encoding reproduces the merge table and builds unseen words from learned pieces.
corpus = {"low": 5, "lower": 2, "lowest": 2, "newer": 6, "newest": 6, "wider": 3, "widest": 3, "wide": 4}
merges = train_bpe(corpus, 10)
assert [a + b for (a, b), _ in merges] == ["we", "ne", "newe", "t</w>", "st</w>", "r</w>", "wi", "wid", "wide", "lo"]
assert encode_word("slowest", merges) == ["s", "lo", "we", "st</w>"]

# 2. Masks: shapes, counts, and the causal layer ignores the future.
assert full_mask(6).sum() == 36 and causal_mask(6).sum() == 21 and cross_mask(4, 6).shape == (4, 6)
layer, x = SelfAttentionLayer(16), torch.randn(1, 8, 16)
x2 = x.clone(); x2[:, 5] = torch.randn(16)
with torch.no_grad():
    assert torch.allclose(layer(x, causal_mask(8))[:, :5], layer(x2, causal_mask(8))[:, :5], atol=1e-6)
    assert not torch.allclose(layer(x)[:, :5], layer(x2)[:, :5], atol=1e-3)
_, w = attention(torch.randn(2, 4, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8), cross_mask(4, 6))
assert torch.allclose(w.sum(-1), torch.ones(2, 4))

# 3. Budget arithmetic.
assert compute_flops(7e9, 140e9) == 6 * 7e9 * 140e9
assert abs(training_time_seconds(125e6, 2.5e9) / 3600 - 5.2) < 0.1
n = lora_params(32, 4096, 16)
assert n == 16_777_216
assert [round(b / GB, 1) for b in (full_finetune_bytes(7e9), lora_bytes(7e9, n), lora_bytes(7e9, n, 0.5, block=64))] == [112.0, 14.3, 4.2]

# 4. LoRA starts equal to the wrapped layer, merges exactly, and counts only A and B as trainable.
base = nn.Linear(32, 24)
lora = LoRALinear(base, r=4, alpha=8)
xx = torch.randn(5, 32)
assert torch.allclose(lora(xx), base(xx))
with torch.no_grad():
    lora.B.normal_()
assert torch.allclose(lora(xx), lora.merged()(xx), atol=1e-5)
assert count_trainable(lora) == 4 * 32 + 24 * 4
assert count_trainable(add_lora(make_mlp(), 1)) == 832 and count_trainable(add_lora(make_mlp(), 16)) == 13312
assert count_trainable(make_mlp()) == 85002

# 5. Quantization: the worked example, block isolation of an outlier, and level counts.
codes, scale, deq = absmax_quantize(torch.tensor([0.12, -0.50, 0.03, 0.31]), 8)
assert codes.flatten().tolist() == [30, -127, 8, 79]
codes4, _, deq4 = absmax_quantize(torch.tensor([0.12, -0.50, 0.03, 0.31]), 4)
assert codes4.flatten().tolist() == [2, -7, 0, 4] and deq4[2] == 0
w = torch.randn(256) * 0.1
w_out = w.clone(); w_out[10] = 20 * w.abs().max()
shared, blocks = absmax_quantize(w_out, 4)[2], absmax_quantize(w_out, 4, block=64)[2]
assert relative_error(w_out, blocks) < relative_error(w_out, shared)
assert len(normal_float_levels()) == 16 and normal_float_quantize(w, 64)[2].shape == w.shape

# 6. The shifts change the images as described.
img = torch.rand(3, 64).numpy()
assert abs(SHIFTS["invert"](img) + img - 1).max() < 1e-6
assert abs(SHIFTS["mild"](img) - (0.6 * img + 0.2)).max() < 1e-6
assert abs(SHIFTS["roll"](img).reshape(3, 8, 8)[:, :, 1:] - img.reshape(3, 8, 8)[:, :, :-1]).max() < 1e-6

print("All Module 06 checks passed.")
