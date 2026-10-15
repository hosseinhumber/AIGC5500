"""Vision Transformer pieces for Module 05: patch embedding and a tiny ViT."""
import torch
import torch.nn as nn

from transformer import TransformerBlock


def patchify(images, patch):
    """Cut a batch of images (B, C, H, W) into non-overlapping square patches and flatten each one.
    Returns (B, n_patches, C * patch * patch). Patches are read left to right, top to bottom."""
    b, c, h, w = images.shape
    assert h % patch == 0 and w % patch == 0, "image size must be divisible by the patch size"
    x = images.unfold(2, patch, patch).unfold(3, patch, patch)       # (B, C, H/p, W/p, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, (h // patch) * (w // patch), c * patch * patch)
    return x


class PatchEmbedding(nn.Module):
    """Flatten each patch and map it to d_model with one shared linear layer."""

    def __init__(self, patch, in_channels, d_model):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_channels * patch * patch, d_model)

    def forward(self, images):
        return self.proj(patchify(images, self.patch))


class TinyViT(nn.Module):
    """Patch embedding, a learned class token, learned position embeddings, encoder blocks, linear head on the class token."""

    def __init__(self, img_size=8, patch=2, in_channels=1, n_classes=10, d_model=32,
                 n_heads=4, d_ff=64, n_layers=2, norm="pre"):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.embed = PatchEmbedding(patch, in_channels, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, norm=norm) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, images):
        x = self.embed(images)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])
