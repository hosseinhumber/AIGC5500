"""Paired image-to-image translation on 8 by 8 digits: given the top half of a digit, produce the bottom half."""
import torch
import torch.nn.functional as F
from gan import Generator, Discriminator


def split_halves(X):
    """Rows 0 to 3 are the input (32 pixels); rows 4 to 7 are the target (32 pixels)."""
    return X[:, :32], X[:, 32:]


def join_halves(top, bottom):
    return torch.cat([top, bottom], 1)


def train_translator(X, adversarial_weight=0.0, l1_weight=10.0, steps=3000, batch_size=128, lr=1e-3, seed=0, loss="l1"):
    """Train a generator G(top, noise) -> bottom with an L1 (or, if loss='l2', squared error) loss, plus (optionally) a conditional discriminator that sees (top, bottom).
    adversarial_weight = 0 is plain regression. Returns the generator."""
    torch.manual_seed(seed)
    top, bottom = split_halves(X)
    G = Generator(d_noise=8, d_out=32, d_hidden=128, n_classes=32, out_act="sigmoid")
    D = Discriminator(d_in=32, d_hidden=128, n_classes=32)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch_size,))
        t, b = top[idx], bottom[idx]
        fake = G(torch.randn(batch_size, G.d_noise), t)
        if adversarial_weight > 0:
            dr, df = D(b, t), D(fake.detach(), t)
            d_loss = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr)) + \
                     F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        err = (fake - b).abs().mean() if loss == "l1" else ((fake - b) ** 2).mean()
        g_loss = l1_weight * err
        if adversarial_weight > 0:
            dg = D(fake, t)
            g_loss = g_loss + adversarial_weight * F.binary_cross_entropy_with_logits(dg, torch.ones_like(dg))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    return G


@torch.no_grad()
def translate(G, top, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G(torch.randn(len(top), G.d_noise, generator=g), top)
