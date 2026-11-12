"""Generative adversarial networks: generator, discriminator, three training objectives, a conditional version."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_out, d_hidden=64, layers=2, act=nn.ReLU, spectral=False):
    mods, d = [], d_in
    for _ in range(layers):
        lin = nn.Linear(d, d_hidden)
        mods += [nn.utils.spectral_norm(lin) if spectral else lin, act()]
        d = d_hidden
    last = nn.Linear(d, d_out)
    mods.append(nn.utils.spectral_norm(last) if spectral else last)
    return nn.Sequential(*mods)


class Generator(nn.Module):
    """Maps noise (and an optional one-hot label) to a sample. out_act='sigmoid' for images in [0, 1], None for raw coordinates."""

    def __init__(self, d_noise=8, d_out=2, d_hidden=64, n_classes=0, out_act=None):
        super().__init__()
        self.d_noise, self.n_classes = d_noise, n_classes
        self.net = mlp(d_noise + n_classes, d_out, d_hidden)
        self.out_act = out_act

    def forward(self, z, c=None):
        out = self.net(z if c is None else torch.cat([z, c], 1))
        return torch.sigmoid(out) if self.out_act == "sigmoid" else out


class Discriminator(nn.Module):
    """Returns a logit (a real-valued score for the Wasserstein critic). With n_classes > 0 it also sees the label."""

    def __init__(self, d_in=2, d_hidden=64, n_classes=0, spectral=False):
        super().__init__()
        self.net = mlp(d_in + n_classes, 1, d_hidden, act=lambda: nn.LeakyReLU(0.2), spectral=spectral)

    def forward(self, x, c=None):
        return self.net(x if c is None else torch.cat([x, c], 1)).squeeze(1)


def gradient_penalty(D, real, fake, c=None):
    """WGAN-GP: penalize the critic's gradient norm, measured on random points between real and fake, for deviating from 1."""
    t = torch.rand(len(real), 1)
    x = (t * real + (1 - t) * fake).requires_grad_(True)
    g, = torch.autograd.grad(D(x, c).sum(), x, create_graph=True)
    return ((g.norm(dim=1) - 1) ** 2).mean(), g.norm(dim=1).mean().item()


def train_gan(G, D, X, y_onehot=None, objective="nonsat", steps=2000, batch_size=128, lr_g=2e-3, lr_d=2e-3,
              n_critic=1, gp_weight=10.0, seed=0, betas=(0.5, 0.9), log_every=0):
    """Alternate discriminator and generator updates.
    objective: 'minimax'  generator minimizes log(1 - D(G(z)))   (the original, saturating loss)
               'nonsat'   generator maximizes log D(G(z))        (the standard fix)
               'wgan_gp'  critic maximizes D(real) - D(fake) with a gradient penalty, generator maximizes D(G(z))
    Returns a dict of histories: d_loss, g_loss, d_real (mean probability or score on real), d_fake, g_grad (norm of the loss gradient at the generator output)."""
    torch.manual_seed(seed)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    h = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": [], "g_grad": [], "gp_norm": []}
    n = len(X)
    for step in range(steps):
        for _ in range(n_critic):
            idx = torch.randint(0, n, (batch_size,))
            real = X[idx]
            c = None if y_onehot is None else y_onehot[idx]
            with torch.no_grad():
                fake = G(torch.randn(batch_size, G.d_noise), c)
            dr, df = D(real, c), D(fake, c)
            if objective == "wgan_gp":
                gp, gpn = gradient_penalty(D, real, fake, c)
                d_loss = df.mean() - dr.mean() + gp_weight * gp
                h["gp_norm"].append(gpn)
            else:
                d_loss = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr)) + \
                         F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        idx = torch.randint(0, n, (batch_size,))
        c = None if y_onehot is None else y_onehot[idx]
        fake = G(torch.randn(batch_size, G.d_noise), c)
        fake.retain_grad()
        df = D(fake, c)
        if objective == "minimax":
            g_loss = -F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))   # = log(1 - D(G(z)))
        elif objective == "nonsat":
            g_loss = F.binary_cross_entropy_with_logits(df, torch.ones_like(df))     # = -log D(G(z))
        else:
            g_loss = -df.mean()
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        h["d_loss"].append(float(d_loss.detach())); h["g_loss"].append(float(g_loss.detach()))
        h["g_grad"].append(float(fake.grad.norm(dim=1).mean()))
        if objective == "wgan_gp":
            h["d_real"].append(float(dr.mean().detach())); h["d_fake"].append(float(df.mean().detach()))
        else:
            h["d_real"].append(float(torch.sigmoid(dr).mean().detach())); h["d_fake"].append(float(torch.sigmoid(df).mean().detach()))
    return h


@torch.no_grad()
def sample_gan(G, n, seed=0, label=None):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, G.d_noise, generator=g)
    c = None if G.n_classes == 0 else torch.nn.functional.one_hot(torch.full((n,), label), G.n_classes).float()
    return G(z, c)
