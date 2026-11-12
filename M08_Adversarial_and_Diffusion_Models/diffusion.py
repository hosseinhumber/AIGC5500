"""Denoising diffusion (DDPM) on vectors: noise schedule, the forward process in closed form, a noise-prediction network, sampling."""
import math
import torch
import torch.nn as nn


class Schedule:
    """Linear beta schedule. alpha_bar[t] is the share of the original signal left at step t."""

    def __init__(self, T=100, beta_min=1e-4, beta_max=0.1):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)


def q_sample(sched, x0, t, noise=None):
    """Jump straight to step t of the forward process: x_t = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps."""
    noise = torch.randn_like(x0) if noise is None else noise
    ab = sched.alpha_bar[t][:, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise


class Denoiser(nn.Module):
    """Predicts the noise that was added, given the noisy input, the step, and an optional one-hot label."""

    def __init__(self, d_data=2, d_hidden=128, T=100, n_classes=0):
        super().__init__()
        self.T, self.n_classes = T, n_classes
        self.net = nn.Sequential(nn.Linear(d_data + 16 + n_classes, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_data))

    @staticmethod
    def time_features(t, T):
        freqs = torch.exp(torch.linspace(0, math.log(100.0), 8))
        a = (t.float()[:, None] / T) * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t, c=None):
        h = [x, self.time_features(t, self.T)] + ([] if c is None else [c])
        return self.net(torch.cat(h, 1))


def train_diffusion(model, sched, X, y_onehot=None, steps=3000, batch_size=128, lr=2e-3, seed=0):
    """Minimize the mean squared error between the true noise and the predicted noise at a random step."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch_size,))
        x0 = X[idx]
        t = torch.randint(0, sched.T, (batch_size,))
        xt, eps = q_sample(sched, x0, t)
        c = None if y_onehot is None else y_onehot[idx]
        loss = ((model(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def sample_ddpm(model, sched, n, d_data, seed=0, labels=None, return_path=False):
    """Ancestral sampling: start from pure noise and apply the learned reverse step T times."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_data, generator=g)
    c = None if labels is None else torch.nn.functional.one_hot(labels, model.n_classes).float()
    path = [x.clone()]
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t)
        eps = model(x, tt, c)
        a, ab, b = sched.alphas[t], sched.alpha_bar[t], sched.betas[t]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        x = mean + (b.sqrt() * torch.randn(x.shape, generator=g) if t > 0 else 0)
        if return_path: path.append(x.clone())
    return (x, path) if return_path else x


@torch.no_grad()
def sample_ddim(model, sched, n, d_data, n_steps=10, seed=0, labels=None):
    """Deterministic sampler that skips steps: the same trained network, n_steps evaluations instead of T."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_data, generator=g)
    c = None if labels is None else torch.nn.functional.one_hot(labels, model.n_classes).float()
    ts = torch.linspace(sched.T - 1, 0, n_steps).round().long()
    for i, t in enumerate(ts):
        ab = sched.alpha_bar[t]
        ab_next = sched.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        eps = model(x, torch.full((n,), int(t)), c)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x
