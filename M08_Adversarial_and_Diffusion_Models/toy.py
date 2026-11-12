"""Toy two-dimensional data for studying GANs and diffusion models, where every mode can be counted."""
import math
import numpy as np
import torch


def ring_of_gaussians(n, n_modes=8, radius=2.0, std=0.05, seed=0):
    """n points from a mixture of n_modes Gaussians evenly spaced on a circle. Returns (points, mode index)."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, n_modes, (n,), generator=g)
    ang = 2 * math.pi * k.float() / n_modes
    centers = radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return centers + std * torch.randn(n, 2, generator=g), k


def mode_centers(n_modes=8, radius=2.0):
    ang = 2 * math.pi * torch.arange(n_modes).float() / n_modes
    return radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)


def mode_coverage(samples, n_modes=8, radius=2.0, std=0.05, tol=6.0):
    """Return (modes covered, share of samples within tol*std of a mode centre).
    A mode counts as covered if at least 1 percent of the samples land near it."""
    c = mode_centers(n_modes, radius)
    d = torch.cdist(samples, c)
    near = d.min(1)
    ok = near.values < tol * std
    counts = torch.bincount(near.indices[ok], minlength=n_modes)
    return int((counts >= 0.01 * len(samples)).sum()), float(ok.float().mean())


def grid_centers(side=5, spacing=1.0):
    r = (torch.arange(side).float() - (side - 1) / 2) * spacing
    return torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2)


def grid_of_gaussians(n, side=5, spacing=1.0, std=0.05, seed=0):
    """n points from a side-by-side grid of Gaussians (25 modes for side=5). Returns (points, mode index)."""
    g = torch.Generator().manual_seed(seed)
    c = grid_centers(side, spacing)
    k = torch.randint(0, len(c), (n,), generator=g)
    return c[k] + std * torch.randn(n, 2, generator=g), k


def grid_coverage(samples, side=5, spacing=1.0, std=0.05, tol=6.0):
    """(modes covered, share of samples within tol*std of a mode centre); a mode counts if it holds at least 0.4 percent of the samples."""
    c = grid_centers(side, spacing)
    near = torch.cdist(samples, c).min(1)
    ok = near.values < tol * std
    counts = torch.bincount(near.indices[ok], minlength=len(c))
    return int((counts >= 0.2 / len(c) * len(samples)).sum()), float(ok.float().mean())
