"""Moving around a latent space: interpolation, grids, traversals and sampling."""
import torch


@torch.no_grad()
def interpolate_latent(model, x1, x2, steps=8):
    """Decode points on the straight line between the latent means of two images (autoencoder or VAE)."""
    if hasattr(model, "mu"):
        z1, z2 = model.encode(x1[None])[0], model.encode(x2[None])[0]
        dec = lambda z: torch.sigmoid(model.decode(z))
    else:
        z1, z2 = model.encode(x1[None]), model.encode(x2[None])
        dec = lambda z: model.decode(z)
    ts = torch.linspace(0, 1, steps)[:, None]
    z = (1 - ts) * z1 + ts * z2
    return dec(z)


def interpolate_pixels(x1, x2, steps=8):
    """The baseline: a straight line between the two images in pixel space."""
    ts = torch.linspace(0, 1, steps)[:, None]
    return (1 - ts) * x1[None] + ts * x2[None]


@torch.no_grad()
def decode_grid(model, lim=2.5, n=12, vae=True):
    """Decode an n by n grid of a two-dimensional latent space. Returns an array of shape (n, n, 8, 8), top row = largest z2."""
    zs = torch.linspace(-lim, lim, n)
    zz = torch.stack(torch.meshgrid(zs, zs.flip(0), indexing="xy"), -1).reshape(-1, 2)
    out = model.decode(zz)
    out = torch.sigmoid(out) if vae else out
    return out.reshape(n, n, 8, 8)


@torch.no_grad()
def sample_prior(model, n, seed=0):
    """Draw z from the standard normal prior and decode. Intensities in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, model.d_latent if hasattr(model, "d_latent") else model.dec[0].in_features, generator=g)
    out = model.decode(z)
    return torch.sigmoid(out) if hasattr(model, "mu") else out


@torch.no_grad()
def sample_conditional(model, digit, n, seed=0):
    """Conditional VAE: draw z from the prior and decode with the one-hot label of the requested digit."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, model.d_latent, generator=g)
    c = torch.nn.functional.one_hot(torch.full((n,), digit), model.n_classes).float()
    return torch.sigmoid(model.decode(z, c))
