"""Plain autoencoders: a bottleneck network trained to rebuild its own input."""
import numpy as np
import torch
import torch.nn as nn


class AE(nn.Module):
    """Encoder, bottleneck of size d_latent, decoder. With linear=True there are no hidden layers or activations."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=2, linear=False):
        super().__init__()
        if linear:
            self.enc = nn.Linear(d_in, d_latent)
            self.dec = nn.Linear(d_latent, d_in)
        else:
            self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_latent))
            self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def train_ae(model, X, epochs=300, lr=2e-3, batch_size=64, noise=0.0, seed=0):
    """Train with mean squared error. noise > 0 corrupts the input but still rebuilds the clean image (a denoising autoencoder).
    Returns the mean loss of each epoch."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        total = 0.0
        for i in range(0, len(X), batch_size):
            xb = X[perm[i:i + batch_size]]
            xin = xb + noise * torch.randn(xb.shape, generator=g) if noise > 0 else xb
            loss = ((model(xin) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        hist.append(total / len(X))
    return hist


@torch.no_grad()
def reconstruction_mse(model, X):
    return ((model(X) - X) ** 2).mean().item()


def pca_mse(X_train, X_eval, k):
    """Mean squared error of rebuilding X_eval from the top k principal components of X_train."""
    X_train = X_train.numpy().astype(np.float64)
    X_eval = X_eval.numpy().astype(np.float64)
    mu = X_train.mean(0)
    _, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
    P = Vt[:k]
    rec = (X_eval - mu) @ P.T @ P + mu
    return float(((rec - X_eval) ** 2).mean())
