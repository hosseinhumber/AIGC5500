"""The digits adaptation task used in Parts 4 and 5 of Module 06.

A small MLP is pretrained on 8 by 8 digit images, then adapted to images that
have been changed in one of three ways. Only 100 changed images are used for
adaptation; accuracy is measured on the other changed images.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SHIFTS = {
    "mild": lambda x: 0.6 * x + 0.2,
    "roll": lambda x: np.roll(x.reshape(-1, 8, 8), 1, axis=2).reshape(-1, 64),
    "invert": lambda x: 1.0 - x,
}


def load_data():
    """1,257 pretraining images and 540 held-out images (fixed split)."""
    d = load_digits()
    X, y = d.data / 16.0, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return t(Xtr), torch.tensor(ytr), t(Xte), torch.tensor(yte)


def make_mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def accuracy(net, X, y):
    with torch.no_grad():
        return (net(X).argmax(1) == y).float().mean().item()


def pretrain(seed, data, steps=600, batch_size=64, lr=1e-3):
    Xtr, ytr, _, _ = data
    torch.manual_seed(seed)
    net = make_mlp()
    opt = torch.optim.Adam(net.parameters(), lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(Xtr), (batch_size,), generator=g)
        loss = nn.functional.cross_entropy(net(Xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def shifted_split(data, shift, n_adapt=100):
    """Shift the held-out images; the first n_adapt are for adaptation, the rest for evaluation."""
    _, _, Xte, yte = data
    Xs = torch.tensor(SHIFTS[shift](Xte.numpy()), dtype=torch.float32)
    return Xs[:n_adapt], yte[:n_adapt], Xs[n_adapt:], yte[n_adapt:]


def adapt(net, Xa, ya, params, steps=300, lr=1e-3):
    """Full-batch Adam on the given parameters only. Returns the (modified) network."""
    opt = torch.optim.Adam(params, lr)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(net(Xa), ya)
        opt.zero_grad(); loss.backward(); opt.step()
    return net
