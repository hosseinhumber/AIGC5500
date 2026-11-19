"""Interpretability, calibration and robustness tools (Part 4)."""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress import train as _train


# ---------------------------------------------------------------- a small CNN for 8 x 8 digits
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 4, 10)

    def features(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.c1(x))
        return F.relu(self.c2(x))                 # (B, 16, 8, 8): the last convolutional maps

    def head(self, f):
        return self.fc(F.adaptive_avg_pool2d(f, 2).flatten(1))   # pool each map to 2 x 2, then a linear layer

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(X, y, epochs=30, seed=0, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    m = SmallCNN()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(m(X[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


# ---------------------------------------------------------------- attribution for images
def grad_cam(model, x, target=None):
    """Grad-CAM heat map (8 x 8) for one image x of shape (64,). Returns (map, predicted class)."""
    model.eval()
    f = model.features(x[None]).detach().requires_grad_(True)
    logits = model.head(f)
    c = int(logits.argmax(1)) if target is None else target
    logits[0, c].backward()
    w = f.grad.mean((2, 3), keepdim=True)                       # one weight per feature map
    cam = F.relu((w * f).sum(1))[0].detach()
    cam = cam / (cam.max() + 1e-12)
    return cam.numpy(), c


def saliency(model, x, target=None):
    """Absolute gradient of the class score with respect to the pixels (8 x 8)."""
    model.eval()
    x = x[None].clone().requires_grad_(True)
    out = model(x)
    c = int(out.argmax(1)) if target is None else target
    out[0, c].backward()
    s = x.grad.abs().view(8, 8)
    return (s / (s.max() + 1e-12)).numpy()


def occlusion_map(model, x, target=None, size=2, fill=0.0):
    """Drop in the class probability when each size x size patch is replaced by `fill` (8 x 8 map)."""
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(x[None]), 1)[0]
        c = int(p.argmax()) if target is None else target
        img = x.view(8, 8)
        out = np.zeros((8, 8)); cnt = np.zeros((8, 8))
        for i in range(0, 8 - size + 1):
            for j in range(0, 8 - size + 1):
                z = img.clone(); z[i:i + size, j:j + size] = fill
                drop = (p[c] - F.softmax(model(z.view(1, -1)), 1)[0, c]).item()
                out[i:i + size, j:j + size] += drop; cnt[i:i + size, j:j + size] += 1
    return out / cnt


def rank_corr(a, b):
    ra = np.argsort(np.argsort(np.asarray(a).ravel())); rb = np.argsort(np.argsort(np.asarray(b).ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def sanity_randomization(model, X, n=30, seed=0):
    """Mean rank correlation between saliency of the trained model and of the same model with random weights.
    A trustworthy explanation method should change a lot, so this should be near zero."""
    torch.manual_seed(seed)
    rnd = SmallCNN().eval()
    cs = []
    for i in range(n):
        c = int(model(X[i][None]).argmax(1))
        cs.append(rank_corr(saliency(model, X[i], c), saliency(rnd, X[i], c)))
    return float(np.mean(cs))


def deletion_curve(model, x, order, steps=8, fill=0.0):
    """Class probability as the most important pixels (given by `order`, a flat ranking, most important first) are removed."""
    model.eval()
    with torch.no_grad():
        c = int(model(x[None]).argmax(1)); out = []
        z = x.clone(); per = max(1, len(order) // steps)
        out.append(F.softmax(model(z[None]), 1)[0, c].item())
        for k in range(steps):
            z[order[k * per:(k + 1) * per]] = fill
            out.append(F.softmax(model(z[None]), 1)[0, c].item())
    return out


# ---------------------------------------------------------------- exact Shapley values
def exact_shapley(f, x, baseline):
    """Exact Shapley values of f(x) for a tabular model, with absent features set to `baseline`.
    f takes a 2-D numpy array and returns a 1-D array. Cost grows as 2^d, fine for d <= 12."""
    d = len(x)
    x = np.asarray(x, dtype=float); baseline = np.asarray(baseline, dtype=float)
    from math import factorial
    cache = {}
    def v(S):
        if S not in cache:
            z = baseline.copy(); idx = list(S)
            z[idx] = x[idx]; cache[S] = float(f(z[None])[0])
        return cache[S]
    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(d):
            for S in itertools.combinations(others, r):
                w = factorial(r) * factorial(d - r - 1) / factorial(d)
                phi[i] += w * (v(tuple(sorted(S + (i,)))) - v(tuple(S)))
    return phi, v(()), v(tuple(range(d)))


# ---------------------------------------------------------------- calibration
def ece(probs, y, bins=10):
    """Expected calibration error with equal-width confidence bins."""
    probs = np.asarray(probs); y = np.asarray(y)
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def reliability(probs, y, bins=10):
    """Per-bin (mean confidence, accuracy, count) for a reliability diagram."""
    probs = np.asarray(probs); y = np.asarray(y)
    conf = probs.max(1); acc = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float(conf[m].mean()) if m.any() else np.nan, float(acc[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows


def fit_temperature(logits, y):
    """Single temperature T minimizing validation cross entropy (grid then refine)."""
    logits = torch.as_tensor(logits); y = torch.as_tensor(y)
    best, bt = 1e9, 1.0
    for t in np.concatenate([np.linspace(0.3, 5, 95), np.linspace(5, 15, 21)]):
        l = F.cross_entropy(logits / float(t), y).item()
        if l < best: best, bt = l, float(t)
    return bt


def overfit_mlp(Xtr, ytr, hidden=256, epochs=300, seed=0):
    """A deliberately overconfident classifier: large, trained long on few examples."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(epochs):
        loss = F.cross_entropy(m(Xtr), ytr); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


# ---------------------------------------------------------------- corruptions and shift
def corrupt(X, kind, severity, seed=0):
    """X is (N, 64) in [0, 1]. kinds: noise, blur, occlude, brightness, shift."""
    g = torch.Generator().manual_seed(seed)
    img = X.view(-1, 8, 8).clone()
    if kind == "noise":
        img = img + severity * torch.randn(img.shape, generator=g)
    elif kind == "blur":
        k = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        for _ in range(int(severity)):
            img = F.conv2d(F.pad(img[:, None], (1, 1, 1, 1), mode="replicate"), k[None, None])[:, 0]
    elif kind == "occlude":
        s = int(severity)
        for n in range(len(img)):
            i, j = torch.randint(0, 8 - s + 1, (2,), generator=g).tolist(); img[n, i:i + s, j:j + s] = 0
    elif kind == "brightness":
        img = img * severity
    elif kind == "shift":
        img = torch.roll(img, shifts=int(severity), dims=2)
    return img.clamp(0, 1).view(-1, 64)


def auroc(pos, neg):
    """Probability that a random positive scores higher than a random negative."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def max_softmax(model, X):
    with torch.no_grad():
        return F.softmax(model(X), 1).max(1).values.numpy()
