"""Distillation and pruning for small multilayer perceptrons."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def distill_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * cross entropy with the hard label."""
    soft = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1), reduction="batchmean") * T * T
    return alpha * soft + (1 - alpha) * F.cross_entropy(student_logits, y)


def train(model, X, y, epochs=60, lr=3e-3, batch=64, seed=0, teacher=None, T=4.0, alpha=0.7, masks=None, wd=0.0):
    """Train with cross entropy, or with the distillation loss when a teacher is given.
    `masks` (a list matching the Linear layers) are re-applied after every step so pruned weights stay zero."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if teacher is not None:
        teacher.eval()
        with torch.no_grad():
            tl = teacher(X)
    lin = [m for m in model if isinstance(m, nn.Linear)]
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            out = model(X[b])
            loss = distill_loss(out, tl[b], y[b], T, alpha) if teacher is not None else F.cross_entropy(out, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if masks is not None:
                with torch.no_grad():
                    for m, mk in zip(lin, masks): m.weight.mul_(mk)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def magnitude_masks(model, sparsity, scope="global"):
    """Masks that zero the smallest-magnitude weights. Biases are never pruned."""
    lin = [m for m in model if isinstance(m, nn.Linear)]
    if scope == "global":
        allw = torch.cat([m.weight.detach().abs().flatten() for m in lin])
        k = int(sparsity * allw.numel())
        thr = allw.kthvalue(k).values if k > 0 else allw.min() - 1
        return [(m.weight.detach().abs() > thr).float() for m in lin]
    masks = []
    for m in lin:
        w = m.weight.detach().abs().flatten(); k = int(sparsity * w.numel())
        thr = w.kthvalue(k).values if k > 0 else w.min() - 1
        masks.append((m.weight.detach().abs() > thr).float())
    return masks


def apply_masks(model, masks):
    lin = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for m, mk in zip(lin, masks): m.weight.mul_(mk)


def structured_prune(model, keep):
    """Remove whole hidden neurons (rows of one layer and the matching columns of the next) with the
    smallest L2 norm of incoming weights. Returns a smaller, still dense, network."""
    lin = [m for m in model if isinstance(m, nn.Linear)]
    keep_idx = []
    for m in lin[:-1]:
        k = max(1, int(round(keep * m.out_features)))
        keep_idx.append(m.weight.detach().norm(dim=1).topk(k).indices.sort().values)
    sizes = [lin[0].in_features] + [len(i) for i in keep_idx] + [lin[-1].out_features]
    new = mlp(sizes)
    nl = [m for m in new if isinstance(m, nn.Linear)]
    prev = torch.arange(lin[0].in_features)
    for j, (m, n_) in enumerate(zip(lin, nl)):
        rows = keep_idx[j] if j < len(keep_idx) else torch.arange(m.out_features)
        with torch.no_grad():
            n_.weight.copy_(m.weight[rows][:, prev]); n_.bias.copy_(m.bias[rows])
        prev = rows
    return new


# ---------------------------------------------------------------- experiments used by notebook 1 and the learning content
def timeit(fn, repeats=200, warmup=20):
    """Median wall-clock seconds of fn() over `repeats` calls, after a warm-up."""
    import time
    for _ in range(warmup): fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def exp_distillation(X, y, Xte, yte, seeds=range(5), labelled_fraction=0.05, student_hidden=32, T=4.0):
    """Student of `student_hidden` units, trained four ways, with a 512-512 teacher. Returns per-seed accuracies."""
    out = {k: [] for k in ("teacher_full", "teacher_small", "student_full_hard", "student_full_distilled",
                           "student_small_hard", "student_small_distilled", "student_small_transfer")}
    for seed in seeds:
        torch.manual_seed(seed)
        # full labelled data
        tf = train(mlp([64, 512, 512, 10]), X, y, epochs=60, seed=seed)
        sh = train(mlp([64, student_hidden, 10]), X, y, epochs=60, seed=seed)
        sd = train(mlp([64, student_hidden, 10]), X, y, epochs=60, seed=seed, teacher=tf, T=T, alpha=0.7)
        out["teacher_full"].append(accuracy(tf, Xte, yte)); out["student_full_hard"].append(accuracy(sh, Xte, yte))
        out["student_full_distilled"].append(accuracy(sd, Xte, yte))
        # a small labelled set, the rest unlabelled
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
        n = int(len(X) * labelled_fraction); lab, unl = idx[:n], idx[n:]
        ts = train(mlp([64, 512, 512, 10]), X[lab], y[lab], epochs=60, seed=seed, wd=1e-4)
        s1 = train(mlp([64, student_hidden, 10]), X[lab], y[lab], epochs=100, seed=seed)
        s2 = train(mlp([64, student_hidden, 10]), X[lab], y[lab], epochs=100, seed=seed, teacher=ts, T=T, alpha=0.7)
        Xa = torch.cat([X[lab], X[unl]]); ya = torch.cat([y[lab], torch.zeros(len(unl), dtype=torch.long)])  # y of unlabelled is never used (alpha = 1)
        s3 = train(mlp([64, student_hidden, 10]), Xa, ya, epochs=100, seed=seed, teacher=ts, T=T, alpha=1.0)
        out["teacher_small"].append(accuracy(ts, Xte, yte)); out["student_small_hard"].append(accuracy(s1, Xte, yte))
        out["student_small_distilled"].append(accuracy(s2, Xte, yte)); out["student_small_transfer"].append(accuracy(s3, Xte, yte))
    return out


def exp_temperature(X, y, Xte, yte, temps=(1, 2, 4, 8), seeds=range(5), labelled_fraction=0.05, student_hidden=32):
    """Accuracy of a student distilled on an unlabelled transfer set at several temperatures."""
    res = {t: [] for t in temps}
    for seed in seeds:
        torch.manual_seed(seed)
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
        n = int(len(X) * labelled_fraction); lab, unl = idx[:n], idx[n:]
        ts = train(mlp([64, 512, 512, 10]), X[lab], y[lab], epochs=60, seed=seed, wd=1e-4)
        Xa = torch.cat([X[lab], X[unl]]); ya = torch.zeros(len(Xa), dtype=torch.long)
        for t in temps:
            s = train(mlp([64, student_hidden, 10]), Xa, ya, epochs=100, seed=seed, teacher=ts, T=float(t), alpha=1.0)
            res[t].append(accuracy(s, Xte, yte))
    return res


def exp_pruning(X, y, Xte, yte, sparsities=(0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99), seeds=range(3), finetune_epochs=20):
    """Global magnitude pruning of a 512-512 network, with and without fine-tuning."""
    out = {s: {"no_ft": [], "ft": [], "nonzero": []} for s in sparsities}
    for seed in seeds:
        torch.manual_seed(seed)
        base = train(mlp([64, 512, 512, 10]), X, y, epochs=60, seed=seed)
        for s in sparsities:
            m = copy.deepcopy(base); masks = magnitude_masks(m, s); apply_masks(m, masks)
            out[s]["no_ft"].append(accuracy(m, Xte, yte))
            train(m, X, y, epochs=finetune_epochs, lr=1e-3, seed=seed, masks=masks)
            out[s]["ft"].append(accuracy(m, Xte, yte)); out[s]["nonzero"].append(count_nonzero(m))
    return out


def exp_structured(X, y, Xte, yte, keeps=(1.0, 0.5, 0.25, 0.125, 0.0625), seeds=range(3), finetune_epochs=20):
    """Remove whole hidden neurons, fine-tune, and report accuracy and parameter count."""
    out = {k: {"no_ft": [], "ft": [], "params": None} for k in keeps}
    for seed in seeds:
        torch.manual_seed(seed)
        base = train(mlp([64, 512, 512, 10]), X, y, epochs=60, seed=seed)
        for k in keeps:
            m = structured_prune(base, k)
            out[k]["no_ft"].append(accuracy(m, Xte, yte)); out[k]["params"] = count_params(m)
            train(m, X, y, epochs=finetune_epochs, lr=1e-3, seed=seed)
            out[k]["ft"].append(accuracy(m, Xte, yte))
    return out


def to_sparse_mlp(model):
    """Forward function that uses sparse (CSR) weight matrices, to test whether unstructured sparsity buys speed."""
    lin = [m for m in model if isinstance(m, nn.Linear)]
    sp = [m.weight.detach().to_sparse_csr() for m in lin]
    bs = [m.bias.detach() for m in lin]
    def f(x):
        for i, (w, b) in enumerate(zip(sp, bs)):
            x = torch.sparse.mm(w, x.T).T + b
            if i < len(sp) - 1: x = torch.relu(x)
        return x
    return f


def exp_latency(X, y, batch=256, seed=0):
    """Median forward time for one batch of `batch` images, one CPU thread."""
    torch.manual_seed(seed)
    base = train(mlp([64, 512, 512, 10]), X, y, epochs=30, seed=seed)
    xb = X[:batch]
    rows = []
    def bench(name, fn, params, nonzero, acc_model=None):
        with torch.no_grad(): t = timeit(lambda: fn(xb))
        rows.append(dict(name=name, ms=t * 1e3, params=params, nonzero=nonzero))
    with torch.no_grad():
        bench("dense 512-512", base, count_params(base), count_nonzero(base))
        for s in (0.9, 0.98):
            m = copy.deepcopy(base); apply_masks(m, magnitude_masks(m, s))
            bench(f"{int(s * 100)}% unstructured, dense matmul", m, count_params(m), count_nonzero(m))
            bench(f"{int(s * 100)}% unstructured, sparse matmul", to_sparse_mlp(m), count_params(m), count_nonzero(m))
        for k in (0.25, 0.0625):
            m = structured_prune(base, k)
            bench(f"structured, keep {k:g} of the neurons", m, count_params(m), count_nonzero(m))
    return rows
