"""Plumbing for the Final Project. These helpers do the parts that are easy to get subtly wrong and that
every group needs identically: seeds, four disjoint splits, calibration error, timing, peak memory, saving tables.
They contain no model, no training loop, no generator, no policy and no compression: those are yours.

Read the code before you use it. If you change a helper, say so in the README and say why.
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch together. Call it at the start of every run, with the run's own seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def four_way_split(y, sizes, seed=2026, train_cap=None):
    """Stratified split of example indices into 'train', 'holdout', 'val' and 'test'.

    y         : integer labels, shape (N,)
    sizes     : dict of examples per class for each split, e.g. {"train": 250, "holdout": 100, "val": 100, "test": 200}
    seed      : the SPLIT seed. Use one fixed value for the whole project, not the training seed, so that every
                model and every seed sees exactly the same splits.
    train_cap : optional dict {class: n} that limits a class in the training split only. Use it to make a class
                rare in training while it keeps its natural share in holdout, val and test.

    Returns a dict of index arrays. The splits are disjoint; this is checked.
    """
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    out = {k: [] for k in ("train", "holdout", "val", "test")}
    for c in np.unique(y):
        pool = rng.permutation(np.where(y == c)[0])
        start = 0
        for name in ("train", "holdout", "val", "test"):
            n = sizes[name]
            if name == "train" and train_cap and c in train_cap:
                n = train_cap[c]
            if start + n > len(pool):
                raise ValueError(f"class {c}: not enough examples for split '{name}' ({start + n} > {len(pool)})")
            out[name] += pool[start:start + n].tolist()
            start += n
    out = {k: rng.permutation(np.array(v)) for k, v in out.items()}
    assert_disjoint(out)
    return out


def assert_disjoint(splits):
    """Raise if any example index appears in two splits."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(np.asarray(splits[a]).tolist()) & set(np.asarray(splits[b]).tolist())
            if both:
                raise AssertionError(f"splits '{a}' and '{b}' share {len(both)} examples")


def ece(probs, y, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins. probs: (N, K) probabilities."""
    probs = np.asarray(probs)
    y = np.asarray(y)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_points(probs, y, bins: int = 10, min_count: int = 5):
    """(mean confidence, accuracy) per bin, for a reliability diagram. Bins with fewer than min_count examples are skipped."""
    probs = np.asarray(probs)
    y = np.asarray(y)
    conf, correct = probs.max(1), (probs.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= min_count:
            pts.append((float(conf[m].mean()), float(correct[m].mean())))
    return pts


def time_model(model, x, n_warm: int = 30, n_rep: int = 300, threads: int = 1):
    """Time one forward pass of a PyTorch model on input x. Reports milliseconds.

    One thread by default, a warm-up before timing, and many repeats, because the first calls include one-off
    set-up and a single timing says nothing about spread. Report the batch size and thread count with the numbers.
    Returns {"median_ms", "p95_ms", "n_rep", "threads", "batch"}.
    """
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    ts = []
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        for _ in range(n_rep):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(old)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "n_rep": n_rep, "threads": threads, "batch": int(x.shape[0])}


_PEAK_SCRIPT = """
import sys, threading, time, torch, psutil
torch.set_num_threads(1)
torch.nn.functional.conv2d(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 3, 3))   # initialise the runtime first
proc = psutil.Process(); base = proc.memory_info().rss; peak = [base]; stop = [False]
def sample():
    while not stop[0]:
        peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.0005)
th = threading.Thread(target=sample); th.start()
m = torch.load(sys.argv[1], weights_only=False); m.eval()
x = torch.randn(*[int(v) for v in sys.argv[2].split(",")])
with torch.no_grad():
    for _ in range(30): m(x)
stop[0] = True; th.join()
print((peak[0] - base) / 1e6)
"""


def peak_memory_mb(model, input_shape, path: str = "_tmp_model.pt") -> float:
    """Peak growth in resident memory (MB) of a fresh process that loads the whole model and runs inference.

    A fresh process makes the number repeatable. It includes the weights. The model must be picklable with
    torch.save(model) (define your classes in a module, not inside the notebook). input_shape includes the batch
    dimension, for example (1, 3, 32, 32). The temporary file is removed afterwards.
    """
    import torch
    torch.save(model, path)
    try:
        out = subprocess.check_output([sys.executable, "-c", _PEAK_SCRIPT, path, ",".join(str(v) for v in input_shape)])
    finally:
        if os.path.exists(path):
            os.remove(path)
    return float(out.decode().strip())


def mean_sd(values):
    """Mean and sample standard deviation of per-seed values. Report the per-seed values as well, never only these."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def save_table(rows, name: str, folder: str = "results") -> str:
    """Save a list of dicts as CSV and JSON in the results folder, so that every table in the report is
    produced by code and can be compared with the report. Returns the CSV path."""
    import csv
    os.makedirs(folder, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(folder, f"{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return csv_path
