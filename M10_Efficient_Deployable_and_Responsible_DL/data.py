"""The 8 by 8 digits used throughout Module 10, and a synthetic lending data set for the bias audit."""
import numpy as np
import torch
from sklearn.datasets import load_digits


def load_digits_split(seed=0, test_fraction=0.2):
    """Return (X_train, y_train, X_test, y_test). Pixels are scaled to [0, 1] and flattened to 64 values."""
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    te, tr = idx[:n_test], idx[n_test:]
    return torch.tensor(X[tr]), torch.tensor(y[tr]), torch.tensor(X[te]), torch.tensor(y[te])


def lending_data(n=6000, seed=0, label_bias=1.0, proxy_strength=0.8):
    """A synthetic loan data set in which the truth is known.

    Each applicant has a hidden creditworthiness, a group (0 or 1), and observed features. The group is
    NOT given to the model as a feature, but `neighbourhood` is a proxy for it. Historical approvals
    (the label we train on) were biased against group 1 by `label_bias` units of credit score, whereas
    the real repayment outcome depends only on creditworthiness.
    Returns a dict of numpy arrays: X, group, y_hist (biased label), y_true (repayment), names.
    """
    rng = np.random.default_rng(seed)
    group = (rng.random(n) < 0.4).astype(int)
    credit = rng.normal(0.0, 1.0, n)                        # same distribution in both groups
    income = 0.8 * credit + rng.normal(0, 0.6, n)
    history = 0.7 * credit + rng.normal(0, 0.7, n)
    debt = -0.5 * credit + rng.normal(0, 0.8, n)
    noise = rng.normal(0, 1, n)
    neighbourhood = proxy_strength * (group - 0.4) / 0.49 + np.sqrt(1 - proxy_strength ** 2) * rng.normal(0, 1, n)
    X = np.stack([income, history, debt, neighbourhood, noise], 1).astype(np.float32)
    y_true = (credit + rng.normal(0, 0.5, n) > 0).astype(int)           # repays the loan
    y_hist = (credit - label_bias * group + rng.normal(0, 0.5, n) > 0).astype(int)  # biased past approvals
    return dict(X=X, group=group, y_hist=y_hist, y_true=y_true,
                names=["income", "payment history", "debt", "neighbourhood", "noise"])
