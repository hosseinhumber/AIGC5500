"""Run with: python -m pytest tests  (or: python tests/test_utils.py). Add your own tests for your own code."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import four_way_split, assert_disjoint, ece, mean_sd


def test_splits_are_disjoint_and_sized():
    y = np.repeat(np.arange(4), 100)
    s = four_way_split(y, {"train": 40, "holdout": 20, "val": 20, "test": 20}, seed=1, train_cap={3: 5})
    assert_disjoint(s)
    assert len(s["train"]) == 40 * 3 + 5
    assert all(len(s[k]) == 80 for k in ("holdout", "val", "test"))
    assert (y[s["train"]] == 3).sum() == 5 and (y[s["val"]] == 3).sum() == 20


def test_split_is_reproducible():
    y = np.repeat(np.arange(3), 50)
    a = four_way_split(y, {"train": 20, "holdout": 10, "val": 10, "test": 10}, seed=7)
    b = four_way_split(y, {"train": 20, "holdout": 10, "val": 10, "test": 10}, seed=7)
    assert all((a[k] == b[k]).all() for k in a)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    assert ece(perfect, y) == 0.0
    wrong = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
    assert abs(ece(wrong, y) - 0.5) < 1e-9


def test_mean_sd():
    m, s = mean_sd([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(s - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
