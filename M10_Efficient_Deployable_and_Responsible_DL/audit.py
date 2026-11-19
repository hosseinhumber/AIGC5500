"""Bias audit tools and a model card generator (Part 5)."""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_logistic(X, y, epochs=400, lr=0.05, seed=0, l2=1e-3):
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(lin(((X - mu) / sd)).squeeze(1), y) + l2 * (lin.weight ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    def predict_proba(Z):
        Z = torch.as_tensor(Z, dtype=torch.float32)
        with torch.no_grad(): return torch.sigmoid(lin(((Z - mu) / sd))).squeeze(1).numpy()
    predict_proba.coef = (lin.weight.detach() / sd).squeeze(0).numpy()
    return predict_proba


def group_rates(pred, y, group):
    """Selection rate, true positive rate, false positive rate and accuracy per group."""
    out = {}
    for g in np.unique(group):
        m = group == g
        p, t = pred[m], y[m]
        out[int(g)] = dict(n=int(m.sum()), selection=float(p.mean()),
                           tpr=float(p[t == 1].mean()) if (t == 1).any() else float("nan"),
                           fpr=float(p[t == 0].mean()) if (t == 0).any() else float("nan"),
                           accuracy=float((p == t).mean()))
    return out


def gaps(rates):
    a, b = rates[0], rates[1]
    return dict(demographic_parity_gap=abs(a["selection"] - b["selection"]),
                tpr_gap=abs(a["tpr"] - b["tpr"]), fpr_gap=abs(a["fpr"] - b["fpr"]),
                disparate_impact_ratio=min(a["selection"], b["selection"]) / max(a["selection"], b["selection"]))


def proxy_leakage(X, group, col, seed=0):
    """How well one column predicts the protected group (accuracy of a threshold rule, 0.5 = no leakage with balanced groups)."""
    x = X[:, col]; best = 0.0
    for t in np.quantile(x, np.linspace(0.05, 0.95, 37)):
        acc = max(((x > t) == group).mean(), ((x <= t) == group).mean()); best = max(best, acc)
    base = max(group.mean(), 1 - group.mean())
    return float(best), float(base)


def threshold_for_rate(scores, rate):
    return float(np.quantile(scores, 1 - rate))


def per_group_thresholds(scores, group, target_selection):
    """Group-specific thresholds that give each group the same selection rate (a post-processing mitigation)."""
    return {int(g): threshold_for_rate(scores[group == g], target_selection) for g in np.unique(group)}


def apply_thresholds(scores, group, th):
    return np.array([scores[i] > th[int(group[i])] for i in range(len(scores))]).astype(int)


def model_card(name, intended_use, data, metrics, subgroup, limitations, extra=None):
    """Return the model card as a Markdown string. `metrics` and `subgroup` are plain dicts of numbers."""
    L = [f"# Model card: {name}", "", "## Intended use", intended_use, "", "## Data", data, "", "## Headline metrics"]
    for k, v in metrics.items(): L.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")
    L += ["", "## Results by subgroup"]
    for g, r in subgroup.items():
        L.append(f"- group {g}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    L += ["", "## Known limitations", limitations]
    if extra:
        L += ["", "## Notes", extra]
    return "\n".join(L) + "\n"
